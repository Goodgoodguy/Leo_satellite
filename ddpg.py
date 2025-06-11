from copy import deepcopy
import random
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core3 as core
from spinup.utils.logx import EpochLogger
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
from read_res import draw_result_net, draw_result_ret
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class OUNoise:
    """Ornstein-Uhlenbeck process."""
 
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
 
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()
 
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state


def ddpg(env_fn, actor_critic=core.DIS_ActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 环境参数
    ontime = 1
    offtime = 9

    env, test_env = env_fn.Env(), env_fn.Env()
    obs_dim = np.prod(env.observation_space["Requests"])
    inv_dim = np.prod(env.observation_space["InvFlag"])
    # 波束资源选择用户，net输出为[beam*rbg，user]
    fin_act_dim = env.action_space.shape[0]  # 环境交互动作维度
    act_dim = np.prod(env.action_space.shape)  # 30个波束*12个带频段， 1200个用户从卫星角度出发需要完成的决策：30个波束中12个子频段的分配情况
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim+inv_dim, act_dim=act_dim, size=replay_size)

    # 初始化OU噪声生成器
    ou_noise = OUNoise(size=act_dim)  # 动作空间维度作为size参数

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o_info, a, r, o2_info, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = o_info[:, :obs_dim]
        o = torch.as_tensor(o, dtype=torch.float32).to(device)
        a = torch.as_tensor(a).to(device)

        
        q = ac.q(o,a)

        # Bellman backup for Q function 使用ac_targ不带梯度
        with torch.no_grad():
            # 需要取flag
            fla2 = o2_info[:, -inv_dim:]
            fla2 = torch.as_tensor(fla2, dtype=torch.int32).to(device)
            o2 = o2_info[:, :obs_dim]
            o2 = torch.as_tensor(o2, dtype=torch.float32).to(device)

            r = torch.as_tensor(r, dtype=torch.float32).to(device)
            d = torch.as_tensor(d, dtype=torch.float32).to(device)

            a = onehot_from_logits(fin_act_dim, ac_targ.act(o2, fla2))
            q_pi_targ = ac_targ.q(o2, a)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o_info = data['obs']
        o = o_info[:, :obs_dim]
        o = torch.as_tensor(o, dtype=torch.float32).to(device)
        fla = o_info[:, -inv_dim:]
        fla = torch.as_tensor(fla, dtype=torch.int32).to(device)
    
        # 获取策略网络输出的logits（带梯度）
        a_logtis = ac.act(o, fla)  # 假设这里输出的是未归一化的logits
        a = get_categorical(fin_act_dim, a_logtis)

        q_pi = ac.q(o, a)

        
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # 检查策略网络参数是否可训练
    for param in ac.pi.parameters():
        assert param.requires_grad, "Policy network parameters should require gradients."

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    
    def onehot_from_logits(act_dim, logits, eps=0.01):
        logits = logits.reshape(logits.shape[0], act_dim, -1)
        logits = torch.softmax(logits, dim=-1)
        #  生成最优动作的独热（one-hot）形式
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
        # 生成随机动作,转换成独热形式
        # rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        #     np.random.choice(range(logits.shape[1]), size=logits.shape[0])
        # ]],
        #                                 requires_grad=False).to(logits.device)
        # 通过epsilon-贪婪算法来选择用哪个动作
        return argmax_acs.reshape(logits.shape[0], -1)
        # return torch.stack([
        #     argmax_acs[i] if r > eps else rand_acs[i]
        #     for i, r in enumerate(torch.rand(logits.shape[0]))
        # ])
    def get_categorical(act_dim, logits, temperature=0.5):
        # 使用Gumbel-Softmax重参数化近似离散采样（保留梯度）
        # temperature控制软化程度，随训练逐步降低（可选）  
        # gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        # y = (logits + gumbel_noise) / temperature
        # act_logits = torch.softmax(y, dim=-1)  # 软化后的"连续动作"近似离散采样
        # y_hard = onehot_from_logits(fin_act_dim, act_logits)
        # act = (y_hard.to(logits.device) - y).detach() + y
        
        logits = logits.reshape(logits.shape[0], act_dim, -1)
        act = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1) 
        # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
        # 正确地反传梯度
        return act.reshape(logits.shape[0], -1)
    
    def get_action(o_info, noise_scale):
        fla = o_info[-inv_dim:]
        fla = torch.as_tensor(fla, dtype=torch.int32).reshape(-1, inv_dim)
        obs = o_info[:obs_dim]
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1, obs_dim)

        # a, _ = ac.step(obs, fla)
        a_logtis = ac.act(obs, fla)
        # noise = ou_noise.sample().rehape(a_logtis.shape)
        # a_logtis += noise_scale * noise
        a = get_categorical(fin_act_dim, a_logtis)
        return a.cpu().detach().numpy()
    
    def get_randmo_action(o_info):
        fla = o_info[-inv_dim:]
        fla = torch.as_tensor(fla, dtype=torch.int32).reshape(-1, inv_dim)
        temp = fla.int().reshape( env.beam_number, -1)  # batch_size*可服务波束数量*user总数
        am1 = temp.unsqueeze(1).expand(-1, env.rbgnumber, -1)  # batch_size*可服务波束数量*子频带数量*user总数
        am1 = am1.reshape(-1, env.user_number)  # batch_size*（可服务波束数量*子频带数量）*当前服务user总数
        am2 = torch.zeros((*am1.shape[:-1], 1), dtype=torch.int, device=am1.device)  # 不分的一列  生成一个batch_size*（可服务波束数量*子频带数量）放在最后
        amask = torch.cat((am2, am1), 1).bool()  # 形成当前服务user总数+1

        a_logits = torch.as_tensor(env.action_space.sample(), dtype=torch.float32)  # tensor [1, 2520]
        a_logits = a_logits.masked_fill_(amask, -np.inf).flatten().unsqueeze(0)  # 子信道维度mark|波束维度mark
        a = onehot_from_logits(fin_act_dim, a_logits)
        return a.numpy()


    def test_agent():
        for j in range(num_test_episodes):
            _, o_info = test_env.reset(ontime, offtime)
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                # a = get_action(o_info, 0)
                a = get_randmo_action(o_info)
                _, o_info, r, d = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    _, o_info = env.reset(ontime, offtime)
    fig, ax = plt.subplots()
    ax.set_title("LEO Train")
    x = []
    y = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if(t % steps_per_epoch == 0):
            pbar = tqdm(total=steps_per_epoch)
            # upper_per_rbg = setting.rbgcapa
            ac.eval()
            ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler, ep_rbg_used = 0, 0, 0, 0, 0, 0, 0, 0
            epoch_tx, epoch_capacity, epoch_waiting, epoch_reward, epoch_newbytes, epoch_bler, epoch_rbg_used = 0, 0, 0, 0, 0, 0, 0
            ep_r1, ep_r2 = 0, 0
            sum_tx = 0
            error = 0
            final_waiting = 0
            start_time = time.time()

        # TODO：什么时候使用随机采样
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_randmo_action(o_info) if random.random() < 0.01 else get_action(o_info, act_noise)
        else:
            # 随机采样作softmax？？
            a = get_randmo_action(o_info)

        # Step the env
        _, o2_info, r, d = env.step(a)
        
        # TODO: 处理reward，当前reward为总吞吐量
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o_info, a, r, o2_info, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o_info = o2_info
        pbar.update(1)

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            _, o_info = env.reset(ontime, offtime)
            d, ep_ret, ep_len = False, 0, 0
            fla = o_info[-inv_dim:]
            fla = torch.as_tensor(fla, dtype=torch.int32)
            obs = o_info[:obs_dim]
            obs = torch.as_tensor(obs, dtype=torch.float32)

        # TODO:
        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            if 'QVals' not in logger.epoch_dict or len(logger.epoch_dict['QVals']) == 0:
                logger.store(QVals=0)
                logger.store(LossPi=0)
                logger.store(LossQ=0)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            if (t+1) % steps_per_epoch * 10 == 0:
                draw_result_ret(logger.output_dir)
                if t >= update_after:
                    draw_result_net(logger.output_dir)

        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
