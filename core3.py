import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from jedi.inference.value.instance import SelfName

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act(),
        nn.BatchNorm1d(sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# 策略网络类
class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)
    
# 离散action的actor
# base class
class Actor(nn.Module):

    def _getOutputLogits(self, obs, invFlag):
        raise NotImplementedError

    def forward(self, obs, invFlag, act=None):
        pi = self._getOutputLogits(obs, invFlag)
        # 留个接口
        # if act is not None:
        #     logp_a = self._log_prob_from_distribution(pi, act)
        #     print('logp_a',logp_a)
        return pi
# sub class
class MultiCategoricalActor(Actor):

    def __init__(self, env, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.max_req = env.user_number
        self.enb_cnt = env.max_server_beam
        self.rbg_cnt = env.rbgnumber
        self.actdim = act_dim
        self.obs_dim = obs_dim
        self.out_dim = act_dim
        # self.logits_net = mlp([obs_dim] + [1024]+[512]+ list(hidden_sizes) + [self.out_dim], activation)
        self.logits_net = mlp([obs_dim] +list(hidden_sizes) + [self.out_dim], activation)
        # print('非循环网络结构')
        # print("net structure:", self.logits_net)

    # obs1200个用户的观测数据15，rbgMap30个波束12个子频段使用情况，invFlag30个波束内1200分布&使用情况
    def _getOutputLogits(self, obs, invFlag):
        # assert len(obs.shape) < 3
        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        # batch_size = 1
        # 根据rbgMap构造mask
        # rm1 = rbgMap.int().reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, self.max_req)
        # rm2 = torch.zeros((*rm1.shape[:-1], 1), dtype=torch.int, device=rm1.device)
        # rmask = torch.cat((rm2, rm1), 2).bool()

        temp = invFlag.int().reshape(batch_size, self.enb_cnt, -1)  # batch_size*可服务波束数量*user总数
        am1 = temp.unsqueeze(2).expand(-1, -1, self.rbg_cnt, -1)  # batch_size*可服务波束数量*子频带数量*user总数
        am1 = am1.reshape(batch_size, -1, self.max_req)  # batch_size*（可服务波束数量*子频带数量）*当前服务user总数
        am2 = torch.zeros((*am1.shape[:-1], 1), dtype=torch.int, device=am1.device)  # 不分的一列  生成一个batch_size*（可服务波束数量*子频带数量）放在最后
        amask = torch.cat((am2, am1), 2).bool()  # 形成当前服务user总数+1

        # inp = torch.cat((obs, rbgMap.float(), invFlag.float()), 0 if len(obs.shape) == 1 else 1)
        logits = self.logits_net(obs)
        # print("logit shape:", logits.shape)
        logits = logits.reshape(amask.shape)
        # logits = logits.masked_fill_(rmask | amask, -np.inf)  # 子信道维度mark|波束维度mark
        logits = logits.masked_fill_(amask, -np.inf)  # 子信道维度mark|波束维度mark
        # print(logits.shape)
        # input()
        # return Categorical(logits=logits).sample()
        return logits.reshape(batch_size, -1)

# 评价网络类
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class DIS_ActorCritic(nn.Module):

    def __init__(self, env,
                 hidden_sizes=(256,256),
                 activation=nn.Tanh, use_cuda=True):
        super().__init__()

        observation_space = env.observation_space
        action_space = env.action_space
        obs_dim = np.prod(observation_space['Requests'])
        act_dim = np.prod(action_space.shape)
        # q_act_dim = action_space.shape[0]
        self.use_cuda = use_cuda
        self.obs_dim = obs_dim

        # build policy and value functions
        self.pi = MultiCategoricalActor(env, obs_dim, act_dim, hidden_sizes, activation)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        if use_cuda:
            self.pi = self.pi.to(device)
            self.q = self.q.to(device)

    def step_with_grad(self, obs, fla):
        if self.use_cuda:
            obs = obs.to(device)
            fla = fla.to(device)
            pi = self.pi._getOutputLogits(obs, fla)
        return pi
        
    def act(self, obs, fla):
        if self.use_cuda:
            obs = obs.to(device)
            fla = fla.to(device)
        # with torch.no_grad():
            pi = self.pi._getOutputLogits(obs, fla)
        
        # if self.use_cuda:
        #     return pi.cpu()
        # else:
        #     return pi
        return pi
            