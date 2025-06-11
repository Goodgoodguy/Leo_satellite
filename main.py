from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core3 as core
from spinup.utils.logx import EpochLogger
from ddpg import ddpg
from spinup.utils.run_utils import setup_logger_kwargs
import argparse

import satellite_run
import os
global device  # = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--exp_name', type=str, default='ddpg')
    # args = parser.parse_args()

    trace_dir = os.getcwd() + "/result"

    logger_kwargs = setup_logger_kwargs("ddpg-ra", data_dir=trace_dir, datestamp=True)
    # 问题：r = 0
    ddpg(satellite_run,
         actor_critic=core.DIS_ActorCritic,
         ac_kwargs=dict(hidden_sizes=(128,256,128)),
         steps_per_epoch=50,  # 每一轮的步数
         epochs=1000,  # 训练轮数
         replay_size=int(1e4),
         gamma=0.99,  # 折扣因子
         pi_lr = 1e-3, q_lr = 1e-4,
         batch_size=256,
         start_steps=1e9,  # 收集多少随机动作
         act_noise=0.2,
         polyak=0.999,  # 软更新因子
         update_after=1e9,  # 收集多少random step之后开始更新
         update_every=1e9,  # 多少步更新一次
         max_ep_len = 50,
         logger_kwargs=logger_kwargs)
