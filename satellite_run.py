# -*-coding:utf-8-*-
from collections import UserList
import numpy as np
import pandas as pd
from user import *
# from beam_init import *
from LEOSatellite import *
import matplotlib.pyplot as plt  # 约定俗成的写法plt
from gym.spaces import Tuple, MultiDiscrete, Box
# from MaxCI import *
# from core import *
from calculateSinr import *
from gym.spaces import Box

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
np.set_printoptions(linewidth=400)
data_low, data_high = -1, 1  # 归一化后的范围示例

def normalize(data, min_val, max_val, epsilon=1e-8):
    return 2 * (data - min_val) / (max_val - min_val + epsilon) - 1  # 归一化到[-1, 1]


class Env:
    def __init__(self):
        self.beam, self.lat_log = load_tle('./starlink_1.txt',0,0)
        self.beam_number = len(self.beam)
        self.maxdistance = setting.satellite_maxdistance/2  # 距离中心点最大距离 波束半径
        self.user_per_beam = setting.user_per_beam
        self.rbgnumber = setting.satellite_rbg_per_beam
        self.user_number = setting.user_per_beam * setting.satellite_beams
        self.power_discrete = np.array([10,15,20,25])  ## ？
        self.beam_list = list(range(0, self.beam_number, 1))
        self.userlist = 0
        self.request_list = 0
        self.max_server_user = self.beam_number * self.rbgnumber
        self.max_server_beam = setting.satellite_beams  # self.max_server_user // setting.user_per_beam  # 非跳波束场景，不使用
        self.tti = 1
        self.cqi = np.random.randint(15, 16, size=self.max_server_user)
        self.sbcqi = np.random.randint(15, 16, size=(self.max_server_user, self.rbgnumber))
        self.RbgMap = np.zeros((self.max_server_beam, self.rbgnumber))
        self.InvFlag = np.random.randint(1, 2, size=(self.beam_number, self.user_number))  # 波束掩码矩阵
        self.bler = np.zeros(self.max_server_user)
        self.current_cqi_reqest = 0
        self.current_bler_request = 0
        self.request_position_xyz_info = 0
        # self.cellid = np.random.randint(0, 1, size=(self.user_number))
        # self.observation_space = {'Requests': (self.user_number, 15), 'RbgMap': (self.max_server_beam, self.rbgnumber),
        #                           'InvFlag': (self.max_server_beam, self.user_number)}
        self.observation_space = {'Requests': (self.user_number, 4),  # 当前用户的观测state
                                  'InvFlag': (self.max_server_beam, self.user_number),  # 用户--波束约束
                                  'RbgMap': (self.max_server_beam, self.rbgnumber)}
        self.iter = 1
        # 只做信道分配：n表信道资源，m表用户独热编码
        self.action_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(self.rbgnumber * self.max_server_beam, self.user_number + 1),  # 指定二维形状
            dtype=np.float32  # 浮点类型保证连续动作输出
        )
        # 同时输出每个信道的功率、每个信道对应的用户
        # self.action_space = {'Power': (1, self.beam_number * self.rbgnumber),
        #                      'Bandweidth': (1, self.beam_number * self.rbgnumber),}
        self.update_time = 0.001  # step为1ms
        self.extra_infor = {}
        self.last_tti_state = 0

    # 在构建S_DDPG_next之前添加归一化
    def reset(self, on, off):
        self.extra_infor = {}
        self.tti = 1
        self.bler = np.zeros(self.user_number)
        self.cqi = np.random.randint(15, 16, size=self.user_number)
        self.sbcqi = np.random.randint(15, 16, size=(self.user_number, self.rbgnumber))  # user_number*rbg_number
        self.userlist = initial_all_user(self.maxdistance, self.user_per_beam,self.lat_log, ontime=on, offtime=off)
        ########3
        for i in range(len(self.userlist)):
            self.userlist[i].model2_update(tb=0, capacity=0, time_duration=self.update_time)
        position_xyz0, position_log_lat0 = get_user_position(self.userlist, self.max_server_user)  # 有最大可服务用户限制，导出请求用户位置坐标
        S0, self.request_list = get_user_traffic_info(self.userlist, 0)
        # cat_reqandposition_xyz：返回发出请求user的[编号， [user位置*3]，[所在波束中心位置*3]，[未知含义坐标*3]，所在波束编号]
        cat_reqandposition_xyz, beam_number = userconnectsate(self.userlist, position_xyz0, self.beam, self.request_list,
                                                              self.user_number, self.max_server_user)
        self.request_position_xyz_info = cat_reqandposition_xyz
        S0['beam_number'] = beam_number  # request users对应的波束
        # TODO: step中取信道增益和干扰
        self.last_tti_state = S0
        self.InvFlag = self.generate_InvFlag(S0['beam_number'].to_numpy())  # 标记用户所在波束位置 7*35
        # print('self.InvFlag', self.InvFlag)
        S_PPO_0 = {'Requests': S0.iloc[:, 0:15].to_numpy().flatten(), 'RbgMap': self.RbgMap.flatten(),
                   'InvFlag': self.InvFlag.flatten()}
        # S_DDPG_0 = {'ChannalGain': S0['channalgain'].to_numpy().flatten(), 'Interference': S0['interference'].to_numpy().flatten(),
        #            'Thoughout': S0['last_time_txdata'].to_numpy().flatten(), 'Request': S0['waitingdata'].to_numpy().flatten()}
        # 对各个观测值应用归一化
        S_DDPG_0 = {
            'ChannalGain': normalize(S0['channalgain'].to_numpy().flatten(), np.min(S0['channalgain']), np.max(S0['channalgain'])),
            'Interference': normalize(S0['interference'].to_numpy().flatten(), np.min(S0['interference']), np.max(S0['interference'])),
            'Thoughout': normalize(S0['last_time_txdata'].to_numpy().flatten(), 0, np.max(S0['last_time_txdata'])),
            'Request': normalize(S0['waitingdata'].to_numpy().flatten(), 0, np.max(S0['waitingdata']))
        }
        S_DDPG = np.concatenate((S_DDPG_0['ChannalGain'], S_DDPG_0['Interference'], S_DDPG_0['Thoughout'], S_DDPG_0['Request'], self.InvFlag.flatten()), -1)
        return S0, S_DDPG


    def step(self, action=0):
        self.extra_infor = {}
        last_time_request = self.request_list
        action = self.reshape_act_tensor(action, last_time_request)  # [35, 10] 35个用户分别获得所在波束的哪些资源
        #############根据上一时刻采取动作更新下一时刻的bler和cqi#########################
        # request_position_xyz_info：返回发出请求user的[编号， [user位置*3]，[所在波束中心位置*3]，[未知含义坐标*3]，所在波束编号]
        tb_list, rbg_list, capa, channalgain, interfer = get_tx(action, self.request_position_xyz_info)
        ####################################
        position_xyz, position_log_lat, next_state, self.request_list = updata(self.userlist, tb_list,
                                                                               last_time_request, capa, channalgain, interfer)
        # print("num of user's request", len(self.request_list))
        # satuedict, Beam, UeLinkSate = userconnectsate(position_xyz, epoch, self.tti)
        # cat_reqandposition_xyz, beam_number = self.deal_data(Beam, self.request_list, position_xyz, satuedict)
        cat_reqandposition_xyz, beam_number = userconnectsate(self.userlist, position_xyz, self.beam, self.request_list,
                                                              self.user_number, self.max_server_user)
        # TODO:request_list为空处理
        self.request_position_xyz_info = cat_reqandposition_xyz
        next_state['beam_number'] = beam_number
        self.last_tti_state.iloc[:, 8] = next_state.iloc[:, 8] # 第8列表示current_tx_data
        self.extra_infor = self.generate_extra_info(self.last_tti_state, rbg_list, last_time_request, tb_list)
        self.last_tti_state = next_state
        self.InvFlag = self.generate_InvFlag(next_state['beam_number'].to_numpy())
        done = False
        # S_DDPG_0 = {'ChannalGain': next_state['channalgain'].to_numpy().flatten(),
        #             'Interference': next_state['interference'].to_numpy().flatten(),
        #             'Thoughout': next_state['last_time_txdata'].to_numpy().flatten(),
        #             'Request': next_state['waitingdata'].to_numpy().flatten()}
        # 对各个观测值应用归一化
        S_DDPG_0 = {
            'ChannalGain': normalize(next_state['channalgain'].to_numpy().flatten(), np.min(next_state['channalgain']), np.max(next_state['channalgain'])),
            'Interference': normalize(next_state['interference'].to_numpy().flatten(), np.min(next_state['interference']), np.max(next_state['interference'])),
            'Thoughout': normalize(next_state['last_time_txdata'].to_numpy().flatten(), 0, np.max(next_state['last_time_txdata'])),
            'Request': normalize(next_state['waitingdata'].to_numpy().flatten(), 0, np.max(next_state['waitingdata']))
        }
        S_DDPG_next = np.concatenate(
            (S_DDPG_0['ChannalGain'], S_DDPG_0['Interference'], S_DDPG_0['Thoughout'], S_DDPG_0['Request'], self.InvFlag.flatten()), -1)
        # tb_list作为reward，reward单位Mbps
        return next_state, S_DDPG_next, sum(tb_list)/1e6, done

    def generate_extra_info(self, state, rbg_list, req, tb_list):
        beam_user_connectlist = state['beam_number'].to_numpy()
        user_rbgbumber_dict = dict(zip(req, rbg_list))
        #print('req', req)
        #print('rbg_list', rbg_list)
        #print('tb_list', tb_list)
        for i in range(int(max(beam_user_connectlist))):
            enb_info = state[state['beam_number'] == i + 1]
            # print("state[beam_number]", state["beam_number"])
            if enb_info.empty:
                continue
            else:
                index = np.where(beam_user_connectlist == i + 1)
                rbg_number_used = 0
                enb_req_total = len(index[0])
                unassigned_total = 0
                enb_rbg_list = []
                for j in index[0]:
                    rbg_number_used += user_rbgbumber_dict[j]
                    enb_rbg_list.append(user_rbgbumber_dict[j])
                    if user_rbgbumber_dict[j] == 0:
                        unassigned_total += 1
                self.extra_infor['enb' + str(i + 1)] = {'enb':i+1,'enb_req_total': enb_req_total,
                                                        'unassigned_total': unassigned_total,
                                                        'number_of_rbg_nedded': enb_info['number_of_rbg_nedded'].sum(),
                                                        'rbg_used': rbg_number_used,
                                                        'newdata': enb_info['newdata'].sum(),
                                                        'waitingdata': enb_info['waitingdata'].sum(),
                                                        'last_time_txdata': enb_info['last_time_txdata'].sum(),
                                                        # 'time_duration': enb_info['time_duration'].sum(),
                                                        'total_txdata': enb_info['total_txdata'].sum(),
                                                        'average_throughput': enb_info['average_throughput'].sum(),
                                                        'rbg_usable': self.rbgnumber,'capacity':enb_info['capacity'].sum()}
        # print(self.extra_infor)
        return self.extra_infor

    def printposition_xyz(self):
        for i in range(len(self.userlist)):
            print('user{0} position_xyz{1}'.format(i, self.userlist[i].position_xyz))

    def generate_InvFlag(self, data):
        flag = np.random.randint(1, 2, size=(self.max_server_beam, self.user_number))
        for i in range(self.max_server_beam):
            b = np.where(data == i + 1)
            flag[i][b] = 0
        return flag

# 只保留发出request用户的子频段资源矩阵，其余设为0？
    def reshape_act_tensor(self, act, request_list):
        act = np.where(act.reshape(self.max_server_user, -1) == 1)[1]
        act_matrix = np.zeros((self.user_number, self.rbgnumber), dtype='int64')
        # assert act.shape[0] == 1, "act维度不为(x,)"
        for i in request_list:
            index = np.where(act == i + 1)[0]  # 资源1-70，哪些资源分给用户request_list[i]
            # index = index[0]  # 360个频段中第index个分给用户request_list[i]，index可能不只一个
            for y in index:
                # if y / self.rbgnumber == 
                act_matrix[i][y % self.rbgnumber] = 1  # index[y] % self.rbgnumber操作将30*12-->12
                # TODO: 
                # 判断用户i是否在波束y/self.rbgnumber
        # print('power action',act)
        # print("act_matrix shape :",np.shape(act_matrix))
        return act_matrix  # 生成发出请求用户获得的子频段资源矩阵


if __name__ == '__main__':
    env = Env()
    on = 10
    off = 1
    S0, _ = env.reset(on, off)
    beam_new = np.zeros(12)

    for i in range(100):
        new_ = []
        action = np.zeros(72)
        next_s, _, _, _ = env.step(0, 0, action)
        for i in range(12):
            beam_new = next_s[next_s['beam_number'] == i + 1]['newdata'].sum()
            new_.append(beam_new)
        new_ = np.array(new_)
        beam_new = beam_new + new_
    c = beam_new.sum()
    print(c)
    d = beam_new / c
    print(d)
    power = 3794.7331922020558
    beam_power = d * power
    print(beam_power)
    db_beam = 10 * np.log10(beam_power)
    print(db_beam)
    print(beam_new)