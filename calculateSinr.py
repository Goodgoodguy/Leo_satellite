# -*-coding:utf-8-*-
import numpy as np
import setting
import math as ma
import scipy.special as sc

power = False

# func below are used to form shadow rician
def PochhammerSymbol(x,n):
    if n == 0:
        y = 1
    else:
        y = 1
        for k in range(1, n + 1):
            y = y*(x + k - 1)
    return y

def Kummer(a,b,z,maxit):
    # Implementation
    ytemp = 1
    for k in range(1, maxit + 1):
        ytemp = ytemp \
                + PochhammerSymbol(a,k)/(PochhammerSymbol(b,k) \
                * ma.factorial(k))*z^k
        y = ytemp
    return y

# ku波段用户链路,暂时未找到完整的参数
class calculate_tool:
    def __init__(self):
        # power_action=np.array([22.87974038,24.93794295,26.5090619,25.46126201,25.08990887,23.35311763,21.91114878,24.91139132,23.91362996,27.19319921,26.58206721,24.14658395])
        # power_action=np.array([24.51073719,24.90290911,24.71413423,24.62848788,25.73554529,23.93189764,26.35381523,25.46034243,23.79002661,25.29328968,25.71562453,24.16128749])
        # power_action = np.array([25.00950944,24.61750836,25.55550514,24.70035181,24.78942288,24.10023297,25.62994134,25.02212127,23.54360225,25.48291179,26.37034709,24.45415818])
        power_action = np.random.randint(20, 21, size=100)
        # self.Gt_sate = 38.5 # dbi
        self.Gr_user = 0 # 未确定数值
        self.loss_path = -180.1  # DB,未确定数值
        self.PowerT_beam = power_action
        self.user_number = setting.user_per_beam * setting.satellite_beams
        self.Hleo = setting.satellite_height
        # self.G_peak = 65 - self.PowerT_beam  11.15之前是65
        self.G_peak = 55 - self.PowerT_beam
        self.ones = np.repeat(0.2524, 100)
        self.gama = 0.5 # 天线效率
        self.noisy = 2.5118864315095823e-12  # TODO：-174 dBm·Hz−1
        self.bw = setting.bw
        self.rbg_number = setting.satellite_rbg_per_beam
        self.beam_numbers = setting.satellite_beams
        self.rb_bw = self.bw / self.rbg_number
        # 以子带为单位分配功率
        self.power_subbw = np.random.randint(140, 141, size=self.beam_numbers*self.rbg_number)
        self.angle = 0
        self.frequency = setting.frequency
        self.c = 3e8

    """
    input:
        b = Scalar (real), Average power of multipath component
        m = Scalar (real), Fading severity parameter
        Omega = Scalar (real), Average power of LOS component
        x, random number generated

    output:
        f, PDF
        F, CDF
    """
    def ShadowedRicianDistribution(self,b,m,Omega,x):
        lambd = 1/(2*b)
        alpha = (2*b*m)/(2*b*m + Omega)
        beta = Omega/(2*b*(2*b*m + Omega))

        # Theoretical PDF
        f = []
        for k in x:
            f.append(((alpha^m)*lambd*ma.exp(-x*lambd)*Kummer(m,1,beta*k)))

        # Theoretical CDF
        sumk = []
        F = []
        for p in x:
            for q in range(len(x)):
                mmk = ma.gamma(m+k)
                mk = ma.gamma(m)
                betabylambdak = (beta/lambd)^k
                gammak = sc.gammaincc(k+1,lambd*p)
                sumk.append = (mmk/(mk*(ma.factorial(k))^2))*betabylambdak*gammak
            F.append(alpha*sum(sumk))
        return f, F
    
    '''
    自由路径损失计算 user_i所在波束passloss，单位：dB

        天线增益矩阵计算user_i与user_j所在波束天线接收增益，单位：dB
    '''
    def get_GainAndPL(self, position_info):
        # 矩阵为全体用户矩阵
        beam_number_connect = self.user_number  # 发出request的用户list
        theta_matrix = np.zeros((self.user_number, self.user_number))  # 角度矩阵[i][j]user_i与user_j所在波束中心的倾角
        distance_matrix = np.zeros((self.user_number, self.user_number))  # 距离矩阵[i][j]user_i与user_j所在波束中心的距离
        Gain_matrix = np.zeros((self.user_number, self.user_number), dtype=np.float32)  # 天线增益矩阵[i][j]user_i与user_j所在波束天线增益
        PathLoss_matrix = np.zeros((self.user_number, self.user_number), dtype=np.float32)  # 天线增益矩阵[i][j]user_i与user_j所在波束天线增益
        for i in range(self.user_number):
            # request_users中用户i索引，获取所在波束位置
            if position_info.size == 0:
                continue
            request_users = np.where(position_info[:, 0] == i)[0]
            if request_users.size == 0:
                continue
            
            user_position = position_info[request_users[0]][1:4]  # position_info的排序方式同request_users
            for j in range(self.user_number):
                if position_info.size == 0:
                    continue
                other_request_users = np.where(position_info[:, 0] == i)[0]
                if other_request_users.size == 0:
                    continue
                beam_position_j = position_info[other_request_users[0]][4:7]  # 用户j所在波束位置
                beam_label = int(position_info[other_request_users[0]][-1])  # 用户j所在波束编号
                # U_i与U_j所在波束中心坐标距离-->影响U_i的channalgain[i]
                distance = np.sqrt(np.sum((user_position - beam_position_j) ** 2))
                distance_matrix[i][j] = distance  # 用户i到用户j所在的波束中心距离
                theta = np.degrees(np.arctan(distance / self.Hleo))  # 单位deg
                theta_matrix[i][j] = theta  # 用户i到用户j所在的波束的波束中心轴线夹角
                # 用户j所在波束对用户i的波束增益，dB
                Gain_matrix[i][j] = self.G_peak[beam_label - 1] - (
                        (12 * (10 ** (self.G_peak[beam_label - 1] / 10))) / self.gama) * np.square(
                    theta_matrix[i][j] / (70 * np.pi))
                
                # U_i与U_j所在波束中心坐标距离-->影响U_i的channalgain[i]
                distance_ub = np.sqrt(np.sum((user_position - beam_position_j) ** 2))
                distance_us = np.sqrt(distance_ub ** 2 + self.Hleo ** 2)  # 用户与卫星距离
                PathLoss_matrix[i][j] = ma.pow(self.c / (4 * ma.pi * distance_us * self.frequency), 2)
                if (Gain_matrix[i][j] < 0):
                    Gain_matrix[i][j] = 0
        # Gain_matrix = 10 ** (Gain_matrix / 10)  # 倍数
        self.angle = np.diag(theta_matrix)  # 对角元素为U_i与U_i所在波束中心的夹角
        PathLoss_matrix[PathLoss_matrix == 0] = 1
        PathLoss_matrix = 10 * np.log10(PathLoss_matrix)  # dB
        return Gain_matrix, PathLoss_matrix  # 天线增益

    def get_sinr(self, action, position_info):
        # print('action',action.shape)
        rbgnumber = action.shape[1]

        Gain_matrix, PathLoss_matrix = self.get_GainAndPL(position_info)  # 单位：dB

        sinr_matrix = np.zeros((self.user_number, rbgnumber))
        capa_matrix = np.zeros((self.user_number, rbgnumber))
        inte_matrix = np.zeros((self.user_number, rbgnumber))

        interfer_list = np.zeros(self.user_number)
        channalgain_list = np.zeros(self.user_number)

        for i in range(self.user_number):
            if position_info.size == 0:
                continue
            request_users = np.where(position_info[:, 0] == i)[0]
            if request_users.size == 0:
                continue
            for j in range(rbgnumber):
                if action[i][j] == 0:
                    continue
                else:
                    Gain_self = Gain_matrix[i][i]
                    beam_label = int(position_info[request_users[0]][10])            
                    Path_self = PathLoss_matrix[i][i]  # Path_self
                    channalgain_list[i] = Gain_self + self.Gr_user + Path_self  # dB
                    # print('Gain_self + self.Gr_user + Path_self: %f && channalgain_list: %f', Gain_self + self.Gr_user + Path_self,  channalgain_list[i])
                    
                    # power_self = (10 ** (channalgain_list[i] / 10)
                    #             * 10 ** (self.power_subbw[(beam_label-1)*self.rbg_number+j] / 10))
                    
                    power_self = (10 ** (channalgain_list[i] / 10)
                                * self.power_subbw[(beam_label-1)*self.rbg_number+j])
            
                    index = np.where(action[:, j] == 1)[0]  # 找到所有使用该子频段用户
                    # 每个波束功率恒定20
                    # print('^^^^^^^^^^^^^^^^^^^^^^^^^',power_self/self.noisy)
                    # print('power_self', power_self)
                    if len(index) == 1:  # 没有其他用户使用相同频段
                        sinr = power_self / self.noisy
                        sinr_matrix[i][j] = sinr
                        capa_matrix[i][j] = sinr
                        inte_matrix[i][j] = 0
                        continue
                    # 除当前用户i之外的，使用同一频段的用户
                    other_user_interference_index = np.delete(index, np.where(index == i)[0])
                    interfer_i_j = 0
                    # 其他用户造成的干扰计算方式
                    for user_k in other_user_interference_index:
                        if position_info.size == 0:
                            continue
                        user_k_index = np.where(position_info[:, 0] == user_k)[0]
                        if user_k_index.size == 0:
                            continue
                        beam_label_user_k = int(position_info[user_k_index[0]][-1])
                        L_p = PathLoss_matrix[i][user_k]  # dB
                        G_t = Gain_matrix[i][user_k]  # dB
                        beam_number = int(position_info[user_k_index[0]][-1])
                        # self.angle表示U_k与其所在波束中心坐标的夹角，位于1dB内不会对其他用户造成同频干扰
                        if self.angle[user_k] < self.ones[beam_number -1]:
                            inter = 0  # 判断造成干扰的角度是1dB角
                        else:  # 相互干扰的两个用户均位于onedB，如何处理？
                            # inter = (10 ** ((G_t + self.Gr_user + L_p) / 10)
                            #             * 10 ** (self.power_subbw[(beam_label_user_k-1)*self.rbg_number+j] / 10))
                            inter = (10 ** ((G_t + self.Gr_user + L_p) / 10)
                                    * self.power_subbw[(beam_label_user_k-1)*self.rbg_number+j])
                        interfer_i_j += inter

                    sinr = power_self / (self.noisy + interfer_i_j)
                    capa = power_self / self.noisy
                    # U_i在j子带的信道容量
                    sinr_matrix[i][j] = sinr
                    capa_matrix[i][j] = capa
                    inte_matrix[i][j] = interfer_i_j
            interfer_list[i] = np.sum(inte_matrix[i])
        interfer_list[interfer_list == 0]  = 1.0
        interfer_list = 10 * np.log10(interfer_list)
        # interfer_list[interfer_list == -np.inf]  = 0
        return sinr_matrix, capa_matrix, channalgain_list, interfer_list

"""
    作用: 根据上一时刻采取动作和上一时刻用户位置确定TB,RBG,信干噪比,信道容量
"""
def get_tx(action, position_info):
    tool = calculate_tool()
    sinr, capacity, channalgain, interfer = tool.get_sinr(action, position_info)
    # print(sinr)
    # slot单位ms
    tb_matrix = np.log2(sinr + 1) * tool.rb_bw / 1000
    capacity_matrix = np.log2(capacity + 1) * tool.rb_bw / 1000
    # print(tb_matrix)
    tb = np.sum(tb_matrix, axis=1)
    capa = np.sum(capacity_matrix, axis=1)
    # print('tb', tb_matrix)
    rbglist = np.sum(action, axis=1)
    return tb, rbglist, capa, channalgain, interfer




def get_beam_per_capcaity():
    tool = calculate_tool()
    label = []
    for i in range(12):
        # Gain_self = 10 * np.log10(tool.G_peak[i])
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.loss_path) / 10) * (10 ** (tool.PowerT_beam[i] / 10))
        sinr = power_self / tool.noisy
        # print(sinr)
        cap = np.log2(sinr + 1) * tool.bw / 1000
        label.append(cap)

    return label


if __name__ == '__main__':
    label = get_beam_per_capcaity()
    print(label)
