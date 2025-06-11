# -*-coding:utf-8-*-
# from sys import last_traceback
from turtle import shape
import numpy as np
import pandas as pd
import math as m
import random

import setting
from table import *


class user:
    def __init__(self, maxdistfromorigin, lat_long_areacenter, cbrrate, ontime=10, offtime=2):
        """
        :param maxdistfromorigin: 距离中心点的最大距离单位米
        :param ontime: 单位ms,指数分布的均值
        :param offtime: 单位ms,指数分布的均值
        """
        self.maxdistance = maxdistfromorigin  # user距离中心点的最大距离单位米
        self.position = np.array([0, 0, 0], dtype="float64")  # 原点地心，笛卡尔坐标系
        self.lat_long_areacenter = lat_long_areacenter
        self.log_lat_coordinates = np.array([0, 0, 0], dtype="float64")  # 用户参数：经、纬度、高度
        self.nextjingweiposision = np.array([0, 0, 0], dtype="float64")  # 更新后经纬度
        # 信道参数
        self.channalgain = 0  # 用户信道增益
        self.interference = 0  # 用户受到干扰
        # 业务信息
        self.throughput = 0  # Mbps
        self.request = 0  # 0 无请求 1 发出请求
        self.ontime = 9  # 业务持续时间
        self.offtime_restore = offtime
        self.offtime = np.random.exponential(offtime)
        self.traffictype = {'text': ontime/4, 'voice': ontime/2, 'video': ontime}  # 业务类型指数分布均值参数 
        self.qci_type = {'None': 0, 'text': 1, 'voice': 2, 'video': 3}
        self.qci = 0
        self.waiting_data_finally = 0  # 执行动作后剩余数据量
        self.waitingbit = 0  # 当前时刻待送送数据量
        self.cbrrate = cbrrate  # 表示用户平均业务量？bit/ms  >0有业务需求， <=0 没有业务需求
        self.transmit_rate_one_channal = 10
        self.newarrivaldata = 0  # 新到达数据
        self.current_txdata = 0
        self.total_txdata = 0
        self.type = None  # 业务型型
        self.number_of_rbg_nedded = 0
        self.max_number_of_rbg = g_nofRbg
        self.current_waiting_data = 0
        self.index = 0
        self.average_throughput = 0.000001  # 1bytes
        self.capacity = 0
        self.global_max_server = setting.satellite_beams * setting.satellite_rbg_per_beam
        # 随机位置
        self.movespeed = 30  # 用户移动速度 米每秒
        self.earth_radius = 6371000  # 地球半径
        # 由GRS80和WGS84定义的以米为单位的地球半长轴
        self.earth_semimajor_axis = 6378137
        # GRS80定义的地球第一偏心率
        self.earth_grs80_eccentricity = 0.0818191910428158
        # WGS84定义的地球第一偏心率
        self.earth_wgs84_eccentricity = 0.0818191908426215
        self.earthspheroidtype = {'sphere': 0, 'grs80': 1, 'wgs84': 2}  # 三种地球模型
        self.initial_random_position(self.lat_long_areacenter)  # 初始化用户位置函数
        self.movedistance = 0  # 每次更新用户移动距离
        self.randomangle = self.random_angle()  # 产生基于用户速度的经纬度的变化角度，用于用户移动随机
        # 卫星参数
        self.time_delay = 0
        self.height = setting.satellite_height  # 卫星轨道高度
        self.angle_user = 0
        self.capacity = 0
        self.generate_angele_user(self.lat_long_areacenter, self.position, self.height)  # 用户-卫星和

    # 产生基于用户速度的经纬度的变化角度，用于用户移动随机
    def random_angle(self):
        direction = np.cos(np.random.uniform(0, math.pi, size=3))
        speed = self.movespeed
        # speed = np.random.uniform(self.movespeed - 10, self.movespeed, size=3)
        randomangle = speed * direction
        randomangle = (randomangle / (2 * np.pi * self.earth_radius)) * 360
        zaxischangerate = np.random.uniform(-1, 1)
        randomangle[2] = 0

        # print(self.speed)
        # print(self.direction)
        # print(self.randomangle)
        return randomangle

    # 閺囧瓨鏌婂Ο鈥崇€�1閿涘本鐦″▎鈩冩纯閺備即鈧瀚ㄩ梾蹇旀簚閻ㄥ嫮些閸斻劍鏌熼崥鎴濇嫲鐞涘矁绻橀柅鐔哄芳
    def model1_update(self, tb, bler, cqi, time_duration=0.001):
        self.randomangle = self.random_angle()
        currentpositionxyz = self.position
        # print(self.randomangle)
        # input()
        # print(speed_vector)
        self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])
        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5
        if user_beamcenter_distance >= self.maxdistance:
            self.randomangle = -self.randomangle
            self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
            self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
            self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
            if self.log_lat_coordinates[2] < 0:
                self.log_lat_coordinates[2] = 0
            self.position = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                 self.log_lat_coordinates[1],
                                                                 self.log_lat_coordinates[2],
                                                                 self.earthspheroidtype['sphere'])
        self.position = userxyz_afterupdate
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5

        self.traffic_updata(tb, bler, cqi)

    # 更新模型2 按照恒定的行进速率和方向前进，直到到达边界，然后重新调用random_angle函数产生随机方向和距离
    def model2_update(self, tb, capacity, time_duration=0.001):
        currentpositionxyz = self.position
        self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])  # 更新后用户位置xyz
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])  # 波束xyz

        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5  # 计算更新后用户位置和波束中心的距离
        if user_beamcenter_distance <= self.maxdistance:  # 确保用户在最大约束范围内活动
            self.position = userxyz_afterupdate
        else:
            # self.position[0],self.position[1]=self.calcu_intersection(self.log_lat_coordinates)
            while (True):
                self.randomangle = self.random_angle()
                self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
                self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
                self.log_lat_coordinates[2] = self.log_lat_coordinates[2] + self.randomangle[2] * time_duration
                # print('-----------------')
                if self.log_lat_coordinates[2] < 0:
                    self.log_lat_coordinates[2] = 0
                userxyz_afterupdate2 = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                            self.log_lat_coordinates[1],
                                                                            self.log_lat_coordinates[2],
                                                                            self.earthspheroidtype['sphere'])
                user_areacenter_distance2 = np.sum(np.square(userxyz_afterupdate2 - areacenterxyz)) ** 0.5
                if user_areacenter_distance2 <= self.maxdistance:
                    self.position = userxyz_afterupdate2
                    break
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5  # 获取最大移动距离的原因？？
        self.traffic_updata(tb, capacity)  # 流量更新

    # 随机选择三种业务并按照指数分布随机产生业务的持续时间
    def trafficduration(self):
        type = 'None'
        if self.offtime > 0:
            self.offtime -= 1
            if self.offtime < 0:
                self.offtime = 0
                ################
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:
                    self.ontime = np.random.exponential(self.traffictype['text'])
                    type = 'text'
                    self.qci = self.qci_type[type]
                elif traffic_choice == 2:
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    type = 'voice'
                    self.qci = self.qci_type[type]
                else:
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    type = 'video'
                    self.qci = self.qci_type[type]
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(self.offtime_restore)
                self.qci = 0

        return self.ontime
    # tb表step0已发送数据
    def waiting_data(self, tb, capacity):
        # rand_number = np.random.uniform(0, 1)
        # if rand_number <= bler:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata
        # else:
        #     self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb

        self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb  # step0剩余数据报=step0待发送数据+step0新到达数据-step0已发送
        if self.request == 1 and self.waiting_data_finally >= 0:  # 判断当前step有效传输数据量 # 有请求&&当前step剩余数据报>0
            self.current_txdata = tb
        elif self.request == 1 and self.waiting_data_finally < 0:  # 有请求，没有剩余数据
            self.current_txdata = self.waitingbit + self.newarrivaldata
        else:
            self.current_txdata = 0

        # if self.request == 1 and rand_number > bler and self.waiting_data_finally >= 0:
        #     self.current_txdata = tb
        # elif self.request == 1 and rand_number > bler and self.waiting_data_finally < 0:
        #     self.current_txdata = self.waitingbit + self.newarrivaldata
        # else:
        #     self.current_txdata = 0
        if self.current_txdata == 0 and self.request == 1:  # 用户有请求但卫星没有提供资源
            self.time_delay += 1  # 用户时延++
        if self.waiting_data_finally < 0:
            self.waiting_data_finally = 0
        self.throughput=((self.current_txdata/0.001))/(1024**2)  # 当前t时刻，单位Mbits/s
        self.waitingbit = self.waiting_data_finally  # step1待发送数据 = step0剩余数据
        self.newarrivaldata = self.cbrrate * 1 if self.ontime > 1 else self.cbrrate * self.ontime  # step1新到达数据
        self.current_waiting_data = self.waitingbit+self.newarrivaldata  # 当前t时刻为发送-->下轮待发送业务
        current_data_total = self.waitingbit + self.newarrivaldata
        if current_data_total > 0:  # 用户请求更新
            self.request = 1
        else:
            self.request = 0
        self.capacity = capacity
        ######原文献计算所需带宽
        self.number_of_rbg_nedded = RbgCountRequired(current_data_total)  # 不需要计算所需带宽
        # if self.number_of_rbg_nedded > self.max_number_of_rbg:
        #     self.number_of_rbg_nedded = self.max_number_of_rbg
        ###############################

        self.index+=1
        self.total_txdata = self.total_txdata+self.current_txdata  # 总吞吐量计算
        self.average_throughput = (self.total_txdata / (self.index / 1000)) / 1024 ** 2  # 单位转换

    def traffic_updata(self, tb, capacity):
        self.trafficduration()
        self.waiting_data(tb, capacity)

    # 初始化用户位置函数，以波束中心点为中心，随机生成self.maxdistance范围内用户经纬度
    def initial_random_position(self, beampara):

        originlatitude = beampara[0]
        originlongitude = beampara[1]
        maxaltitude = beampara[2]

        # 除去南北极
        if originlatitude >= 90:
            originlatitude = 89.999
        elif originlatitude <= -90:
            originlatitude = -89.999

        if maxaltitude < 0:
            maxaltitude = 0

        originlatituderadians = originlatitude * (np.pi / 180)
        originlongituderadians = originlongitude * (np.pi / 180)
        origincolatitude = (np.pi / 2) - originlatituderadians

        # 圆心角弧度数的最大值
        a = 0.99 * self.maxdistance / self.earth_radius
        if a > np.pi:
            a = np.pi

        d = np.random.uniform(0, self.earth_radius - self.earth_radius * np.cos(a))
        phi = np.random.uniform(0, np.pi * 2)
        alpha = math.acos((self.earth_radius - d) / self.earth_radius)
        theta = np.pi / 2 - alpha
        randpointlatitude = math.asin(
            math.sin(theta) * math.cos(origincolatitude) + math.cos(theta) * math.sin(origincolatitude) * math.sin(phi))
        intermedlong = math.asin((math.sin(randpointlatitude) * math.cos(origincolatitude) - math.sin(theta)) / (
                math.cos(randpointlatitude) * math.sin(origincolatitude)))
        intermedlong = intermedlong + np.pi / 2
        if phi > (np.pi / 2) and phi <= ((3 * np.pi) / 2):
            intermedlong = -intermedlong
        randpointlongtude = intermedlong + originlongituderadians
        randaltitude = np.random.uniform(0, maxaltitude)

        self.position = self.GeographicTocartesianCoordinate(randpointlatitude * (180 / np.pi),
                                                             randpointlongtude * (180 / np.pi), randaltitude,
                                                             self.earthspheroidtype['sphere'])
        self.log_lat_coordinates = [randpointlatitude * (180 / np.pi), randpointlongtude * (180 / np.pi),
                                    randaltitude]  # 度数为单位
        # print(self.pointposition)
        return self.position, self.log_lat_coordinates
        # beam_center经纬度 user_position笛卡尔坐标系， height卫星轨道高度
    def generate_angele_user(self, beam_center, user_position, height):
        beam_position = self.GeographicTocartesianCoordinate(beam_center[0], beam_center[1], beam_center[2], self.earthspheroidtype['sphere'])
        beam2user = m.sqrt((beam_position[0]-user_position[0])**2+(beam_position[1]-user_position[1])**2
                           +(beam_position[2]-user_position[2])**2)
        # print("beam2user", beam2user)
        angle_user = m.atan(beam2user/height)
        # print("angle_user", self.angle_user)
        self.angle_user = m.degrees(angle_user)
        # print("angle_user", self.angle_user)
        return self.angle_user


    # 将经纬度坐标转换为笛卡尔坐标系
    def GeographicTocartesianCoordinate(self, latitude, longitude, altitude, sphType):
        latitudeRadians = latitude * m.pi / 180
        longitudeRadians = longitude * m.pi / 180
        # print("longitudeRadians", longitudeRadians)
        # print("latitudeRadians", latitudeRadians)
        # a: semi - major axis of earth
        # e: first eccentricity of earth
        EARTH_RADIUS = 6371e3
        EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
        EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
        EARTH_SEMIMAJOR_AXIS = 6378137
        EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
        if sphType == "SPHERE":
            a = EARTH_RADIUS
            e = 0
        if sphType == "GRS80":
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_GRS80_ECCENTRICITY
        else:  # if sphType == WGS84
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_WGS84_ECCENTRICITY
        Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
        # print("rn", Rn)
        x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
        y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
        z = (Rn + altitude) * m.sin(latitudeRadians)
        # z = ((1 - pow(e, 2)) * Rn + altitude) * m.sin(latitudeRadians)
        cartesianCoordinates = np.array([x, y, z], dtype='float64')
        return cartesianCoordinates

    def get_distance(self):
        areaxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                       self.lat_long_areacenter[2], self.earthspheroidtype['sphere'])
        distance = (np.sum(np.square(self.position - areaxyz))) ** 0.5
        return distance



# 閺嶈宓侀悽銊﹀煕瀵版鍩岄惃鍕炊鏉堟挸娼�,娑撳﹣绔撮弮璺哄煝鐠囬攱鐪伴崪灞间繆闁挸顔愰柌蹇旀纯閺傛壆鏁ら幋铚傜秴缂冾喖鑻熼弴瀛樻煀閻€劍鍩涘ù渚€鍣虹拠閿嬬湴
def updata(user, tb, last_time_request, capacity, channalgain, interfer):
    user_list = user
    tb_list = np.zeros(len(user_list))
    capacity_list = np.zeros(len(user_list))
    req_num = len(last_time_request)
    for i in range(len(user_list)):
        user_list[i].model2_update(tb[i], capacity[i])
        user_list[i].channalgain = channalgain[i]
        user_list[i].interference = interfer[i]
    user_position_xyz, user_position_log_lat_coordinates = get_user_position(user_list, req_num)
    traffic_info, user_request = get_user_traffic_info(user_list, req_num)
    return user_position_xyz, user_position_log_lat_coordinates, traffic_info, user_request


def initial_all_user(maxdistance, numofuser,beam_position, ontime=5, offtime=1):
    """
        sparse the user 
    """
    userlist = []
    position = beam_position[:,1:]
    # 用户需求分布 [200M, 400M]
    # cbrrate_list1 = [random.randint(setting.user_cbrrate_min, setting.user_cbrrate_max) for i in range(10000)]
    # cbrrate_list2 = [0]*2000
    # cbrrate_list = cbrrate_list1 + cbrrate_list2
    # random_weights = [random.randint(0,1) for i in range(0,12000)]
    # # 需求存在
    # cbrrate_list1 = [random.randint(setting.user_cbrrate_min, setting.user_cbrrate_max) for i in range(10000)]
    # 只有波束1用户有需求
    cbrrate_list_test = [setting.user_cbrrate_avg, 0, 0, 0, 0, 0, 0]
    for i in range(len(position)):
        # userlist += ([user(maxdistance, position[i],
        #               random.choices(cbrrate_list1, k = 1)[0], ontime, offtime) for i in range(numofuser)])
        userlist += ([user(maxdistance, position[i],
                      cbrrate_list_test[i], ontime, offtime) for j in range(numofuser)])
    return userlist

# 根据用户所在位置找到所属波束
def get_user_position(user, req_num):
    # 初始化用户和泊松分布均值
    userlist = user
    user_position_XYZ = []
    user_position_log_lat_coordinates=[]
    ###############
    # 随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(userlist)):
        if (len(user_position_XYZ) >= req_num):
            break
        if userlist[i].request == 1:
            # user_positionAndnumber.append((i, userlist[i].log_lat_coordinates))  # 娴犮儱鍘撶紒鍕埌瀵繐鐨㈤崣鎴ｆ崳娑撴艾濮熺拠閿嬬湴閻ㄥ嫮鏁ら幋椋庣椽閸欏嘲鎷扮€电懓绨查惃鍕秴缂冾喖鐡ㄩ弨鎹愮箻閸掓銆�
            position = userlist[i].position
            position2 = userlist[i].log_lat_coordinates
            user_position_XYZ.append(position)
            user_position_log_lat_coordinates.append(position2)
    ################################
    # for i in range(numofuser):
    #     userlist[i].model2_update()
    #     user_position.append(userlist[i].jingweiposition)
    # print('----',user_positionAndnumber)

    return user_position_XYZ,user_position_log_lat_coordinates


def get_user_log_lat_coordinates(user):
    userlist = user
    user_position_list=[]
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position=userlist[i].log_lat_coordinates
            user_position_list.append(position)
    return user_position_list

# TODO：修改这部分代码，检查逻辑
def get_user_traffic_info(user, max_req):
    userlist = user
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):  # time_delay >= 6 request==0情况会出现吗？
        if (userlist[i].request == 1 or userlist[i].time_delay >= 6) and max_req < userlist[i].global_max_server:
            """
                The beam preferentially covers the early requesting user
            """
            user_request.append(i)  # 发出request用户数量大于最大可服务用户数量时，user[i].time_delay超过6时，该请求长时间未得到回应，作废
            user[i].time_delay = 0  # 即刻分配资源
        elif userlist[i].request == 1:  # 超出最大服务用户数量限制
            """
                Temporarily put the current request on hold and wait for the beam service to finish
            """
            user[i].time_delay += 1
            if user_request.count(i) == 1:  # 该情况是否会出现
                user_request.remove(i)
        else:
            pass
        traffic_info.append(
            (i, userlist[i].log_lat_coordinates[0], userlist[i].log_lat_coordinates[1], userlist[i].log_lat_coordinates[2],
             userlist[i].angle_user, userlist[i].current_waiting_data, userlist[i].newarrivaldata, userlist[i].request,
             userlist[i].current_txdata,  userlist[i].qci,
             userlist[i].number_of_rbg_nedded, userlist[i].total_txdata, userlist[i].throughput, userlist[i].average_throughput,
             userlist[i].capacity, userlist[i].channalgain, userlist[i].interference))

    while len(user_request) > userlist[0].global_max_server:
        """
            preferentially pop the user that time_delay is small 
        """
        min_delay = 50
        for user_ind in user_request:
            if userlist[user_ind].time_delay == 0 and user_request.count(user_ind) == 1:    
                user_request.remove(user_ind)
            elif userlist[user_ind].time_delay < min_delay:
                min_delay = userlist[user_ind].time_delay
                if min_delay == 0 and user_request.count(user_ind) == 1:
                    user_request.remove(user_ind)

    traffic_info = np.array(traffic_info, dtype='float')
    traffic_info = pd.DataFrame(traffic_info,
                                columns=['user', 'jing', 'wei', 'gao', 'angle', 'waitingdata', 'newdata', 'request',
                                         'last_time_txdata','qci', 'number_of_rbg_nedded','total_txdata',
                                         'throughput(mbps)', 'average_throughput', 'capacity', 'channalgain', 'interference'])
    return traffic_info, user_request


def get_all_user_position_and_request(user):
    userlist = user
    position_and_req = []
    for i in range(len(userlist)):
        position = userlist[i].position.tolist()
        position_and_req.append((i, position[0], position[1], position[2], userlist[i].request))
    position_and_req = np.array(position_and_req, dtype='float')

    return position_and_req





