# -*-coding:utf-8-*-
import numpy as np
from sgp4.model import Satellite

# SATELLITE
satellite_height = 600000
satellite_beams = 7
satellite_rbg_per_beam = 10
satellite_maxdistance = 20000  # ²¨ÊøÖ±¾¶

# SYSTEM
user_per_beam = 5
bw = 400e6
frequency = 20e9

# USER
user_cbrrate_max = 400000  # bit/ms
user_cbrrate_min = 200000  # bit/ms
user_cbrrate_avg = 300000  # bit/ms
PF = False
