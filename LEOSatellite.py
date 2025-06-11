import numpy as np
import math as m
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72

import setting


# // https://celestrak.com/NORAD


def read_tle_file(TLE_PATH):
        satlist = []
        tle_str1 = []
        tle_str2 = []
        f = open(TLE_PATH)
        lines = f.readlines()
        for i in range(len(lines) // 3):

            #satname = int(lines[3 * i][8:11])
            #satlist.append(satname)
            satlist.append(lines[3 * i])
            tle_str1.append(lines[3 * i + 1])
            tle_str2.append(lines[3 * i + 2])

        return satlist, tle_str1, tle_str2



def createInitBeamCenterPos(satnum, numPointsof, SateBLH, numPoints, arcDist):
    beamCenterAlloc = []
    beamCenterAlloc_lat_log = []
    for i in range(numPointsof, numPointsof + numPoints):
        cellid = i
        azimuth = (i + 1) * 360.0 / numPoints
        # print("azimuth", azimuth)
        a = 6378137.0
        b = 6356752.3142
        f = 1.0 / 298.257223563
        alpha1 = azimuth * m.pi / 180.0
        sinAlpha1 = m.sin(alpha1)
        cosAlpha1 = m.cos(alpha1)
        tanU1 = (1 - f) * m.tan(SateBLH[0] * m.pi / 180.0)
        cosU1 = 1 / m.sqrt((1 + tanU1 * tanU1))
        sinU1 = tanU1 * cosU1
        sigma1 = m.atan2(tanU1, cosAlpha1)
        sinAlpha = cosU1 * sinAlpha1
        cosSqAlpha = 1 - sinAlpha * sinAlpha
        uSq = cosSqAlpha * (a * a - b * b) / (b * b)
        A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

        sigma = arcDist / (b * A)
        sigmaP = 2 * m.pi
        sinSigma = m.sin(sigma)
        cosSigma = m.cos(sigma)
        cos2SigmaM = m.cos(2 * sigma1 + sigma)
        while (abs(sigma - sigmaP) > 1e-12):
            deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                                                               B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (
                                                                       -3 + 4 * cos2SigmaM * cos2SigmaM)))
            sigmaP = sigma
            sigma = arcDist / (b * A) + deltaSigma
        tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
        lat2 = m.atan(
            (sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1) / ((1 - f) * m.sqrt(sinAlpha * sinAlpha + tmp * tmp)))
        lambd = m.atan((sinSigma * sinAlpha1) / (cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1))
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        L = lambd - (1 - C) * f * sinAlpha * (
                sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
        pointPosition = GeographicToCartesianCoordinates(lat2 * 180 / m.pi, SateBLH[1] + L * 180 / m.pi, 0, "GRS80")
        lat_log = ConstructFromVector(pointPosition[0], pointPosition[1], pointPosition[2], "GRS80")
        beamCenterAlloc.append([cellid, pointPosition[0], pointPosition[1], pointPosition[2]])
        beamCenterAlloc_lat_log.append([cellid, lat_log[0], lat_log[1], lat_log[2]])
    return beamCenterAlloc, beamCenterAlloc_lat_log

def setInitBeamCenterPos(i, SateBLH, type):
    if type == "STARLINK":
        numPointsofC1 = 1
        numPointsofC2 = 6
        # numPointsofC3 = 18
        # numPointsofC4 = 27
        # numPointsofC5 = 43

        maxArcDistFromSubPos = (3 ** 0.5 + 1) * setting.satellite_maxdistance / 2
        arcDist1 = maxArcDistFromSubPos * 0 / 3
        arcDist2 = maxArcDistFromSubPos * 2 / 3
        # arcDist3 = maxArcDistFromSubPos * 5 / 11
        # arcDist4 = maxArcDistFromSubPos * 7 / 11
        # arcDist5 = maxArcDistFromSubPos * 9 / 11

        allocate1, lat_log1 = createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2, lat_log2 = createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        # allocate3, lat_log3 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2, SateBLH, numPointsofC3, arcDist3)
        # allocate4, lat_log4 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2 + numPointsofC3, SateBLH, numPointsofC4, arcDist4)
        # allocate5, lat_log5 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2 + numPointsofC3 + numPointsofC4, SateBLH, numPointsofC5, arcDist5)
        beam = np.array(allocate1 + allocate2)  # + allocate3 + allocate4 + allocate5)
        lat_log = np.array(lat_log1 + lat_log2)  # + lat_log3 + lat_log4 + lat_log5)
    return beam, lat_log

def sateposition(satlist, tle_str1, tle_str2, epoch, step):
    satnum = len(satlist)
    position = []
    velocity = []
    undersate = []
    SateBLH = []
    allocate = []
    SateId = []
    for i in range(satnum):
        time = (step + epoch * 100) * 0.001  # 每个epoch100个step，每个step1ms
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = (time - hours * 3600 - minutes * 60)
        satellite = twoline2rv(tle_str1[i], tle_str2[i], wgs72)  # 根据卫星道道息息生成卫星类
        position1, velocity1 = satellite.propagate(2022, 9, 20, hours, minutes, seconds)  # 生成卫星位置(xyz)和速度矢量  
        # print("position1", position1)
        # print("velocity1", velocity1)
        position1 = (position1[0] * 1000, position1[1] * 1000, position1[2] * 1000)  
        # print("position1", position1)
        position.append(position1)
        SateId.append([satlist[i], position1])
        velocity.append(velocity1)
        SateBLH1 = ConstructFromVector(position1[0], position1[1], position1[2], "GRS80")  # xyz坐标转换成经度度
        SateBLH.append(SateBLH1)
        underposition =GeographicToCartesianCoordinates(SateBLH1[0], SateBLH1[1], 0, "GRS80")  
        # print("underposition", underposition)
        undersate.append(underposition)  # 星下坐标xyz
        # allocate.extend(setInitBeamCenterPos(i, SateBLH1, "STARLINK"))
        beam_xyz, beam_lat= setInitBeamCenterPos(i, SateBLH1, "STARLINK")

    return position, SateId, SateBLH, undersate, beam_xyz, beam_lat


def GeographicToCartesianCoordinates(latitude, longitude, altitude, sphType):
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
    cartesianCoordinates = [x, y, z]
    return cartesianCoordinates


def ConstructFromVector(x, y, z, sphType):
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY

    latitudeRadians = m.asin(z / m.sqrt(x ** 2 + y ** 2 + z ** 2))
    latitude = latitudeRadians * 180 / m.pi

    if x == 0 and y > 0:
        longitude = 90
    elif x == 0 and y < 0:
        longitude = -90
    elif x < 0 and y >= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi + 180
    elif x < 0 and y <= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi - 180
    else:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi

    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))
    altitude = m.sqrt(x**2+y**2+z**2)-Rn
    # print("altitude", altitude)
    return [latitude, longitude, altitude]


# # 锟斤拷锟捷笛匡拷锟斤拷锟斤拷锟斤拷系锟斤拷锟姐经锟斤拷维锟饺猴拷锟斤�?
# def ConstructFromVector (x, y, z, type):
#     if type == "SPHERE":
#         m_polarRadius = 6378137.0
#     if type == "WGS84":
#         m_polarRadius = 6356752.314245
#     if type == "GRS80":
#         m_polarRadius = 6356752.314103
#     equatorRadius = 6378137.0  # 锟斤拷锟斤拷刖� 锟斤拷锟斤拷锟斤�?
#     m_equatorRadius = equatorRadius
#     m_e2Param = ((m_equatorRadius * m_equatorRadius) - (m_polarRadius * m_polarRadius)) / (m_equatorRadius * m_equatorRadius )
#
#     # distance from the position point (P) to earth center point (origin O)
#     op = m.sqrt(x * x + y * y + z * z)
#
#     if op > 0:
#         # longitude calculation
#         lon = m.atan(y / x)
#         # scale longitude between - PI and PI (-180 and 180 in degrees)
#         if (x != 0) | (y != 0):
#             m_longitude = m.atan(y / x) * 180.0 / m.pi
#             if x < 0:
#                 if y > 0:
#                     m_longitude = 180 + m_longitude
#                     lon = lon - m.pi
#             else:
#                     m_longitude = -180 + m_longitude
#                     lon = m.pi + lon
#         # Geocentric latitude
#         latG = m.atan(z / (m.sqrt(x * x + y * y)))
#         # Geocentric latitude (of point Q, Q is intersection point of segment OP and reference ellipsoid)
#         latQ = m.atan(z / ((1 - m_e2Param) * m.sqrt(x * x + y * y)))
#         # calculate radius of the curvature
#         rCurvature = m_equatorRadius / m.sqrt(1 - m_e2Param * m.sin(latQ) * m.sin(latQ))
#         # x, y, z of point Q
#         xQ = rCurvature * m.cos(latQ) * m.cos(lon)
#         yQ = rCurvature * m.cos(latQ) * m.sin(lon)
#         zQ = rCurvature * (1 - m_e2Param) * m.sin(latQ)
#         # distance OQ
#         oq = m.sqrt(xQ * xQ + yQ * yQ + zQ * zQ)
#         # distance PQ is OP - OQ
#         pq = op - oq
#         # length of the normal segment from point P of line (PO) to point T.
#         # T is intersection point of linen the PO normal and ellipsoid normal from point Q.
#         tp = pq * m.sin(latG - latQ)
#         m_latitude = (latQ + tp / op * m.cos(latQ - latG)) * 180.0 / m.pi
#         m_altitude = pq * m.cos(latQ - latG)
#
#     return [m_latitude, m_longitude, m_altitude]



def limitquare():
    radius = 30000  
    m_DayNumLen = 0.03
    EARTH_RADIUS = 6378137
    # center
    centerLatLonAlt = GeographicToCartesianCoordinates(0, 0, 0, "GRS80")
    return centerLatLonAlt, radius


def load_tle(TLE_PATH, epoch, step):
    """
    output:    
        position: satelite position -> list: []
        SateId: sateid+position -> list: [[],[]..]
        SateBLH: [latitude, longitude, altitude] -> list: [[],[]..]
        undersate: [x, y, z] -> list: [[],[]..]
        beamallocate: beam postion [cell,x,y,z] + [cell, lati, longti, alti]
    """
    satlist, tle_str1, tle_str2 = read_tle_file(TLE_PATH)  # satlist锟斤拷锟斤拷锟叫憋拷锟斤拷tle_str1TLE锟侥硷拷锟斤拷一锟斤�?,tle_str2TLE锟侥硷拷锟节讹拷锟斤�?
    # print("satlist", satlist)
    # print("tle_str1", tle_str1)
    # print("tle_str2", tle_str2)

    position, SateId, SateBLH, undersate, beamallocate, beam_lat = sateposition(satlist, tle_str1, tle_str2, epoch,
                                                                      step)  # position(x,y,z),velocity,undersate(B,L,H)
    return beamallocate, beam_lat

def userconnectsate(userlist, userposition, beamposition, request, usernumer, max_req):
    all_connect_info = []
    req_num = len(request)
    beam_number = np.zeros(len(userlist))
    # print(len(beam_number))
    # input()
    # beam_connect = []
    # print("user position,req:",len(userposition), "--", len(request))
    req = min(len(userposition),len(request))
    for i in range(req):
        if i >= len(userposition):
            break 
        user = userposition[i]
        distance_max = setting.satellite_maxdistance/2  # 最大距离：半径10km
        # num_sate = 0
        connect_beam_position = 0
        for j in range(len(beamposition)):
            beam = beamposition[j][1:] # xyz_cord
            distance = np.sqrt(np.sum((user - beam) ** 2))
            if distance < distance_max:
                # num_sate += 1
                distance_max = distance
                connect_beam_position = beam
                beam_number[request[i]] = beamposition[j][0] + 1
        # print("satelites which are able to be seen:", num_sate)
        if(distance_max >= setting.satellite_maxdistance/2):
            print("user", request[i], "can't be connected to any beam")
        user_connect_info = np.hstack((request[i], user, connect_beam_position, [0, 0, 0],beam_number[request[i]]))
        all_connect_info.append(user_connect_info)
        """     
        for i in range(len(beam_connect)):
            print(len(beam_connect), "user connect to beam :", beam_connect[i]) 
        """
    # print("position",len(all_connect_info))
    all_connect_info = np.array(all_connect_info)  # beam_number：request用户对应波束(根据地理位置，用户位置-波束中心)
    return all_connect_info, beam_number  
    # all_connect_info：返回发出请求user的[编号， [user位置*3]，[所在波束中心位置*3]，[未知含义坐标*3]，所在波束编号]

def seesate(epoch, step):
    TLE_PATH = "./starlinkTle.txt"
    satlist, tle_str1, tle_str2 = read_tle_file(TLE_PATH)  # satlist锟斤拷锟斤拷锟叫憋拷锟斤拷tle_str1TLE锟侥硷拷锟斤拷一锟斤�?,tle_str2TLE锟侥硷拷锟节讹拷锟斤�?
    # print("satlist", satlist)
    # print("tle_str1", tle_str1)
    # print("tle_str2", tle_str2)

    position, SateId, SateBLH, undersate, beamallocate = sateposition(satlist, tle_str1, tle_str2, epoch,
                                                                      step)  # position(x,y,z),velocity,undersate(B,L,H)
    # position为锟斤拷锟斤拷位锟矫ｏ拷x,y,z锟斤拷锟斤拷SateId为锟斤拷锟角憋拷锟�+锟斤拷锟斤拷位锟矫ｏ拷undersate为锟斤拷锟斤拷锟斤拷锟铰点集锟较ｏ拷allocate为锟斤拷锟斤拷锟斤拷锟斤�?+小锟斤拷锟斤拷锟�?+锟斤拷锟斤拷锟斤拷锟斤拷
    centerLatLonAlt, radius = limitquare()
    # print("position", position)   # xyz
    # x, y, z = GeographicToCartesianCoordinates(SateBLH[0][0], SateBLH[0][1], SateBLH[0][2], "GRS80")
    # a, b, c = ConstructFromVector(x, y, z, "GRS80")
    # x, y, z = ConstructFromVector(position[0][0], position[0][1], position[0][2], "GRS80")
    # a, b, c = GeographicToCartesianCoordinates(x, y, z, "GRS80")
    # print("position[0]", position[0])
    # print("a", a)
    # print("b", b)
    # print("c", c)
    # print("centerLatLonAlt", centerLatLonAlt)
    # print("radius", radius)
    BeamSeeAlloocate = []
    satsum = len(satlist)
    seesate = []
    seesateId = []
    distance_max = 812000
    for i in range(satsum):
        # distance 锟斤拷锟斤拷锟斤拷锟铰碉拷锟斤拷锟矫伙拷锟秸硷拷锟斤拷锟斤拷锟斤拷锟斤拷之锟斤拷木锟斤拷锟�
        distance = m.sqrt((undersate[i][0]-centerLatLonAlt[0])**2+(undersate[i][1]-centerLatLonAlt[1])**2+(undersate[i][2]-centerLatLonAlt[2])**2)
        # sate2undersate 锟斤拷锟斤拷锟斤拷锟斤拷锟铰碉拷之锟斤拷木锟斤拷锟�
        sate2undersate = m.sqrt((position[i][0]-undersate[i][0])**2+(position[i][1]-undersate[i][1])**2+(position[i][2]-undersate[i][2])**2)
        # sate2undersate 锟斤拷锟斤拷锟斤拷锟矫伙拷锟秸硷拷锟斤拷锟斤拷锟斤拷锟斤拷之锟斤拷木锟斤拷锟�?
        sate2center = m.sqrt((position[i][0]-centerLatLonAlt[0])**2+(position[i][1]-centerLatLonAlt[1])**2+(position[i][2]-centerLatLonAlt[2])**2)
        equatorRadius = 6378137.0
        # print("DISTANCE", m.sqrt((undersate[i][0])**2+(undersate[i][1])**2+(undersate[i][2])**2))
        # print("position", position[i][0])
        # print("position", position[i][1])
        # print("position", position[i][2])
        # print("SateBLH", SateBLH[i][0])
        # print("SateBLH", SateBLH[i][1])
        # print("SateBLH", SateBLH[i][2])
        # print("undersate", undersate[i][0])
        # print("undersate", undersate[i][1])
        # print("undersate", undersate[i][2])
        # print("distance", distance)
        # print("SateBLH", SateBLH[i][2])
        # print("sate2undersate", sate2undersate)
        # print("sate2center", sate2center)
        A = m.acos(equatorRadius/(equatorRadius+SateBLH[i][2]))*180/m.pi
        L = A*m.pi*equatorRadius/180
        # print("A", A)
        # print("L", L)

        if distance < distance_max:
            seesate.append(SateBLH[i])
            seesateId.append(SateId[i])
            for j in range(len(beamallocate)):
                if beamallocate[j][0] == SateBLH[i]:
                    BeamSeeAlloocate.append(beamallocate[j] + [position[i]])
    print("satelites which we can see:", len(seesate))
    return seesate, seesateId, BeamSeeAlloocate

""" 
def userconnectsate(ueposition, epoch, step):
    seesatenum, SeesateId, beamallocate = seesate(epoch, step)
    # print("seesatenum", seesatenum)
    # print("SeesateId", SeesateId)
    # print("beamallocate", beamallocate)
    uenum = len(ueposition)

    satuedict = []
    Beam = []
    UeLinkSate = []
    beamnum = len(beamallocate)
    for i in range(uenum):
        ue = []
        min = []
        link = []
        beam = []
        for j in range(beamnum):
            uebeamdistance = m.sqrt((beamallocate[j][2][0]-ueposition[i][0])**2+(beamallocate[j][2][1]-ueposition[i][1])**2+(beamallocate[j][2][2]-ueposition[i][2])**2)
            if len(ue) != 0:
                mindistance = ue[0]
                if mindistance > uebeamdistance:
                    ue.remove(mindistance)
                    ue.append(uebeamdistance)
                    min.remove(min[0])
                    min.append([ueposition[i]]+ beamallocate[j])
                    link.remove(link[0])
                    link.append([ueposition[i], beamallocate[j][1]])
                    beam.remove(beam[0])
                    beam.append([beamallocate[j][1], beamallocate[j][2]])

            else:
                ue.append(uebeamdistance)
                min.append([ueposition[i]] + beamallocate[j])
                link.append([ueposition[i], beamallocate[j][1]])
                beam.append([beamallocate[j][1], beamallocate[j][2]])
        #print(min[0])

        satuedict.append(min[0])  
        Beam.append(beam[0])
        UeLinkSate.append(link[0])  
    return satuedict, Beam, UeLinkSate
 """    

if __name__ == "__main__":
    satlist, tle_str1, tle_str2 = read_tle_file('./starlinkTle.txt')
    print(satlist,tle_str1,tle_str2)