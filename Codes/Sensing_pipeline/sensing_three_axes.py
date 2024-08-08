import pandas as pd
import numpy as np
import datetime
import serial
import struct
import math
from Rosmaster_Lib import Rosmaster
import pickle
import sys

import ctypes

import matplotlib
import tkinter
from reconstruct_tag import find_vertex_points, calculate_rotation_angle
matplotlib.use('TkAgg')
# from localize import ComputeD

import matplotlib.pyplot as plt

def gaussian(x, pos, wid):
    g = np.exp(-((x - pos) / (0.60056120439323 * wid)) ** 2)
    return g


# ------------Experiment Settings-------------
# 加载共享库
ddtw_c = ctypes.CDLL('./DDTW.so')

# 定义C函数的参数类型
ddtw_function = ddtw_c.DDTW_matching_res
ddtw_function.argtypes = [ctypes.POINTER(ctypes.c_double * 3), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
ddtw_function.restype = ctypes.c_double

# 加载共享库
localize_c = ctypes.CDLL('./localize.so')
# 定义C函数的参数类型
localize_function = localize_c.ComputeD
localize_function.argtypes = [ctypes.c_double * 3, ctypes.c_double * 3, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double * 1]
localize_function.restype = ctypes.c_double

scenario = "Detecting_Polaris_"
# Number of sensors
num = 9
# experiment times
times = 1
# COM port for connecting sensor array, jetson nano
COM = "/dev/rplidar"
# Windows port
# COM = "COM12"
# load matching template
gt_datas = []
# file = "temp/template_36_5.pkl"
# with open(file, 'rb') as f:
#     gt_datas = pickle.load(f)

# ---------Initial Parameters and Thresholds---------
# These parameters and thresholds are empirically determined for the specific sensors used
# May need to finetune to optimize results if used different sensors
wnd = 3 # Gaussian Smoother for the raw signal data
wnd_d = 1  # Gaussian Smoother for the 1st derivative data

delta_thrd_tol =[120, 120, 120, 120, 120, 120, 120, 120, 120] 
delta_thrd_tol =[30, 30, 30, 30, 30, 30, 30, 30, 30] 
amp_thrd_tol = [1.05] * num
start_raw_amp_x = [0] * num
start_raw_amp_y = [0] * num
start_raw_amp_z = [0] * num

# tag and sensor layout
theta, h, l = 0, 15, 20

# Constants and Globals
raw_result = []
raw_name = ['Time Stamp'] + ['Sensor ' + str(i) for i in range(1, num + 1)]
n = 0  # index for the data points
no = 0  # index for the peaks

# raw z-axis data, smoothed z-axis data, 1st derivative of z-axis data, smoothed 1st derivative of z-axis data
x, sx, dx, sdx = ([[] for _ in range(num)] for _ in range(4))
y, sy, dy, sdy = ([[] for _ in range(num)] for _ in range(4))
z, sz, dz, sdz = ([[] for _ in range(num)] for _ in range(4))
sig_tol, s_sig_tol, ds_sig_tol, sd_sig_tol = ([[] for _ in range(num)] for _ in range(4))
# slope and raw data list of three-axis
slope_list_tol, raw_list_tol = ([[] for _ in range(num)] for _ in range(2))
slope_thrd_tol = [10000] * num
estimate_tol = [True] * num

LastRAW_tol = [0] * num  # Record the index of the last peak
LastRAW_x, LastRAW_y, LastRAW_z = ([0] * num for _ in range(3))

S_flag_tol = [0] * num

magnet_info = []
tag_info = []
amp_tol, amp2_tol = 0, 0


cur_n = -1
flag_s = False
flag_update = 0

# smoother for the raw data and 1st derivative
SmoothVector_r = gaussian(np.arange(wnd + 1)[1:], wnd / 2, wnd / 2)
SmoothVector_r = SmoothVector_r / (np.sum(SmoothVector_r))
SmoothVector_d = gaussian(np.arange(wnd_d + 1)[1:], wnd_d / 2, wnd_d / 2)
SmoothVector_d = SmoothVector_d / (np.sum(SmoothVector_d))

sensors = np.zeros((num, 3))
data = bytearray(4 * (3 * num))
cnt = 1
current = 0
offset_list = []

fin_angle, fin_dis = [0] * num, [0] * num
last_index = 0
 # Connect to the car
car = Rosmaster()
car.create_receive_threading()
enable = True
car.set_auto_report_state(enable, forever=False)

car_flag = True


# Detect the magnet in real time
def detectMag():
    arduino = serial.Serial(port=COM, baudrate=115200, timeout=None)

    print("Begin detecting at", str(datetime.datetime.now()))
    # car.set_car_motion(0.2, 0, 0)
    first = 0
    t = []
    global cnt, current, n_z, n, no, z, sz, dz, sdz, amp_tol, amp2_tol
    global x, sx, dx, sdx, y, sy, dy, sdy, sig_tol, s_sig_tol, ds_sig_tol, sd_sig_tol
    global slope_thrd_tol, slope_list_tol, estimate_tol, raw_result, raw_list_tol
    global delta_t, offset_list, fin_angle, fin_dis, magnet_info,cur_n, flag_s, gt_datas
    global theta, h, l, tag_info, flag_update
    global LastRAW_tol, LastRAW_x, LastRAW_y, LastRAW_z, last_index, car_flag
    try:
        while True:
            arduino.flushInput()
            flag = 1    
            arduino.readinto(data)
            
            raw_result_tmp = [current]
            for i in range(num):
                sensors[i, 0], = struct.unpack('f', data[(i * 12):(4 + i * 12)])
                sensors[i, 1], = struct.unpack('f', data[(i * 12 + 4):(8 + i * 12)])
                sensors[i, 2], = struct.unpack('f', data[(i * 12 + 8):(12 + i * 12)])
                
                # save to 4 float numbers
                sensors[i, 0] = round(sensors[i, 0], 4)
                sensors[i, 1] = round(sensors[i, 1], 4)
                sensors[i, 2] = round(sensors[i, 2], 4)
                raw_result_tmp.append((sensors[i, 0], sensors[i, 1], sensors[i, 2]))
                            
            if first == 0:
                for i in range(num):
                    tmp = np.sqrt(sensors[i, 0] ** 2 + sensors[i, 1] ** 2 + sensors[i, 2] ** 2)
                    if math.isnan(tmp) or tmp > 1000:
                        flag = 0

            if flag == 1:
                if(first == 0):
                    first = 1
                    print("Start detection")
                for i in range(num):
                    x[i].append(sensors[i, 0]-start_raw_amp_x[i])
                    y[i].append(sensors[i, 1]-start_raw_amp_y[i])
                    z[i].append(sensors[i, 2]-start_raw_amp_z[i])
                    # tol = np.sqrt(sensors[i, 0] ** 2 + sensors[i, 1] ** 2 + sensors[i, 2] ** 2)
                    tol = np.sqrt((sensors[i,0]-start_raw_amp_x[i]) ** 2 + (sensors[i,1]-start_raw_amp_y[i]) ** 2 + (sensors[i,2]-start_raw_amp_z[i]) ** 2)
                    sig_tol[i].append(tol)
            else:
                print("Invalid data")
                continue
            
            raw_result.append(raw_result_tmp)
            t.append(current)
            interval = 10000
            current +=  interval/ 1000 / 1000
            

            # buffer some x-axis data points before starting the detection
            if n < wnd:
                for i in range(num):
                    # tmp = np.sqrt(sensors[i, 0] ** 2 + sensors[i, 1] ** 2 + sensors[i, 2] ** 2)
                    tmp = np.sqrt((sensors[i,0]-start_raw_amp_x[i]) ** 2 + (sensors[i,1]-start_raw_amp_y[i]) ** 2 + (sensors[i,2]-start_raw_amp_z[i]) ** 2)
                    sx[i].append(sensors[i, 0]-start_raw_amp_x[i])
                    dx[i].append(0)
                    sdx[i].append(0)
                    sy[i].append(sensors[i, 1]-start_raw_amp_y[i])
                    dy[i].append(0)
                    sdy[i].append(0)
                    sz[i].append(sensors[i, 2]-start_raw_amp_z[i])
                    dz[i].append(0)
                    sdz[i].append(0)
                    # sig_tol[i].append(tmp)
                    s_sig_tol[i].append(tmp)
                    ds_sig_tol[i].append(0)
                    sd_sig_tol[i].append(0)
                n += 1
                continue

            # Smoothing the raw data
            for i in range(num):
                sx[i].append(np.sum(SmoothVector_r * (np.array(x[i][-wnd:]))))
                sy[i].append(np.sum(SmoothVector_r * (np.array(y[i][-wnd:]))))
                sz[i].append(np.sum(SmoothVector_r * (np.array(z[i][-wnd:]))))
                s_sig_tol[i].append(np.sum(SmoothVector_r * (np.array(sig_tol[i][-wnd:]))))

            # Calculate the 1st derivative
            for i in range(num):
                last_point = s_sig_tol[i][-1]
                second_point = s_sig_tol[i][-2]
                derivative = (last_point - second_point) / 2
                ds_sig_tol[i].append(derivative)

            # smooth the 1st derivative
            for i in range(num):
                sd_sig_tol[i].append(np.sum(SmoothVector_d * (np.array(ds_sig_tol[i][-wnd_d:]))))
            

            ind_tmp, n_tmp = [], []
            amp_tmp_list = []   

            for i in range(num):
                S_flag_tol[i] = 0
                if len(slope_list_tol[i]) == 20:
                    if estimate_tol[i]:
                        slope_thrd_tol[i] = amp_thrd_tol[i] * np.abs(np.array(slope_list_tol[i][1:])).mean()
                        print('sensor %d: Pre-done with slope threshold equaling to %.2f' % (
                            i + 1, slope_thrd_tol[i]))
                        
                        start_raw_amp_x[i] = np.array(x[i][1:]).mean()
                        start_raw_amp_y[i] = np.array(y[i][1:]).mean()
                        start_raw_amp_z[i] = np.array(z[i][1:]).mean()
                        estimate_tol[i] = False
                    
                # Detect the positive peak of a magnet
                if sd_sig_tol[i][-1] < 0 and sd_sig_tol[i][-2] >= 0:
                    slope_tol = sd_sig_tol[i][-1] - sd_sig_tol[i][-3]
                    slope_list_tol[i].append(slope_tol)
                    raw_index = n - wnd
                    raw_tol = s_sig_tol[i][n - wnd]
                    raw_list_tol[i].append(raw_tol)
                    if(len(slope_list_tol[i]) < 30):
                        continue
                    # if(len(slope_list_tol[i]) == 30):
                    #     print("Begin detecting")
                    count = 0
                    if(car_flag):
                        for i_ind in range(num):
                            # if total slope_list_tol is larger than 35, start moving
                            if(len(slope_list_tol[i_ind]) > 30):
                                count += 1
                            if(count == num):
                                print("Begin detecting")
                                car.set_car_motion(0.06, 0, 0)
                                car_flag = False

                
                    if slope_tol <= -slope_thrd_tol[i]:
                        if raw_tol - LastRAW_tol[i] >= delta_thrd_tol[i]:
                            amp_tol = raw_tol - LastRAW_tol[i]
                            amp_tmp_z = sz[i][raw_index] - LastRAW_z[i]
                            amp_tmp_x = sx[i][raw_index] - LastRAW_x[i]
                            amp_tmp_y = sy[i][raw_index] - LastRAW_y[i]
                            amp_tmp_list.append([amp_tmp_x, amp_tmp_y, amp_tmp_z, amp_tol])
                            cur_index = i
                            ind_tmp.append(cur_index)
                            cur_n = n
                            n_tmp.append(cur_n)
                            flag_s = True
                            # print(raw_index)
                            print("Sensor %d: detct a magnet, tol-amp: %.2f" % (i + 1, amp_tol))
                            print("n: ", n)
                            no += 1
                    else:
                        forward_points = 20
                        incre_points = 10
                        # for i in range(num):
                        #     if(flag_update == 0):
                        #         LastRAW_x[i] = (np.array(sx[i][raw_index - forward_points: raw_index - forward_points + incre_points])).mean()
                        #         LastRAW_y[i] = (np.array(sy[i][raw_index - forward_points: raw_index -forward_points + incre_points])).mean()
                        #         LastRAW_z[i] = (np.array(sz[i][raw_index - forward_points: raw_index - forward_points + incre_points])).mean()
                        #         LastRAW_tol[i] = (np.array(s_sig_tol[i][raw_index - forward_points: raw_index-forward_points+incre_points])).mean()
                        #         flag_update = 1
            if flag_s:
                magnet_info.append([cur_n, ind_tmp, amp_tmp_list])
                flag_s = False
                # print(magnet_info)
        

            # k = 0
            after_ind = 15
            if(len(magnet_info) > 0 and n - magnet_info[0][0] == after_ind):
                tmp_n = magnet_info[0][0]
                tmp_ind = magnet_info[0][1]
                info = []
                for j in range(len(tmp_ind)):
                    
                    tmp_index = tmp_ind[j]
                    info.append(tmp_index)
                    info.append(tmp_n)
                    # if flag_:
                    # print("tmp_n: ", tmp_n)
                    amp_x = sx[tmp_index][tmp_n-after_ind : tmp_n+after_ind]
                    amp_y = sy[tmp_index][tmp_n-after_ind : tmp_n+after_ind]
                    amp_z = sz[tmp_index][tmp_n-after_ind : tmp_n+after_ind]
                    # amp_tol = s_sig_tol[tmp_index][tmp_n-30:tmp_n+30]
                    # amp_list = np.array([amp_x, amp_y, amp_z]).T
                    amp_tol = [list(group) for group in zip(amp_x, amp_y, amp_z)]

                    # 转换Python列表到C兼容的二维数组
                    # 获取列表的尺寸
                    N = len(amp_tol)

                    # 创建C兼容的二维数组类型
                    ArrayType = (ctypes.c_double * 3) * N
                    c_array = ArrayType()

                    # 将Python list数据填充到C兼容的数组中
                    for g, row in enumerate(amp_tol):
                        c_array[g] = (ctypes.c_double * 3)(*row)
                    # DDTW matching
                    # length = ctypes.c_int(180)
                    ang_gran = 36
                    dis_gran = 5
                    test_points = ctypes.c_int(N)  
                    gt_points = ctypes.c_int(80)
                    axis = ctypes.c_int(3)
                    # 调用C函数
                    start_time = datetime.datetime.now()
                    res = ddtw_function(c_array, ang_gran, dis_gran, test_points, gt_points, axis)
                    fin_angle[i] = res
                    print("DDTW angle: ", res)
                    end_time = datetime.datetime.now()
                    print("DDTW time: ", (end_time - start_time).microseconds/1000)
                    # continue
                    # fin_angle[i], fin_dis[i] = DDTW_matching_res(amp_list, gt_datas, lat_dis=5, deg=10)
                    
                    info.append(fin_angle[i])

                    # print(fin_angle[i], fin_dis[i])

                    amp1_tmp = magnet_info[0][2][j][:3]


                    # localize and discard the repeated detection
                    if tmp_index == 0:
                        tmp_sec_index = 1
                        amp2_tmp_x = (sx[tmp_sec_index][tmp_n]) - LastRAW_x[tmp_sec_index]
                        amp2_tmp_y = (sy[tmp_sec_index][tmp_n]) - LastRAW_y[tmp_sec_index]
                        amp2_tmp_z = (sz[tmp_sec_index][tmp_n]) - LastRAW_z[tmp_sec_index]
                        amp2_tmp = [amp2_tmp_x, amp2_tmp_y, amp2_tmp_z]
                    
                    elif tmp_index == num - 1:
                        tmp_sec_index = num - 2
                        amp2_tmp_x = (sx[tmp_sec_index][tmp_n]) - LastRAW_x[tmp_sec_index]
                        amp2_tmp_y = (sy[tmp_sec_index][tmp_n]) - LastRAW_y[tmp_sec_index]
                        amp2_tmp_z = (sz[tmp_sec_index][tmp_n]) - LastRAW_z[tmp_sec_index]
                        amp2_tmp = [amp2_tmp_x, amp2_tmp_y, amp2_tmp_z] 
                    else:
                        if abs((s_sig_tol[tmp_index - 1][tmp_n])) > \
                        abs((s_sig_tol[tmp_index + 1][tmp_n])):
                            tmp_sec_index = tmp_index - 1
                            amp2_tmp_x = (sx[tmp_sec_index][tmp_n]) - LastRAW_x[tmp_sec_index]
                            amp2_tmp_y = (sy[tmp_sec_index][tmp_n]) - LastRAW_y[tmp_sec_index]
                            amp2_tmp_z = (sz[tmp_sec_index][tmp_n]) - LastRAW_z[tmp_sec_index]
                            amp2_tmp = [amp2_tmp_x, amp2_tmp_y, amp2_tmp_z]
                        else:
                            tmp_sec_index = tmp_index + 1
                            amp2_tmp_x = (sx[tmp_sec_index][tmp_n]) - LastRAW_x[tmp_sec_index]
                            amp2_tmp_y = (sy[tmp_sec_index][tmp_n]) - LastRAW_y[tmp_sec_index]
                            amp2_tmp_z = (sz[tmp_sec_index][tmp_n]) - LastRAW_z[tmp_sec_index]
                            amp2_tmp = [amp2_tmp_x, amp2_tmp_y, amp2_tmp_z]
                    
                    # print("amp1_tmp: ", amp1_tmp)
                    # print("amp2_tmp: ", amp2_tmp)
                    # offset_d, offset_d_s1 = ComputeD(amp1_tmp, amp2_tmp, theta, tmp_index, tmp_sec_index, h, l)
                    
                    double_array = (ctypes.c_double * 3)(*amp1_tmp)

                    double_array_2 = (ctypes.c_double * 3)(*amp2_tmp)
                    sol = (ctypes.c_double * 1)(0)
                    # offset_d_s1 = ctypes.c_double(0)
                    start_time = datetime.datetime.now()
                    res = res * np.pi / 180
                    offset_d_s1 = localize_function(double_array, double_array_2, res, tmp_index, tmp_sec_index, h, l, sol)
                    end_time = datetime.datetime.now()
                    # print("localize time: ", (end_time - start_time).microseconds)
                    # print("offset_d_s1: ", offset_d_s1)
                    info.append(float(offset_d_s1))
                    

                    after_ind_2 = 3

                    if info[1] - last_index > after_ind_2:
                        if tag_info == []:
                            tag_info.append(info)
                            # print("1:", tag_info)
                        else:
                            
                            flag_tag = False
                            ind_tag = 0
                            for tag_i in range(len(tag_info)):
                                if abs(tag_info[tag_i][1]- info[1]) < 6 and abs(tag_info[tag_i][3] - info[3]) < 6: 
                                    # if(tag_info[tag_i][2] - info[2] > 10):
                                    flag_tag = True
                                    ind_tag = tag_i
                                    break
                            # if flag_tag:
                            #     tag_info[ind_tag] = info

                            if not flag_tag:
                                tag_info.append(info)
                            # print("2:", tag_info)
                    info = []

                magnet_info.pop(0)
                
                
            # reconstructing the tag layout
            if(len(tag_info) == 5):
                # print(tag_info)
                # reconstruct the tag layout
                # speed: mm
                v = 100
                f = 50
                x_axis = [160-tag_info[0][3], 160-tag_info[1][3], 160-tag_info[2][3], 
                          160-tag_info[3][3], 160-tag_info[4][3]]
                        #   160-tag_info[5][3], 160-tag_info[6][3], 160-tag_info[7][3]]
                # x_axis = [v*(tag_info[1][3] - tag_info[0][3])/f]
                # y_axis = [tag_info[0][1], tag_info[1][1], tag_info[2][1]]
                y_axis = [0, (tag_info[1][1]-tag_info[0][1])/f*v, (tag_info[2][1]-tag_info[0][1])/f*v,
                          (tag_info[3][1]-tag_info[0][1])/f*v, (tag_info[4][1]-tag_info[0][1])/f*v]
                            # (tag_info[5][1]-tag_info[0][1])/f*v, (tag_info[6][1]-tag_info[0][1])/f*v,
                            # (tag_info[7][1]-tag_info[0][1])/f*v]    
            
                print(x_axis, y_axis)
                # y_axis = [0, 40, 40, 80, 80]

                # x, y axis to (x, y) points
                # points = np.array([x, y]).T
                points = [[x_axis[0], y_axis[0]], [x_axis[1], y_axis[1]], [x_axis[2], y_axis[2]],
                          [x_axis[3], y_axis[3]], [x_axis[4], y_axis[4]]]
                    # [x_axis[5], y_axis[5]],
                        #   [x_axis[6], y_axis[6]], [x_axis[7], y_axis[7]]]
                points = np.array(points)
                p1, p2, can = find_vertex_points(points)
                measured_tri = np.array([can, p1, p2])
                triangle_reference = np.array([[0, 80], [0, 0], [80, 80]])
                # heading_angle = np.arccos((x_axis[1] - x_axis[0]) / 40) * 180 / np.pi
                # print("heading_angle: ", heading_angle)
                rotation_angle = calculate_rotation_angle(triangle_reference, measured_tri)
                print("rotation angle:", rotation_angle)
                # orientation = np.arctan((y_axis[1] - y_axis[0]) / (x_axis[1] - x_axis[0])) * 180 / np.pi
                orientation = rotation_angle
                # decode the tag's polarity
                # if (orientation < 0):
                    # orientation = (orientation + 360)%360
                orient_list = [tag_info[0][2] + orientation, tag_info[1][2] + orientation, tag_info[2][2] + orientation,
                               tag_info[3][2] + orientation, tag_info[4][2] + orientation]
                                #  tag_info[5][2] + orientation,
                            #    tag_info[6][2] + orientation, tag_info[7][2] + orientation]

                phase_num = 8               # phase_to_code_map = encode_phase(phase_num)
                decode_info= []
                for phase in orient_list:
                        decoded_code = decode_phase(phase, phase_num)
                        decode_info.append(decoded_code)
                        phase = (phase) % 360
                        print(f"Phase {phase}° decodes to {decoded_code}")

                stop_flag = True
                for i in range(1):
                    if decode_info[i] != "100":
                        stop_flag = False
                
                if stop_flag:
                    print("Stop")
                    car.set_car_motion(0, 0, 0)
                    break
                # car.set_car_motion(0, 0, 0)
                last_index = tag_info[-1][1]
                tag_info = []


           

            # tag layout and orientation reconstruction
                    

    
            n = n + 1
            cnt += 1

    except KeyboardInterrupt:
        # Record the data
        print("Output csv")
        test = pd.DataFrame(columns=raw_name, data=raw_result)
        # Raw data
        test.to_csv("RawData_" + scenario + "_" + str(times) + ".csv")


        test_z = pd.DataFrame(columns=["THR_Z"], data=slope_thrd_tol)
        # Thresholds 
        test_z.to_csv("THR_tol_" + scenario + "_" + str(times) + ".csv")

        # Slope and raw data of x-axis
        for i in range(num):
            # test_slope = pd.DataFrame(columns=["Slope"], data=slope_list_tol[i])
            # test2_raw = pd.DataFrame(columns=["Raw data"], data=raw_list_tol[i])
            # test_slope.to_csv(
            #     'Slope_tol_' + scenario + "_"  + str(times) + "_Sensor" + str(
            #         i + 1) + '.csv')
            # test2_raw.to_csv(
            #     'Raw_tol_' + scenario + "_" + str(times) + "_Sensor" + str(
            #         i + 1) + '.csv')
            
            test_x = pd.DataFrame(columns=["X data"], data=sx[i])
            test_x.to_csv(
                'Smoothed_x_' + scenario + "_" + str(times) + "_Sensor" + str(
                    i + 1) + '.csv')
            test_y = pd.DataFrame(columns=["Y data"], data=sy[i])
            test_y.to_csv(
                'Smoothed_y_' + scenario + "_" + str(times) + "_Sensor" + str(
                    i + 1) + '.csv')
            test_z = pd.DataFrame(columns=["Z data"], data=sz[i])
            test_z.to_csv(
                'Smoothed_z_' + scenario + "_" + str(times) + "_Sensor" + str(
                    i + 1) + '.csv')
            test_sig = pd.DataFrame(columns=["Signal data"], data=s_sig_tol[i])
            test_sig.to_csv(
                'Smoothed_sig_' + scenario + "_" + str(times) + "_Sensor" + str(
                    i + 1) + '.csv')
          
        car.set_car_motion(0, 0, 0)
        print("Exited")


def diff_dist(a,b):
    '''
    Computing drivative differences between a and b of dimension 3
    Arguments:
        a -- dx from signal 1, numpy array of shape ( 3,  )
        b -- dy from signal 2, numpy array of shape ( 3,  )
    Result:
          -- difference of estimated derivatives of x, y
    '''
    da_x = ((a[1][0] - a[0][0]) + (a[2][0]-a[0][0])/2)/2
    da_y = ((a[1][1] - a[0][1]) + (a[2][1]-a[0][1])/2)/2
    da_z = ((a[1][2] - a[0][2]) + (a[2][2]-a[0][2])/2)/2
    db_x = ((b[1][0] - b[0][0]) + (b[2][0]-b[0][0])/2)/2
    db_y = ((b[1][1] - b[0][1]) + (b[2][1]-b[0][1])/2)/2
    db_z = ((b[1][2] - b[0][2]) + (b[2][2]-b[0][2])/2)/2

    return math.sqrt((da_x-db_x)**2 + (da_y - db_y)**2 + (da_z - db_z)**2)


def DDTW(signal_1, signal_2):
    '''
    Arguments:
        signal_1 -- first time series, numpy array of shape ( n1, 3 )
        signal_2 -- second time series, numpy array of shape ( n2, 3 )
    Results:
        ddtw -- distance matrix, numpy array of shape ( n1 - 2, n2 - 2 )
        ddtw_traceback -- traceback matrix, numpy array of shape ( n1 - 2, n2 - 2 )
    ''' 
    assert signal_1.shape[0] != 0 and signal_2.shape[0] != 0, '''Input signals must be a column vectors,
                                                                Please check the input signal dimension.'''
    assert signal_1.shape[0] >= 3 and signal_2.shape[0] >= 3, '''The length of your signal should be 
                                                                 greater than 3 to implement DDTW.'''
    n_rows = signal_1.shape[0]-2
    n_cols = signal_2.shape[0]-2

    ddtw = np.zeros((n_rows,n_cols))
    
    ddtw_traceback = np.zeros((n_rows,n_cols))

    ddtw[0,0] = diff_dist(signal_1[0:3], signal_2[0:3])


    for i in range(1, n_rows):
        ddtw[i,0] = ddtw[i-1,0] + diff_dist(signal_1[i-1:i+2], signal_2[0:3])
        ddtw_traceback[i,0] = 1
    for j in range(1, n_cols):
        ddtw[0,j] = ddtw[0,j-1] + diff_dist(signal_1[0:3], signal_2[j-1:j+2])
        ddtw_traceback[0,j] = 2
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            temp = np.array([ddtw[i-1,j-1], ddtw[i-1,j], ddtw[i,j-1]])
            best_idx = np.argmin(temp)
            ddtw[i,j] = diff_dist(signal_1[i-1:i+2], signal_2[j-1:j+2]) + temp[best_idx]
            ddtw_traceback[i,j] = best_idx
            # print(ddtw[i, j])
    return ddtw, ddtw_traceback


def get_traceback(ddtw,ddtw_traceback):
    i, j = ddtw.shape
    i -= 1
    j -= 1
    x = [i]
    y = [j]
    while (i != 0 or j != 0):
        if i != 0 and j != 0:
            idx_i = [i-1, i-1, i]
            idx_j = [j-1 ,j, j-1]
            idx = int(ddtw_traceback[i,j])
            i = idx_i[idx]
            j = idx_j[idx]
        elif i == 0 and j != 0:
            j = j-1
        elif i != 0 and j == 0:
            i = i-1
        elif i==1 and j == 1:
            i = 0
            j = 0
        x.append(i)
        y.append(j)
    # print(x, y)
    dis = 0
    for i in range(len(x)):
        dis +=  ddtw[x[i], y[i]]
    dis /= len(x)
    # plt.plot(x, y)
    # plt.show()
    return dis


def DDTW_matching_res(amp_list, gt_datas, lat_dis, deg):
    angle = 0
    fin = sys.maxsize
    for i in range(len(gt_datas)):
        gt, test = np.array(gt_datas[i]), np.array(amp_list)
        if i != 0 and i % lat_dis == 0:
            angle += deg
        res, res_2 = DDTW(gt, test)
        dis = get_traceback(res, res_2)
        if dis < fin:
            fin = dis
            fin_angle = angle
            fin_off = i % lat_dis
    return fin_angle, fin_off


import numpy as np

def encode_phase(N):
    # Calculate the number of bits per phase based on N
    num_bits = int(np.log2(N))
    
    # Create a dictionary to map phase to binary code
    phase_to_code = {
        360 / N * i: format(i, f'0{num_bits}b') for i in range(N)
    }
    return phase_to_code

def decode_phase(phase, N):
    # Define the phase boundaries
    decision_boundaries = [(360 / N * i + 360 / N / 2) % 360 for i in range(N)]
    
    # Determine the closest lower boundary
    for boundary in decision_boundaries:
        if phase < boundary:
            index = decision_boundaries.index(boundary)
            break
    else:
        index = 0  # For the case where phase is beyond the last boundary
    
    # Calculate the number of bits per phase based on N
    num_bits = int(np.log2(N))
     
    # Convert the index to binary code
    return format(index % N, f'0{num_bits}b')


if __name__ == '__main__':
    detectMag()
