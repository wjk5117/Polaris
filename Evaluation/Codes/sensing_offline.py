import pandas as pd
import numpy as np
import datetime
import math
import ctypes


# ------------Experiment Settings-------------
# Load the C library: ddtw.so
ddtw_c = ctypes.CDLL('./Utils/ddtw.so')
ddtw_function = ddtw_c.DDTW_matching_res
ddtw_function.argtypes = [ctypes.POINTER(ctypes.c_double * 3), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
ddtw_function.restype = ctypes.c_double
# Load the C library: localize.so
localize_c = ctypes.CDLL('./Utils/localize.so')
localize_function = localize_c.ComputeD
localize_function.argtypes = [ctypes.c_double * 3, ctypes.c_double * 3, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double * 1]
localize_function.restype = ctypes.c_double
scenario = "Offline"
# Number of sensors
num = 9
# experiment times
times = 1
# tag and sensor layout
h, l = 15, 20


# ---------Initial Parameters and Thresholds---------
# These parameters and thresholds are empirically determined for the specific sensors used
# May need to finetune to optimize results if used different sensors
wnd = 3 # Gaussian Smoother for the raw signal data
wnd_d = 1  # Gaussian Smoother for the 1st derivative data
delta_thrd_tol =[30, 31, 30, 30, 31, 31, 30, 31, 35] 
amp_thrd_tol = [1.05] * num
start_raw_amp_x = [0] * num
start_raw_amp_y = [0] * num
start_raw_amp_z = [0] * num


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

def gaussian(x, pos, wid):
    g = np.exp(-((x - pos) / (0.60056120439323 * wid)) ** 2)
    return g

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


def get_csv_data(file_name, sensor_num=9, start=-1):
    file = pd.read_csv(file_name)
    df = pd.DataFrame(file)
    sensors_x, sensors_y, sensors_z = [], [], []
    timedata = []
    for i in range(len(df)):
        document = df[i:i+1]
        sx_temp, sy_temp, sz_temp = [], [], []
        for j in range(sensor_num):
            s_data = list(map(float, document["Sensor " + str(j+1)][i][1:-1].split(', ')))
            if i > start:
                # read the raw data (x, y, z) from the csv file
                sx_temp.append(s_data[0])
                sy_temp.append(s_data[1])
                sz_temp.append(s_data[2])               
        timedata.append(float(document['Time Stamp'][i]))
        if i > start:
            sensors_x.append(sx_temp)
            sensors_y.append(sy_temp)
            sensors_z.append(sz_temp)
    return timedata, sensors_x, sensors_y, sensors_z


car_flag = True
# Detect the magnet
def detectMag(file_name, result_txt):
    time_data_csv, sensors_x_csv, sensors_y_csv, sensors_z_csv = get_csv_data(file_name, 9)
    print("Begin detecting at", str(datetime.datetime.now()))
    first = 0
    t = []
    global cnt, current, n_z, n, no, z, sz, dz, sdz, amp_tol, amp2_tol
    global x, sx, dx, sdx, y, sy, dy, sdy, sig_tol, s_sig_tol, ds_sig_tol, sd_sig_tol
    global slope_thrd_tol, slope_list_tol, estimate_tol, raw_result, raw_list_tol
    global delta_t, offset_list, fin_angle, fin_dis, magnet_info,cur_n, flag_s
    global h, l, tag_info, flag_update
    global LastRAW_tol, LastRAW_x, LastRAW_y, LastRAW_z, last_index, car_flag    

    while True:
        flag = 1
        current = time_data_csv[n]
        raw_result_tmp = [current]
        for i in range(num):
            sensors[i, 0] = sensors_x_csv[n][i]
            sensors[i, 1] = sensors_y_csv[n][i]
            sensors[i, 2] = sensors_z_csv[n][i]
            raw_result_tmp.append((sensors[i, 0], sensors[i, 1], sensors[i, 2]))
                        
        if first == 0:
            for i in range(num):
                tmp = np.sqrt(sensors[i, 0] ** 2 + sensors[i, 1] ** 2 + sensors[i, 2] ** 2)
                if math.isnan(tmp) or tmp > 1000:
                    flag = 0
        if flag == 1:
            if(first == 0):
                first = 1
                print("Initialize the threshold")
            for i in range(num):
                x[i].append(sensors[i, 0]-start_raw_amp_x[i])
                y[i].append(sensors[i, 1]-start_raw_amp_y[i])
                z[i].append(sensors[i, 2]-start_raw_amp_z[i])
                tol = np.sqrt((sensors[i,0]-start_raw_amp_x[i]) ** 2 + (sensors[i,1]-start_raw_amp_y[i]) ** 2 + (sensors[i,2]-start_raw_amp_z[i]) ** 2)
                sig_tol[i].append(tol)
        else:
            print("Invalid data")
            continue
        
        raw_result.append(raw_result_tmp)
        t.append(current)

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
                count = 0
                if(car_flag):
                    for i_ind in range(num):
                        # if total slope_list_tol is larger than 35, start moving
                        if(len(slope_list_tol[i_ind]) > 30):
                            count += 1
                        if(count == num):
                            print("Begin detecting process")
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
                        # print("Sensor %d: detct a magnet, tol-amp: %.2f" % (i + 1, amp_tol))
                        # print("n: ", n)
                        no += 1
        if flag_s:
            magnet_info.append([cur_n, ind_tmp, amp_tmp_list])
            flag_s = False

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
                # print("DDTW angle: ", res)
                end_time = datetime.datetime.now()
                # print("DDTW time: ", (end_time - start_time).microseconds/1000)
                info.append(fin_angle[i])

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
                    abs((s_sig_tol[tmp_index + 1][tmp_n])-15):
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
            
       
        if(len(tag_info) >= 3 and len(tag_info) == 5):
            v = 105
            x_axis, y_axis, points = [], [], []
            for i in range(len(tag_info)):
                x_axis.append(160-tag_info[i][3])
                y_axis.append((tag_info[i][1]-tag_info[0][1])/40*v)
                points.append([x_axis[i], y_axis[i]])

            points = np.array(points)
            p1, p2, can = find_vertex_points(points)
            # print(x_axis, y_axis)
            
            orient_list = []
            if(can is not None):
                measured_tri = np.array([can, p1, p2])
                triangle_reference = np.array([[0, 80], [0, 0], [80, 80]])
                rotation_angle = calculate_rotation_angle(triangle_reference, measured_tri)
                orientation = rotation_angle
                for i in range(len(tag_info)):
                    orient_list.append(tag_info[i][2] + orientation)
                #

                print("Estimated heading angle: " + str(rotation_angle) + " deg")
                localization_err, decode_info = decode_tag(points, orient_list)
                print("The localization error is: " + str(localization_err) + " mm")
                print("The decoded bits are:", decode_info)
                tag_id = int(''.join(decode_info), 2)
                print("The corresponding tag id is:", tag_id)

                # write the four points to the txt file
                with open(result_txt, "a") as f:
                    f.write("Estimated heading angle: " + str(rotation_angle) + " deg\n")
                    f.write("The localization error is: " + str(localization_err) + " mm\n")
                    f.write("The decoded bits are: " + str(decode_info) + "\n")
                    f.write("The corresponding tag id is: " + str(tag_id) + "\n\n")


                last_index = tag_info[-1][1]
                tag_info = []

        n = n + 1
        cnt += 1
        # emulating the real-time detection with 100 Hz
        # time.sleep(0.01)
        if (n == len(time_data_csv)):
            print("End detection")
            break


def decode_tag(points, orient_list):
    magnet_id = []
    for x, y in points:
        x = x - points[0][0]
        if abs(x) <= 20 and abs(y) <= 20:
            magnet_id.append(1)
        elif 20 < abs(x) <= 60 and abs(y) <= 20:
            magnet_id.append(2)
        # elif 60 < abs(x) <= 100 and abs(y) <= 20:
        #     magnet_id.append(3)
        elif abs(x) <= 20 and 20 < abs(y) <= 40:
            magnet_id.append(3)
        elif 20 < abs(x) <= 60 and 20 < abs(y) <= 60:
            magnet_id.append(4)
        elif 60 < abs(x) <= 100 and 20 < abs(y) <= 60:
            magnet_id.append(5)
        elif abs(x) <= 20 and 60 < abs(y) <= 100:
            magnet_id.append(6)
        elif 20 < abs(x) <= 60 and 60 < abs(y) <= 100:
            magnet_id.append(7)
        elif 60 < abs(x) <= 100 and 60 < abs(y) <= 100:
            magnet_id.append(8)
 
    decode_mag = []
    for i in range(len(orient_list)):
        decode_mag.append((magnet_id[i], orient_list[i], points[i]))
    # sort the decode_info by the magnet_id
    decode_mag = sorted(decode_mag, key=lambda x: x[0])

    phase_num = 8               # phase_to_code_map = encode_phase(phase_num)
    decode_info= []
    for mag in decode_mag:
            phase = mag[1]
            decoded_code = decode_phase(phase, phase_num)
            decode_info.append(decoded_code)
            phase = (phase) % 360
            # print(f"Phase {phase}° decodes to {decoded_code}")

    decode_id = [id for id, _, _ in decode_mag]
    
    decode_spatial = ""
    if (decode_id == [1, 4, 5, 6, 8]):
        decode_spatial = "111"
    elif (decode_id == [1, 2, 3, 6, 8]):
        decode_spatial = "000"
    elif (decode_id == [1, 2, 4, 6, 8]):
        decode_spatial = "001"
    elif (decode_id == [1, 2, 5, 6, 8]):
        decode_spatial = "010"
    elif (decode_id == [1, 2, 6, 7, 8]):
        decode_spatial = "011"
    elif (decode_id == [1, 3, 4, 6, 8]):
        decode_spatial = "100"
    elif (decode_id == [1, 3, 5, 6, 8]):
        decode_spatial = "101"
    elif (decode_id == [1, 3, 6, 7, 8]):
        decode_spatial ="110"
    
    measured_points = []
    gt_points = [[0, 0], [40, 40], [80, 40], [0, 80], [80, 80]]
    for i in range(len(decode_mag)):
        measured_x = decode_mag[i][2][0]-decode_mag[0][2][0]
        measured_y = decode_mag[i][2][1]
        measured_points.append([measured_x, measured_y])
    measured_points = np.array(measured_points)
    localization_err = calculate_distance(gt_points, measured_points)

    decode_info.append(decode_spatial)

    return localization_err, decode_info



def calculate_distance(ground_truth, predicted):
    distance = 0
    for i in range(5):
        distance += np.sqrt((ground_truth[i][0] - predicted[i][0]) ** 2 + (ground_truth[i][1] - predicted[i][1]) ** 2)
    return distance/5

# 计算两点之间的距离
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 找到距离最远的两个点
def find_vertex_points(points):
    max_dist = 0
    p1, p2 = points[0], points[1]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = distance(points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                p1, p2 = points[i], points[j]
    candidates = find_third_vertex(points, p1, p2)

    if candidates is None:
        return p1, p2, None
    else:
        return list(p1), list(p2), list(candidates)

def find_third_vertex(points, p1, p2):
    candidate = None
    for point in points:
            if np.allclose(point, p1) or np.allclose(point, p2):
                continue
            angle = calculate_angle(p1, point, p2)
            if np.isclose(angle, 90, atol=15):
                candidate = point

    return candidate
    

def calculate_angle(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi


def calculate_rotation_angle(triangle1, triangle2):
    center1 = np.mean(triangle1, axis=0)
    center2 = np.mean(triangle2, axis=0)
    triangle1_centered = triangle1 - center1
    triangle2_centered = triangle2 - center2

    vector1 = triangle1_centered[0] 
    vector2 = triangle2_centered[0] 
    
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    
    cross_product = np.cross(unit_vector1, unit_vector2)
    
    if cross_product < 0:
        angle = -angle
    
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def decode_phase(phase, N):
    # phase_to_code = encode_phase(N)
    # Define the phase boundaries
    decision_boundaries = [(360 / N * i + 360 / N / 2) % 360 for i in range(N)]
    
    # Determine the closest lower boundary
    for boundary in decision_boundaries:
        if phase < boundary:
            index = decision_boundaries.index(boundary)
            # print(index)
            break
    else:
        index = 0  # For the case where phase is beyond the last boundary
    
    # Calculate the number of bits per phase based on N
    num_bits = int(np.log2(N))
    # gray code
    phase_to_code = {
        0: '111',
        1: '110',
        2: '010',
        3: '011',
        4: '001',
        5: '000',
        6: '100',
        7: '101'
    }
    return phase_to_code[index % N]



if __name__ == '__main__':

    # the offline sensing pipeline for the default tag experiment
    default_tag_file = "../Data/Default_Tag/RawData_1.csv"
    result_txt = "../Results/Default_Tag_Exp.txt"
    detectMag(default_tag_file, result_txt)
    

    # varying_polarity_file = "../Data/Varying_Polarity_Orientations/RawData_10.csv"
    # result_txt = "../Results/Varying_Polarity_Orientations_Exp.txt"
    # detectMag(varying_polarity_file, result_txt)

    # iron_filling_file = "../Data/Iron_Fillings/RawData_1.csv"
    # result_txt = "../Results/Iron_Fillings_Exp.txt"
    # detectMag(iron_filling_file, result_txt)
