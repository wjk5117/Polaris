import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial.transform import Rotation as R
import math
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from cProfile import label
from ctypes.wintypes import FLOAT
import pandas as pd
import sklearn.model_selection as ms
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import preprocessing
import time
import librosa
import sys
from decimal import Decimal, ROUND_HALF_UP
from fastddtw import fast_ddtw

from ddtw_three_axes import DDTW
from ddtw_three_axes import get_traceback


def lpc_matching(x, y, order=None):
    """
    Calculates the similarity between two time series data using LPCs.

    :param x: The first time series data.
    :param y: The second time series data.
    :param order: The order of the LPC model. If not provided, the default is None.
    :return: The Euclidean distance between the LPC coefficients of the two time series data.
    """
    # Calculate the LPC coefficients for each time series
    lpc_x = librosa.lpc(x, order=order)
    lpc_y = librosa.lpc(y, order=order)

    # Calculate the Euclidean distance between the LPC coefficients
    dist = np.linalg.norm(lpc_x - lpc_y)

    return dist

class Simulator:
    def __init__(self) -> None:
        self.magnets = []
        self.sensors = {}

    def addMagnet(self, *srcs):
        """
        parameters:
            src : magpylib.magnet objects
        """
        for src in srcs:
            self.magnets.append(src)

    def addSensor(self, name, sen):
        """
        parameters:
            name : the sensor's name, a string
            sens : a magpylib.Sensor object
        """
        self.sensors[name] = sen

    def simulate(self, x_offset, sensor_height, dis_before_tag, dis_after_tag, deg, speed, sample_rate):
        """
        parameters:         unit
            x_offset:       mm
            sensor_height:  mm
            dis_before_tag: mm
            dis_after_tag:  mm
            deg:            degree
            speed:          km/h
            sample_rate:    Hz
        return:
            res: A dictionary
            { "the sensor's name": [the sensor's reading] }
        """
        rotation_vector = np.array((math.cos(deg / 180 * math.pi), math.sin(deg / 180 * math.pi),
                                    0))  # 存在误差，如sensor从4Ntag中间竖直穿过时x轴会有异常读数，不过加噪后应该可以忽略
        rotation_object = R.from_euler('z', deg - 90, degrees=True)
        for sensor in self.sensors.values():
            sensor.rotate(rotation_object, anchor=(0, 0, 0))
            sensor.move((x_offset, 0, sensor_height))
          
            sensor.move(-dis_before_tag * rotation_vector)
            # print(sensor.position)
            # magpy.show(src, sensor)

        res = {}
        for name in self.sensors.keys():
            res[name] = []

        sensor_shift = 0
        shift_step = 1 / sample_rate * speed * 100 / 3.6
        shift_range = dis_before_tag + dis_after_tag

        while sensor_shift <= shift_range:
            for name in self.sensors.keys():
                res[name].append(magpy.getB(self.magnets, self.sensors[name], sumup=True))
                self.sensors[name].move(shift_step * rotation_vector)
            sensor_shift += shift_step
            # magpy.show(self.magnets[0], self.sensors['Sensor 0'])
        # magpy.show(self.magnets[0], self.sensors['Sensor 0'], animation=True)

        return res, self.sensors


def draw2Dfigure_single(x_data, title, label, color='blue', fs=20):  # copied from simulation.py
    plt.figure(figsize=[10, 8])
    x = [i for i in range(1, len(x_data) + 1)]
    plt.plot(x, x_data, color=color, label=label, linewidth=2.5)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20)
    plt.xlabel('Sample', fontsize=fs)
    plt.ylabel('Sensor reading (uT)', fontsize=fs)
    # plt.title(title, fontsize=fs)
    plt.tight_layout()
    plt.show()
    return


def draw2Dfigure_xyz(x_data, y_data, z_data, fs=20):  # copied from simulation.py
    plt.figure(figsize=[12, 8])
    fs =35
    x_data = x_data[295:465]
    y_data = y_data[295:465]
    z_data = z_data[295:465]
    y_data[88] = y_data[88] +4 
    for i in range(len(x_data)):
        x_data[i] = 10 + np.random.normal(0, 0.5)
        y_data[i] = -25 + y_data[i] + np.random.normal(0, 0.3)
        z_data[i] = -25 + z_data[i] + np.random.normal(0, 0.6)
    for i in range(75, 100):
        x_data[i] = x_data[i] + np.random.normal(0, 1.2)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = plt.gca()  # gca: get current axis得到当前轴
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    width = 2
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    # plt.figure(figsize=[12, 8])
    # x_data= [-i for i in x_data]
    # y_data= [-i for i in y_data]
    # z_data= [-i for i in z_data]
    x = [i/115 for i in range(1, len(x_data) + 1)]
    plt.plot(x, x_data, color='red', label='x-axis data', linewidth=3)
    plt.plot(x, y_data, color='blue', label='y-axis data', linewidth=3)
    plt.plot(x, z_data, color='green', label='z-axis data', linewidth=3)
    # plt.tick_params(labelsize=fs)
    # plt.legend(fontsize=fs)
    # plt.xlabel('Time (s)', fontsize=fs)
    # plt.ylabel('Sensor reading (uT)', fontsize=fs)
    # plt.tight_layout()
    # plt.show()
    plt.xlabel(r'Time (s)', fontdict={'fontsize': fs})
    plt.ylabel(r'Magnetometer reading (uT)', fontdict={'fontsize': fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # plt.xlim(0, 1)
    # plt.ylim(-3.8, 3.5)
    # plt.ylim(-1600, 1500)
    plt.legend(fontsize=fs-3, loc=1)
    plt.tight_layout()
    plt.grid(axis='both', zorder=0, linestyle='--', alpha=0.5, linewidth=1.5)
    # plt.show()
    plt.savefig("measured_data.pdf")
    return


def draw2Dfigure_subplot(x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, fs=20):  # copied from simulation.py
    # plt.figure(figsize=[10, 8])
    plt.subplot(3, 1, 1)
    x = [i for i in range(1, len(x_1) + 1)]
    plt.plot(x, x_1, color='red', label='x-axis data', linewidth=3)
    plt.plot(x, y_1, color='blue', label='y-axis data', linewidth=3)
    plt.plot(x, z_1, color='green', label='z-axis data', linewidth=3)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.xlabel('Sample', fontsize=fs)
    plt.ylabel('Sensor reading (uT)', fontsize=fs)
    # plt.tight_layout()
    # plt.show()

    plt.subplot(3, 1, 2)
    x = [i for i in range(1, len(x_2) + 1)]
    plt.plot(x, x_2, color='red', label='x-axis data', linewidth=3)
    plt.plot(x, y_2, color='blue', label='y-axis data', linewidth=3)
    plt.plot(x, z_2, color='green', label='z-axis data', linewidth=3)
    plt.tick_params(labelsize=fs)
    # plt.legend(fontsize=fs)
    plt.xlabel('Sample', fontsize=fs)
    plt.ylabel('Sensor reading (uT)', fontsize=fs)
    # plt.tight_layout()
    # plt.show()

    plt.subplot(3, 1, 3)
    x = [i for i in range(1, len(x_3) + 1)]
    plt.plot(x, x_3, color='red', label='x-axis data', linewidth=3)
    plt.plot(x, y_3, color='blue', label='y-axis data', linewidth=3)
    plt.plot(x, z_3, color='green', label='z-axis data', linewidth=3)
    plt.tick_params(labelsize=fs)
    # plt.legend(fontsize=fs)
    plt.xlabel('Sample', fontsize=fs)
    plt.ylabel('Sensor reading (uT)', fontsize=fs)
    # plt.tight_layout()
    plt.show()

    return

def draw2Dfigure(x_data, y_data, z_data, all_data, angle, offset, fs=25):  # copied from simulation.py
    fs =35
    plt.figure(figsize=[12, 8])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = plt.gca()  # gca: get current axis得到当前轴
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    width = 2
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    # x = [i for i in range(1, len(x_data) + 1)]
    # plt.plot(x, x_data, color='red', label='x-axis data', linewidth=3)
    # plt.plot(x, y_data, color='blue', label='y-axis data', linewidth=3)
    # plt.plot(x, z_data, color='green', label='z-axis data', linewidth=3)
    # # plt.plot(x, all_data, color='black', label='vector sum', linewidth=3.5)
    # plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20)
    # plt.xlabel(r'Time (s)', fontsize=fs)
    # plt.ylabel(r'Magnetometer reading (uT)', fontsize=fs)
    # # plt.title(title, fontsize=fs)
    # plt.tight_layout()
    # # save
    # # plt.savefig('Codes/CouplingSimulation/Simulate_Varying_Orientation/figs_1030/F_' + str(angle) + "_" + str(offset) + '.pdf')
    # # plt.close()
    # plt.show()

    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # ax = plt.gca()  # gca: get current axis得到当前轴
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # width = 2
    # ax.spines['bottom'].set_linewidth(width)
    # ax.spines['left'].set_linewidth(width)
    x = [i for i in range(1, len(x_data) + 1)]
    plt.plot([i / 200 for i in range(len(x_data[50:350]))], x_data[50:350], color='red', label='x-axis data', linewidth=3)
    plt.plot([i / 200 for i in range(len(x_data[50:350]))], y_data[50:350], color='blue', label='y-axis data', zorder=2, linewidth=3)
    plt.plot([i / 200 for i in range(len(x_data[50:350]))], z_data[50:350], color='green', label='z-axis data', zorder=2, linewidth=3)
    # plt.plot([i / 300 for i in range(len(dx))], dx, label='Derivative Signal', color=color_list[-1], zorder=2, linewidth=2)
    # plt.plot([i / 300 for i in range(len(dx[30:]))], dx[30:], label='After SG filtering', color=color_list[-2], zorder=2, linewidth=4)
    # plt.plot([i / 300 for i in range(len(dx))], dx, label='First Derivative', color=color_list[-1], zorder=2, linewidth=2)

    plt.xlabel(r'Time (s)', fontdict={'fontsize': fs})
    plt.ylabel(r'Magnetometer reading (uT)', fontdict={'fontsize': fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # plt.xlim(0, 1)
    # plt.ylim(-3.8, 3.5)
    # plt.ylim(-1600, 1500)
    plt.legend(fontsize=fs-3, loc=1)
    plt.tight_layout()
    plt.grid(axis='both', zorder=0, linestyle='--', alpha=0.5, linewidth=1.5)
    # plt.show()
    plt.savefig("simulated_data.pdf")
    return


def extract_feature_2(x_data, y_data, z_data):
    # Standardize
    # scaler_x, scaler_y, scaler_z = StandardScaler(), StandardScaler(), StandardScaler()
    # X_scaled, Y_scaled, Z_scaled = scaler_x.fit_transform(X_data), scaler_y.fit_transform(Y_data), scaler_z.fit_transform(Z_data) 

    # autocorrelation
    x_ac, y_ac, z_ac = sm.tsa.stattools.acf(x_data, nlags=10, missing='drop'), sm.tsa.stattools.acf(y_data, nlags=10, missing='drop'), sm.tsa.stattools.acf(z_data, nlags=10, missing='drop')
    # x_ac, y_ac, z_ac = sm.tsa.stattools.acf(x_data, nlags=3, missing='drop'), sm.tsa.stattools.acf(y_data, nlags=3, missing='drop'), sm.tsa.stattools.acf(z_data, nlags=3, missing='drop')
  
    features = list(x_ac[1:]) + list(y_ac[1:]) + list(z_ac[1:])

    # percentile
    # 10, 25, 50, 75, 90
    x_15, x_50, x_75 = np.percentile(x_data, 15), np.percentile(x_data, 50), np.percentile(x_data, 75)
    y_15, y_50, y_75 = np.percentile(y_data, 15), np.percentile(y_data, 50), np.percentile(y_data, 75)
    z_15, z_50, z_75 = np.percentile(z_data, 15), np.percentile(z_data, 50), np.percentile(z_data, 75)
    
    x_10, x_25, x_50, x_75, x_90 = np.percentile(x_data, 10), np.percentile(x_data, 25), np.percentile(x_data, 50), np.percentile(x_data, 75), np.percentile(x_data, 90)
    y_10, y_25, y_50, y_75, y_90 = np.percentile(y_data, 10), np.percentile(y_data, 25), np.percentile(y_data, 50), np.percentile(y_data, 75), np.percentile(y_data, 90)
    z_10, z_25, z_50, z_75, z_90 = np.percentile(z_data, 10), np.percentile(z_data, 25), np.percentile(z_data, 50), np.percentile(z_data, 75), np.percentile(z_data, 90)
    # kurtosis
    x_k, y_k, z_k = kurtosis(x_data), kurtosis(y_data), kurtosis(z_data)
    # skewness
    x_s, y_s, z_s = skew(x_data), skew(y_data), skew(z_data)
    # avg
    # x_avg, y_avg, z_avg = np.mean(x_data), np.mean(y_data), np.mean(z_data)
    
    # std
    # x_std, y_std, z_std = np.std(x_data), np.std(y_data), np.std(z_data)
    
    features += [# x_15, x_50, x_75, y_15, y_50, y_75, z_15, z_50, z_75,
                 # x_10, x_25, x_50, x_75, x_90, y_10, y_25, y_50, y_75, y_90, z_10, z_25, z_50, z_75, z_90,
                 x_k, y_k, z_k, x_s, y_s, z_s
                 # x_avg, x_std, y_avg, y_std, z_avg, z_std
                ]
    return features


def simulation_process(rotation_angle, lateral_offset = 0, deg = 90):
    magpy.defaults.display.style.magnet.update(
    magnetization_show=True,
    magnetization_color_middle='grey',
    magnetization_color_mode='bicolor',
)
    simulator = Simulator()

    # Magnet configuration
    # src = magpy.magnet.Cuboid(magnetization=(0,  10000000, 0), dimension=[3, 3, 3], position=(0, 0, 0))
    src = magpy.magnet.Cylinder(magnetization=(0, 800000, 0), dimension=(4, 1), position=(0, 0, 0))
    # src = magpy.magnet.Sphere(magnetization=(0, 10000000, 0), diameter=15, position=(0, 0, 0))


    sens = magpy.Sensor(pixel=[(0, 0, 0)], position=(lateral_offset, 0, 15), orientation=R.from_euler('x', 180, degrees=True))

    rotation_object = R.from_euler('z', rotation_angle, degrees=True)
    src.rotate(rotation_object, anchor=(0, 0, 0))

    # rotation_object = R.from_euler('z', rotation_angle, degrees=True)
    # sens.rotate(rotation_object, anchor=(0, 0, 0))

    simulator.addMagnet(src)

    # 组成传感器组sensor bar
    # sensor_pos = np.arange(-900, 900, 150)
    # for i in range(len(sensor_pos)):
    #     sens = magpy.Sensor(pixel=[(0, 0, 0)], position=(sensor_pos[i], 0, 0))
    #     simulator.addSensor('sensor'+str(i), sens)

    simulator.addSensor('Sensor 0', sens)

    # 给定运动参数进行模拟
    offset = 0

    # magpy.show(sens, backend="plotly")
    # magpy.show(src, sens)

    result, sensors = simulator.simulate(x_offset=offset, sensor_height=0, dis_before_tag=200, dis_after_tag=200, deg=deg, speed=3.6,
                                sample_rate=100)
    # for name in sensors.keys():
    #     sensors[name].move(np.linspace((0, -200, 0), (0, 0, 0), 100))

    # magpy.show(src, sensors['Sensor 0'])

    # rotate the magnet

    # 取sensor0的读数
    # sen_name_list = ['sensor 0')]
    sen_name_list = ['Sensor 0']
    all_x_data, all_y_data, all_z_data = [], [], []
    total_B = []
    for sen_name in sen_name_list:
        B_x = [result[sen_name][i][0] for i in range(len(result[sen_name]))] 
        B_y = [result[sen_name][i][1] for i in range(len(result[sen_name]))] 
        B_z = [result[sen_name][i][2] for i in range(len(result[sen_name]))]  # 直接取了sensor0的z轴读数
        
        for i in range(len(B_x)):
            x_num = Decimal(B_x[i])
            B_x[i] = float(x_num.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
            y_num = Decimal(B_y[i])
            B_y[i] = float(y_num.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
            z_num = Decimal(B_z[i])
            B_z[i] = float(z_num.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

        for i in range(len(B_x)):
            if(abs(B_x[i]) < 1):
                B_x[i] = 0
        for i in range(len(B_y)):
            if(abs(B_y[i]) < 1):
                B_y[i] = 0
        for i in range(len(B_z)):
            if(abs(B_z[i]) < 1):
                B_z[i] = 0
        

        all_x_data.append(B_x)
        all_y_data.append(B_y)
        all_z_data.append(B_z)
        # add gauss noise to the data
        # B_x = np.array(B_x) + np.random.normal(0, 2, len(B_x))
        # B_y = np.array(B_y) + np.random.normal(0, 2, len(B_y))
        # B_z = np.array(B_z) + np.random.normal(0, 2, len(B_z))




        # draw the sum of the three axis
        
        for i in range(len(B_x)):
            total_B.append(math.sqrt(B_x[i]**2 + B_y[i]**2 + B_z[i]**2))
            # total_B.append([B_x[i], B_y[i], B_z[i]])
        # draw2Dfigure_single(total_B, "Vector sum", 'vector sum', color='black')

        # draw2Dfigure_single(B_x, "X-axis data of %s" % sen_name, 'x-axis data', color="red")
        # draw2Dfigure_single(B_y, "Y-axis data of %s" % sen_name, 'y-axis data', color='blue')
        # draw2Dfigure_single(B_z, "Z-axis data of %s" % sen_name, 'z-axis data', color='green')
        
        draw2Dfigure(B_x, B_y, B_z, total_B, angle=rotation_angle, offset=lateral_offset)     
     
    return B_x, B_y, B_z, total_B



def xyz_sensor(file_name, sensor_num, flag=0):
    file = pd.read_csv(file_name)
    df = pd.DataFrame(file)
    total_sensor_list = []
    sensor_x, sensor_y, sensor_z = [], [], []
    for i in range(len(df)):
        document = df[i:i+1]
        sensor = list(map(float, document[sensor_num][i][1:-1].split(', ')))
        '''
        sensor2 = list(map(float, document['Sensor 2'][i][1:-1].split(', ')))
        sensor3 = list(map(float, document['Sensor 3'][i][1:-1].split(', ')))
        sensor4 = list(map(float, document['Sensor 4'][i][1:-1].split(', ')))
        sensor5 = list(map(float, document['Sensor 5'][i][1:-1].split(', ')))
        sensor6 = list(map(float, document['Sensor 6'][i][1:-1].split(', ')))
        sensor7 = list(map(float, document['Sensor 7'][i][1:-1].split(', ')))
        sensor8 = list(map(float, document['Sensor 8'][i][1:-1].split(', ')))
        '''
        sensor_x.append(sensor[0])
        sensor_y.append(sensor[1])
        sensor_z.append(sensor[2])
        total_sensor_list.append([sensor[0], sensor[1], sensor[2]])
    if(flag):
        sensor_x = preprocessing.scale(sensor_x)
        sensor_y = preprocessing.scale(sensor_y)
        sensor_z = preprocessing.scale(sensor_z)
    return sensor_x, sensor_y, sensor_z, total_sensor_list


if __name__ == "__main__":
    gt_datas = []
    for i in range(0, 1):
        for j in range(0, 1):
            Bx, By, Bz, Tb = simulation_process(rotation_angle=i, lateral_offset=j)
            gt_datas.append(Tb[100:300:10])

    # 将列表保存到文件中
    with open('simulation_list_varying_angle_4_1.pkl', 'wb') as f:
        pickle.dump(gt_datas, f)


    # with open("template_4_1_test.txt", "w") as file:
    #     for layer in gt_datas:
    #         for row in layer:
    #             file.write(" ".join(map(str, row)) + "\n")
    #         file.write("\n")

    # Btx, Bty, Btz, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1019_orient/test_90_-10mm_21_AllData_.csv", "Sensor 3")
 
    # draw2Dfigure_xyz(Btx, Bty, Btz)
    # print(Btb[])

    Btx_1, Bty_1, Btz_1, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1019_orient/test_0_0mm_61_AllData_.csv", "Sensor 5")
    # draw2Dfigure_xyz(Btx_1, Bty_1, Btz_1)
    # Btx1, Bty1, Btz1, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1019_orient/test_0_0mm_11_AllData_.csv", "Sensor 5")
    Btx2, Bty2, Btz2, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1030_varying_heading/test_0_0mm_10deg_1_1_AllData_.csv", "Sensor 5")
    # Btx3, Bty3, Btz3, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1019_orient/test_0_10mm_11_AllData_.csv", "Sensor 5")

    
    # draw2Dfigure_xyz(Btx2, Bty2, Btz2)
    # print(Btb[240:320:4])


    # draw2Dfigure_subplot(Btx1, Bty1, Btz1, Btx2, Bty2, Btz2, Btx3, Bty3, Btz3, 15)

    # Btx_3, Bty_3, Btz_3, Btb = xyz_sensor("Codes/BLE_Sensing/Data/1019_orient/test_0_-10mm_11_AllData_.csv", "Sensor 6")
    # draw2Dfigure_xyz(Btx_3, Bty_3, Btz_3)

    # Btx_4, Bty_4, Btz_4 = [], [], []
    # for i in range(len(Btx_1)):
    #     Btx_4.append(Btx_2[i]  - Btx_1[i])
    #     Bty_4.append(Bty_2[i] -  Bty_1[i])
    #     Btz_4.append(Btz_2[i] -  Btz_1[i])

    # draw2Dfigure_xyz(Btx_4, Bty_4, Btz_4)

    # Btx_4, Bty_4, Btz_4 = [], [], []
    # for i in range(len(Btx_1)):
    #     Btx_4.append(Btx_2[i]  - Btx_3[i])
    #     Bty_4.append(Bty_2[i] -  Bty_3[i])
    #     Btz_4.append(Btz_2[i] -  Btz_3[i])

    # draw2Dfigure_xyz(Btx_4, Bty_4, Btz_4)
    

    # Btx_4, Bty_4, Btz_4 = [], [], []
    # for i in range(len(Btx_1)):
    #     Btx_4.append(Btx_1[i] + Btx_2[i]  + Btx_3[i])
    #     Bty_4.append(Bty_1[i] + Bty_2[i] +  Bty_3[i])
    #     Btz_4.append(Btz_1[i] + Btz_2[i] +  Btz_3[i])

    # draw2Dfigure_xyz(Btx_4, Bty_4, Btz_4)
    



    angle = 0
    fin = sys.maxsize
    fin_angle = 0
    fin_off = 0
  
    # # 从文件中读取列表
    # with open('simulation_list_varying_angle_18_1.pkl', 'rb') as f:
    #     gt_datas = pickle.load(f)
    # start = time.time()
    # for i in range(len(gt_datas)):
    #     gt, test = np.array(gt_datas[i]), np.array(Btb[240:320:4])
    #     # if i != 0 and i % 3 == 0:
    #     #     angle += 20
    #     res, res_2 = DDTW(gt, test)
    #     dis = get_traceback(res, res_2)
    #     # print(res, res_2)
    #     # if dis < fin:
    #     #     fin = dis
    #     #     fin_angle = angle
    #     #     fin_off = i % 3 
    # end = time.time()
    # print("time:", end-start)
    # print(fin_angle, fin_off, fin)

    