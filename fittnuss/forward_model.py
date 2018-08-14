"""
Forward model for FITTNUSS
"""
import numpy as np
from numba.decorators import jit
from scipy import interpolate as ip
#from scipy import optimize as opt
import sys
from configparser import ConfigParser as scp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time

Ds = np.array([[354*10**-6],[177*10**-6],[88.4*10**-6]]) #representative diameters of grain-size classes
cnum = len(Ds) #number of grain-size classes
nu = 1.010 * 10 ** -6 #kinetic viscosity of water (m^2 / s)
R = 1.65 #submerged specific density of water
Cf = 0.004 # 1　#friction coefficient (dimensionless Chezy coeff.)
ngrid = 50 #number of grids for transforming coordinate
sp_grid_num = 100 #number of grids for fixed coordinate
lambda_p = 0.4 #sediment porocity
dt = 0.0001 #time step length
x0 = 10. # m　#initial location of the flow head
g = 9.81 # m/s^2 gravity acceleration

spoints = [] #m location of sampling points (distance from the coastline)
deposit = [] #volume-per-unit-area of each grain-size class at each location
Rw = 0.
U = 0.
H = 0.
T = 0.
C0 = []
ws = []

def read_setfile(configfile):
    """
    A function for parsing a confuguration file to set basic parameters
    """
    global nu, R, Cf, ngrid, sp_grid_num, lambda_p, dt, x0, g, Ds, cnum
    parser = scp()
    parser.read(configfile)#read a configuration file

    nu = parser.getfloat("Physical variables","nu") #kinematic viscosity
    R = parser.getfloat("Sediments","R") #submerged spec. density of sediment
    Cf = parser.getfloat("Physical variables","Cf")#friction coefficient
    ngrid = parser.getint("Calculation","ngrid") #number of grids for a flow
    sp_grid_num = parser.getint("Calculation","sp_grid_num") #num of grids
    lambda_p = parser.getfloat("Sediments","lambda_p") #sediment porocity
    dt = parser.getfloat("Calculation","dt") #time step length
    x0 = parser.getfloat("Calculation","x0") #initial location of flow head
    g = parser.getfloat("Physical variables","g") #gravity acceleration

    Ds_text = parser.get("Sediments","Ds") #grain sizes as strings
    #converting strings of grain diameter to numbers
    Ds_micron = [float(x) for x in Ds_text.split(',') if len(x) !=0]
    Ds = np.array(Ds_micron) * 10 ** -6 #converting micron to m
    Ds = np.array(np.matrix(Ds).T) #transpose the matrix
    cnum = len(Ds) #number of grain-size classes

def set_spoint_interval(sp_grid_num):
    """
    A function to set locations of sampling points
    """
    global spoints
    spoints = np.linspace(0, Rw, sp_grid_num)

def forward(optim_params):
    """
    Calculate the forward model
    the model calculates depositon both from tsunami inundation flow and 
    stagnant phases of the flow
    
    Parameters
    ----------
    optim_params: a numpy array
    
    Returns
    -------
    x: ndarray
        location of transforming grids of flows
    C: ndarray (number of grain-size classes, num of trans. grids)
        sediment concentration at x
    spoints: ndarray
        x coordinates for deposits
    deposit: ndarray (number of grain-size classes, num of trans. grids)
        volume-per-unit-area of each grain-size class

    """
    global deposit
    
    set_params(optim_params) #set initial parameters

    #making initial arrays
    cnum = len(Ds) #number of grain-size classes
    #spatial coordinates in dimensionless transforming grid (0-1.0) 
    x_hat = np.linspace(0, 1.0, ngrid)
    dx = x_hat[1] - x_hat[0]#grid spacing in dimensionless trans. grid
    C = C0 * (1.0 - x_hat) #initial sed. concentration (dimensionless space)
    deposit = np.zeros((cnum,len(spoints))) #initial dep. thick. (real space)
    
    #grain size fractions in active layer on the top of deposit
    Fi_r = np.ones((cnum,len(spoints)))/cnum
    #bed aggradation rate. i.e. \frac{d eta}{dt}
    detadt = np.zeros((cnum,ngrid))
    t_hat = x0 / Rw #dimensionless time
    #arrays to record grain size fractions and agg. rate at previous time step
    dFi_r_dt_prev = []
    detadt_r_prev = []
    
    Cmax = np.max(C0) #a variable to check divergence of calculation

    while (t_hat < 1.0) and (Cmax < 1.0):
        t_hat += dt #increment the dimensionless time
        #calculation of 1 time step
        (C, deposit, Fi_r, detadt, dFi_r_dt_prev, detadt_r_prev) =\
            step(t_hat, x_hat, C, dt, dx, \
                                     deposit, \
                                     Fi_r, detadt, dFi_r_dt_prev, detadt_r_prev)
        #check divergence of calculation
        Cmax = np.max(np.absolute(C))

    if t_hat < 1.0: #abort when calculation diverged
        deposit = deposit / t_hat * 100
        sys.stderr.write('C Exceeded 1.0\n')

    x = Rw * x_hat #convert dimensionless coord. to real space
    #deposition after the termination of flow inundation
    deposit = get_final_deposit(x, C, spoints, deposit)

    #return values of x coord. of flows, concentration, //
    #    x coord. of real space and deposit thickness
    return (x, C, spoints, deposit)

def get_final_deposit(x, C, spoints, deposit):
    """
    calculation of deposition after termination of inundation flow 

    Parameters
    ----------
    x: ndarray
    C: ndarray
    spoints: ndarray
    deposit: ndarray

    Return
    ----------
    deposit: 2d ndarray
    
    """
    h = - H / Rw * spoints + H
    f = ip.interp1d(x, C, kind='linear', bounds_error=False, fill_value=0.0)
    C_r = f(spoints)
    deposit += (h * C_r) / (1 - lambda_p)
    return deposit

def step(t_hat, x_hat, C, dt, dx, \
                                     deposit,\
                                     Fi_r, detadt, dFi_r_dt_prev, detadt_r_prev):
    """
    1時間ステップ分の堆積量ならびに濃度変化の計算を行う
    """
    
    if len(dFi_r_dt_prev) == 0:#最初のステップはルンゲクッタ
        deposit, Fi_r, detadt_r_prev, dFi_r_dt_prev = step_RK(C, deposit, Fi_r, t_hat, x_hat)

    else:#２ステップ目以降はアダムス・バッシュフォースの予測子修正子法
        Fi_r, deposit, dFi_r_dt_prev, detadt_r_prev = step_AB_PC(C, Fi_r, deposit, t_hat, x_hat, dFi_r_dt_prev, detadt_r_prev)

    #濃度の移流に関して陰解法のステップを実行
    C_new = step_implicit_C(t_hat, x_hat, C, dt, dx, spoints, Fi_r)
    
    return C_new, deposit, Fi_r, detadt, dFi_r_dt_prev, detadt_r_prev

def step_RK(C, deposit, Fi_r, t_hat, x_hat):
    """
    ルンゲ・クッタ法による実空間でのALayer粒度分布と堆積量の計算
    """
    
    dt_r = T * dt #実座標系でのタイムステップ
    
    #水深を求める    
    h_max = H * t_hat
    h = - h_max * x_hat + h_max

    #Active Layerの厚さ
    u_star = get_u_star(C,h)
    La = get_La(x_hat, t_hat, u_star)
    
    #ルンゲ・クッタ第1ステップ
    detadt_r1, detadt_r_sum1 = get_detadt_r(C, Fi_r, t_hat, x_hat)
    k1 = 1 / La * (detadt_r1 - Fi_r * detadt_r_sum1)
    
    #ルンゲ・クッタ第2ステップ
    Fi_r_k2 = Fi_r + k1 * dt_r / 2
    detadt_r2, detadt_r_sum2 = get_detadt_r(C, Fi_r_k2, t_hat, x_hat)
    k2 = 1 / La * (detadt_r2 - Fi_r_k2 * detadt_r_sum2)

    #ルンゲ・クッタ第3ステップ
    Fi_r_k3 = Fi_r + k2 * dt_r / 2
    detadt_r3, detadt_r_sum3 = get_detadt_r(C, Fi_r_k3, t_hat, x_hat)
    k3 = 1 / La * (detadt_r3 - Fi_r_k3 * detadt_r_sum3)
    
    #ルンゲ・クッタ第4ステップ
    Fi_r_k4 = Fi_r + k3 * dt_r
    detadt_r4, detadt_r_sum4 = get_detadt_r(C, Fi_r_k4, t_hat, x_hat)
    k4 = 1 / La * (detadt_r4 - Fi_r_k4 * detadt_r_sum4)
    
    #勾配値
    dFi_r_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    detadt_r = (detadt_r1 + 2 * detadt_r2 + 2 * detadt_r3 + detadt_r4)/6
    
    #次回の値の算出    
    Fi_r = Fi_r + dt_r * dFi_r_dt
    Fi_r[Fi_r<0.0] = 0
    Fi_r[Fi_r>1.0] = 1.0
    deposit = deposit + dt_r * detadt_r
    
    return deposit, Fi_r, detadt_r, dFi_r_dt

def get_detadt_r(C, Fi_r, t_hat, x_hat):
    """
    C, Fi_rから実空間スケールでのdetadt_rを求める関数
    """
    x = Rw * t_hat * x_hat #実座標に変換
    
    #水深を求める    
    h_max = H * t_hat
    h = - h_max * x_hat + h_max
    
    #摩擦速度を算出
    u_star = get_u_star(C, h)
    
    #実空間での粒度分布Fi_rを移動座標系での粒度分布Fiへ変換する
    f2 = ip.interp1d(spoints, Fi_r, kind='linear', bounds_error=False, fill_value=0.0)
    Fi = f2(x)
    Fi[Fi<0] = 0
    Fi[Fi>1] = 1.0
    
    #堆積物連行係数を計算（移動座標系）
    Es = get_Es2(h, u_star)

    #移動座標系での堆積速度detadtを求める
    r0 = get_r0_corrected(C, Fi, u_star)
    detadt = ws * (r0 * C - Fi * Es) / (1 - lambda_p)

    #線形補間でサンプリング点における堆積速度を推定
    f = ip.interp1d(x, detadt, kind='linear', bounds_error=False, fill_value=0.0)
    detadt_r = f(spoints)
    detadt_r[detadt_r<0] = 0
    detadt_r_sum = np.sum(detadt_r, axis=0)#総堆積速度
    
    return detadt_r, detadt_r_sum

def step_AB_PC(C, Fi_r, deposit, t_hat, x_hat, dFi_r_dt_prev, detadt_r_prev):
    """
    2次のアダムス・バッシュフォースの予測子修正子法で粒度分布と堆積量を求める
    """
    
    dt_r = dt * T #実時間に変換
    
    #水深を求める    
    h_max = H * t_hat
    h = - h_max * x_hat + h_max
    
    #Active Layerの厚さ
    u_star = get_u_star(C,h)
    La = get_La(x_hat, t_hat, u_star)
        
    #現時刻での堆積速度
    detadt_r, detadt_r_sum = get_detadt_r(C, Fi_r, t_hat, x_hat)
    
    #現時刻でのアクティブレイヤー粒度分布変化率
    dFi_r_dt = 1 / La * (detadt_r - Fi_r * detadt_r_sum)
    
    #予測子
    Fi_rp = Fi_r + dt_r * (1.5 * dFi_r_dt - 0.5 * dFi_r_dt_prev)
    depositp = deposit + dt_r * (1.5 * detadt_r - 0.5 * detadt_r_prev)
    
    #予測子の示す未来の堆積速度
    detadt_rp, detadt_r_sump = get_detadt_r(C, Fi_rp, t_hat, x_hat)
    
    #予測子の示す未来のアクティブレイヤー粒度分布変化率
    dFi_r_dtp = 1 / La * (detadt_rp - Fi_rp * detadt_r_sump)
    
    #修正子
    Fi_rc = Fi_r + dt_r * (1.5 * dFi_r_dtp - 0.5 * dFi_r_dt)
    depositc = deposit * dt_r * (1.5 * detadt_rp - 0.5 * detadt_r)
    
    #解
    Fi_r = 1 / 2 * (Fi_rp + Fi_rc)
    Fi_r[Fi_r<0.0] = 0
    Fi_r[Fi_r>1.0] = 1.0
    deposit = 1 / 2 * (depositp + depositc)
    deposit[deposit<0] = 0
    
    return Fi_r, deposit, dFi_r_dt, detadt_r

@jit('f8[:,:](f8,f8[:],f8[:,:],f8,f8,f8[:],f8[:,:])')
def step_implicit_C(t_hat, x_hat, C, dt, dx, spoints, Fi_r):
    """
    陰解法によって濃度分布の計算を行うための関数
    """
    C_new = np.zeros(C.shape)
    C_new[:,0] = C0[:,0]

    #水深を計算    
    h_max = H * t_hat
    h = - h_max * x_hat + h_max
    
    #摩擦速度を算出
    u_star = get_u_star(C, h)
    
    #線形補間によって次ステップの計算グリッド上での底面粒度分布を求める
    x = Rw * t_hat * x_hat #移動座標系を実座標に変換
    f2 = ip.interp1d(spoints, Fi_r, kind='linear', bounds_error=False, fill_value=0.0)
    Fi = f2(x)
    Fi[Fi<0] = 0
    Fi[Fi>1] = 1.0
    
    #堆積物の連行係数を計算
    Es = get_Es2(h, u_star)
    #Es = get_Es4(h, Fi)
    
    #陰解法により次ステップの濃度分布を計算
    r0 = get_r0_corrected(C, Fi, u_star)
    U_hat = ( 1 - x_hat) / (t_hat) #1D Vector
    r = U_hat * dt / dx #1D Vector
    Sc = np.zeros((len(ws),len(x_hat)))
    Sc[:,0:-1] = Rw * ws / (U * H * t_hat * (1 - x_hat[0:-1])) #2D Matrix
    A = Sc * Fi * Es * dt #2D Matrix
    B = 1 + r + Sc * r0 * dt #2D Matrix
    for i in range(1, C.shape[1]-1):
        C_new[:,i] = (C[:,i] + r[i] * C_new[:,i-1] + A[:,i]) / B[:,i]

    return C_new
    

def set_params(optim_params):
    """
    最適化すべき初期パラメーターを設定する関数
    [C0, Rw, U, H]という配列を与える
    C0: 初期濃度
    Rw: 最大浸水距離
    U:　津波の鉛直平均流速
    H: 海岸線での最大浸水深
    optim_paramsのパラメーター順：
    0→Rw
    1→U
    2→H
    3以降→各粒度階の初期濃度
    """
    global Rw, U, H, T, C0, ws
    
    #規格化のための数値
    Rw = optim_params[0]
    U = optim_params[1]
    H = optim_params[2]
    T = Rw / U

    #堆積作用の計算グリッドを与える
    set_spoint_interval(sp_grid_num)

    #各階級の境界濃度を取得
    C0 = np.zeros((cnum,1))
    for i in range(cnum):
        C0[i] = optim_params[3 + i]
    

    #沈降速度
    ws = get_settling_vel(Ds, nu, g, R)


def get_La(x_hat,t_hat, u_star):
    x = Rw * t_hat * x_hat #移動座標系を実座標に変換
    f2 = ip.interp1d(x, u_star, kind='linear', bounds_error=False, fill_value=0.0)
    u_star_r = f2(spoints)
    La = np.ones(len(u_star_r))
    La[u_star_r>0] = u_star_r[u_star_r>0] ** 2 / (R * g ) / (0.1 * np.tan(0.5236))
    return La

def get_settling_vel(D, nu, g, R):
    """
    堆積物の沈降速度を計算する関数
    Dietrich et al. (1982)
    """
        
    Rep = (R * g * D) ** 0.5 * D / nu; #粒子レイノルズ数
    b1 = 2.891394; b2 = 0.95296; b3 = 0.056835; b4 = 0.002892; b5 = 0.000245
    Rf = np.exp(-b1 + b2 * np.log(Rep) - b3 * np.log(Rep) ** 2 - b4 * np.log(Rep) ** 3 + b5 * np.log(Rep) ** 4)
    ws = Rf * (R * g * D) ** 0.5
    
    return ws

def get_r0(u_star):
    """
    底面近傍濃度/鉛直平均濃度比
    Parker (1982)
    """
    r0 = 1 + 31.5 * (u_star / ws) ** (-1.46)
    return r0
    

def get_Es(u_star):
    """
    Sediment Entrainment Functionを計算する関数
    Garcia and Parker (1991)
    """
    p = 0.1

    Rp = (R * g * Ds) ** 0.5 * Ds / nu
    alpha1 = np.ones(Rp.shape) * 1.0
    alpha2 = np.ones(Rp.shape) * 0.6
    alpha1[Rp<2.36] = 0.586
    alpha2[Rp<2.36] = 1.23    
    
    A = 1.3 * 10 ** -7
    Zu = alpha1 * u_star * Rp ** alpha2 / ws
    Es = p * A * Zu ** 5.0 / (1 + A / 0.3 * Zu ** 5) * np.ones((cnum,ngrid))
    return Es

def get_Es2(h, u_star):
    """
    Sediment Entrainment Functionを計算する関数
    Wright and Parker (2004)
    """
    p = 0.1

    Rp = (R * g * Ds) ** 0.5 * Ds / nu
    alpha1 = np.ones(Rp.shape) * 1.0
    alpha2 = np.ones(Rp.shape) * 0.6
    alpha1[Rp<2.36] = 0.586
    alpha2[Rp<2.36] = 1.23
    
    A = 7.8 * 10 ** -7
    Sf = np.zeros(h.shape)
    Sf[h>0] = u_star[h>0] ** 2 / (g * h[h>0])
    Zu = alpha1 * u_star * Rp ** alpha2 / ws * Sf ** 0.08
    Es = p * A * Zu ** 5.0 / (1 + A / 0.3 * Zu ** 5) * np.ones((cnum,ngrid))
    return Es
    
def get_Es3(u_star):
    """
    Sediment Entrainment Function
    Dufois and Hur (2015)
    """
    
    tau_star = u_star ** 2 / (R * g * Ds)
    Rp = (R * g * Ds) ** 0.5 * Ds / nu
    tau_star_c = (0.22 * Rp ** (-0.6) + 0.06 * np.exp(-17.77 * Rp ** (-0.6)))
    #tau_star_c = 0.03
    
    Es = np.zeros(Ds.shape)
    E0 = 59658 * Ds - 9.86
    E0[Ds<180*10**-6] = 10 ** (18586 * Ds[Ds<180*10**-6] - 3.38)
    
    E = E0 * (tau_star / tau_star_c - 1) ** 0.5
    Es = E / 2650 / ws
        
    return Es

def get_Es4(h, Fi, u_star):
    """
    Sediment Entrainment Function
    van Rijn (1984)
    """
    
    Fi[Fi>1.0] = 1.0
    Fi[Fi<0.0] = 0.0
    d50 = np.sum(Fi * Ds)
    
    tau_star = u_star ** 2 / (R * g * Ds)
    Rp = (R * g * Ds) ** 0.5 * Ds / nu
    tau_star_c = (0.22 * Rp ** (-0.6) + 0.06 * np.exp(-17.77 * Rp ** (-0.6)))
    Tr = (tau_star/tau_star_c - 1)
    a = 0.01
    Es = 0.015 * d50 / a * Tr ** 1.5 / (Rp ** (2/3)) ** 0.3
    Es[Es>0.05] = 0.05
    
    return Es

#@jit('f8[:](f8[:,:],f8[:])')
def get_u_star(C, h):
    """
    Friction Velocityを計算する関数
    """
#    alpha = 0.1
#    K = np.ones(ngrid) * Cf * U ** 2 / alpha  #Kの初期値
#    #K = np.zeros(ngrid)
#    
#    for i in range(3):
#        fk = fK(K,C,h)
#        fkprime = fKprime(K)
#        K = - fk / fkprime + K
#        
#    u_star = np.sqrt(alpha * K)

    u_star = np.ones(ngrid) * Cf ** 0.5 * U
    return u_star

def fK(K, C, h):
    """
    Kを求める方程式
    """
    alpha = 0.1
    fk = - alpha ** (3/2) * Cf ** (-0.5) / U * K ** (3/2) + alpha * K \
         - np.sum(R * g * ws * C * h / U, axis=0)
    
    return fk
    
def fKprime(K):
    """
    Kを求める方程式の勾配
    """
    alpha = 0.1
    fkprime = - (3/2) * alpha ** (3/2) * Cf ** (-0.5) / U * K ** 0.5 + alpha
    
    return fkprime

def export_result(filename1, filename2):
    """
    サンプルデータを作るためのルーチン
    """
    np.savetxt(filename1, spoints, delimiter=',')
    np.savetxt(filename2, deposit, delimiter=',')

def plot_result(C, deposit):
    x_hat = np.linspace(0, 1.0, ngrid)#仮想空間でのグリッド座標
    x = x_hat * Rw#実座標
    
    #層厚のプロット
    plt.subplot(2,1,1)
    for i in range(cnum):
        d = Ds[i].tolist()[0]*10**6
        labelname = '{0:.0f} $\mu$m'.format(d)
        plt.plot(spoints, deposit[i,:], 'o', label=labelname)
    plt.plot(spoints, np.zeros(spoints.shape),'-', label='Original Surface')
    plt.xlabel('Distance from the shoreline (m)')
    plt.ylabel('Deposit Thickness (m)')
    plt.xlim(0,spoints[-1])
    plt.legend()

    #濃度のプロット
    plt.subplot(2,1,2)
    for i in range(cnum):
        d = Ds[i].tolist()[0]*10**6
        labelname = '{0:.0f} $\mu$m'.format(d)
        plt.plot(x, C[i,:]*100., '-', label=labelname)
    plt.xlabel('Distance from the shoreline (m)')
    plt.ylabel('Concentration (%)')
    plt.xlim(0,spoints[-1])
    plt.legend()

    
    plt.show()
    
def plot_result_thick(spointfilename, depfilenames, labels, symbollist):
    """
    結果のデータを層厚および粒度分布という形で表示する関数
    """    
    
    #データをdepositlistに読み込む
    depositlist = []
    spoints=np.loadtxt(spointfilename,delimiter=",")
    for i in depfilenames:
        dep = np.loadtxt(i,delimiter=",")
        depositlist.append(dep)
    
    #プロットの書式を整える
    plt.figure(num=None, figsize=(7, 2), dpi=150, facecolor='w', edgecolor='k')
    fp = FontProperties(size=9)
    plt.rcParams["font.size"] = 9
    
    plt.subplot(1,1,1)
    for i in range(len(depfilenames)):
        totalthick = np.sum(depositlist[i],axis=0).T
        plt.plot(spoints, totalthick, symbollist[i], color = "k", linewidth=0.75,label=labels[i])
    plt.xlabel('Distance from the shoreline (m)')
    plt.ylabel('Deposit Thickness (m)')
    #plt.xlim(0,x[-1])
    #plt.legend(prop = fp, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.legend(prop = fp, loc='best', borderaxespad=1)
    plt.tight_layout()
    plt.show()

def plot_result_gdist(spointfilename, depfilenames, labels, gsizelabels, loc, symbollist):
    
    #データをdepositlistに読み込む
    spoints=np.loadtxt(spointfilename,delimiter=",")
    depositlist = []
    for i in depfilenames:
        dep = np.loadtxt(i,delimiter=",")
        depositlist.append(dep)

    #粒度分布のプロット
    #プロットの書式を整える
    plt.figure(num=None, figsize=(7, 2), dpi=150, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.1, bottom=0.21, right=0.82, top=0.85, wspace=0.25, hspace=0.2)
    fp = FontProperties(size=9)
    plt.rcParams["font.size"] = 9

    #３地点でのプロットをおこなう
    for i in range(3):
        plt.subplot(1,3,i+1)
        gdist = np.zeros(len(gsizelabels))
        for j in range(len(depfilenames)):
            totalthick = np.sum(depositlist[j],axis=0)
            gdistlocal = depositlist[j][:,loc[i]]/totalthick[loc[i]] * 100
            gdist[1:len(gdistlocal)+1]=gdistlocal
            plt.plot(gsizelabels,  gdist, symbollist[j], color="k", linewidth=0.75, label=labels[j])
        plt.xlabel('Grain Size ($\phi$)')
        if i == 0:
            plt.ylabel('Fraction (%)', fontsize = 9)
        
        plt.title('{0:.0f} m'.format(spoints[loc[i]]))
        if i == 2:
            plt.legend(prop = fp, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()
    
def get_r0_corrected(C, Fi, u_star):
    """
    底面近傍濃度/鉛直平均濃度比
    Rouse Distributionに
    van Rijn (1984)の補正を加える
    """

    Z = ws / u_star / 0.4
    r0 = (1.16 + 7.6 * Z ** 1.59) * 1
    for i in range(2):
        Ca_sum = np.sum(r0 * C, axis=0)
        psi = 2.5 * (ws / u_star) ** 0.8 * (Ca_sum / (1 - lambda_p)) ** 0.4
        Z = ws / u_star / 0.4 + psi
        r0 = (1.16 + 7.6 * Z ** 1.59) * 1
    
    return r0

def convert_manningn2Cf(n, h):
    """
    Manning's nを摩擦係数Cfに変換する関数
    """
    Cf = n ** 2 / h ** (1/3) * g
    return Cf
    
def convert_Cf2manningn(Cf, h):
    """
    摩擦係数CfをManning's nに変換する関数
    """
    n = h ** (1/6) * np.sqrt(Cf / g)
    return n



if __name__ == "__main__":

    read_setfile("config.ini")

    parameters = [4000,5.0,10.0, 0.005, 0.005, 0.005, 0.005]

    start = time.time()
    (x, C, x_dep, deposit) = forward(parameters)
    print(time.time() - start, " sec.")
    export_result('sampling_point.txt', 'deposit.txt')
    plot_result(C, deposit)

#    spointfilename = 'sampling_point.txt'
#    flist = ["deposit_U6.txt","deposit_U8.txt", "deposit_U10.txt"]
#    llist = ["$U$ = 6.0 m/s","$U$ = 8.0 m/s", "$U$ = 10.0 m/s"]
#    glist = [0.5, 1.5, 2.5, 3.5, 5, 5.5]
#    loc = [10, 35, 74]
#    symbollist = ['--','-.','-']
#    plot_result_thick(spointfilename, flist, llist, symbollist)
#    plot_result_gdist('sampling_point.txt', flist, llist, glist, loc, symbollist)
