# -*- coding: cp932 -*-
import numpy as np
from scipy import interpolate as ip
import forward_model as fmodel
import time as tm
import scipy.optimize as opt
import matplotlib.pyplot as plt
from configparser import ConfigParser as scp

Nfeval = 1 #最適化計算ステップ数
deposit_o = [] #観察された堆積物の厚さ
spoints = [] #サンプリングポイントの座標
observation_x_file=[]
observation_deposit_file=[]
inversion_result_file=[]
inversion_x_file=[]
inversion_deposit_file=[]
initial_params = []
bound_values = []

def read_setfile(configfile):
    """
    設定ファイル（ファイル名configfile）を読み込んで
    初期値を設定する
    """
    global observation_x_file, observation_deposit_file, inversion_result_file, inversion_x_file, inversion_deposit_file, initial_params, bound_values
    parser = scp()
    parser.read(configfile)#設定ファイルの読み込み

    observation_x_file = parser.get("Import File Names", "observation_x_file")
    observation_deposit_file = parser.get("Import File Names", "observation_deposit_file")
    inversion_result_file = parser.get("Export File Names", "inversion_result_file")
    inversion_x_file = parser.get("Export File Names", "inversion_x_file")
    inversion_deposit_file = parser.get("Export File Names", "inversion_deposit_file")

    #初期値の読み込み
    Rw0 = parser.getfloat("Inversion Options", "Rw0")
    U0 = parser.getfloat("Inversion Options", "U0")
    h0 = parser.getfloat("Inversion Options", "h0")
    C0_text = parser.get("Inversion Options", "C0")
    #文字列を数値配列に変換（カンマ区切り）
    C0 = [float(x) for x in C0_text.split(',') if len(x) !=0]
    initial_params = [Rw0, U0, h0]
    initial_params.extend(C0)
    initial_params = np.array(initial_params)

    #求める値の範囲を読み込む
    Rwmax = parser.getfloat("Inversion Options", "Rwmax")
    Rwmin = parser.getfloat("Inversion Options", "Rwmin")
    Umax = parser.getfloat("Inversion Options", "Umax")
    Umin = parser.getfloat("Inversion Options", "Umin")
    hmax = parser.getfloat("Inversion Options", "hmax")
    hmin = parser.getfloat("Inversion Options", "hmin")
    Cmax_text = parser.get("Inversion Options", "Cmax")
    Cmax = [float(x) for x in Cmax_text.split(',') if len(x) !=0]
    Cmin_text = parser.get("Inversion Options", "Cmin")
    Cmin = [float(x) for x in Cmin_text.split(',') if len(x) !=0]
    bound_values_list = [(Rwmin, Rwmax), (Umin, Umax), (hmin, hmax)]
    for i in range(0, len(Cmax)):
        bound_values_list.append((Cmin[i],Cmax[i]))
    bound_values = tuple(bound_values_list)
    
    #フォワードモデルにも設定を反映させる
    fmodel.read_setfile(configfile)



def costfunction(optim_params):
    """
    計測値と計算値のズレを定量化
    """
    (x, C, x_dep, deposit_c) = fmodel.forward(optim_params)
    f = ip.interp1d(spoints, deposit_o, kind='cubic', bounds_error=False, fill_value=0.0)
    deposit_o_interp = f(x_dep)
    dep_norm = np.matrix(np.max(deposit_o_interp, axis = 1)).T
    residual = np.array((deposit_o_interp - deposit_c)/dep_norm)
    cost = np.sum((residual) ** 2)/len(x)
    return cost

def optimization(initial_params, bound_values, disp_init_cost=True, disp_result=True):
    """
    initial_paramsから出発して最適化計算を行う．
    上限値と下限値はbound_valuesで指定
    観測値に最も適合する結果を出すパラメーターを出力する
    """
    if disp_init_cost:
        #当初のcostfunctionを計算
        cost = costfunction(initial_params)
        print('Initial Cost Function = ', cost, '\n')
    
    #最適化計算を行う
    t0 = tm.clock()
    res = opt.minimize(costfunction, initial_params, method='L-BFGS-B',\
    #res = opt.minimize(imodel.costfunction, optim_params, method='CG',\
                   bounds=bound_values,callback=callbackF,\
                   options={'disp': True})
    print('Elapsed time for optimization: ', tm.clock() - t0, '\n')

    #計算結果を表示する
    if disp_result:
        print('Optimized parameters: ')
        print(res.x)
    
    return res

def readdata(spointfile, depositfile):
    """
    データの読み込みを行う
    """
    global deposit_o, spoints
    
    #データをセットする
    spoints = np.loadtxt(spointfile, delimiter=',')
    deposit_o = np.loadtxt(depositfile, delimiter=',')
    
    return (spoints, deposit_o)

def save_result(resultfile, spointfile, depositfile, res):
    """
    逆解析結果を保存する
    """
    #Forward Modelを計算
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x)

    #最適解を保存
    fmodel.export_result(spointfile, depositfile)
    np.savetxt(resultfile, res.x)
    

def callbackF(x):
    """
    途中経過の表示
    """
    global Nfeval
    print('{0: 3d}  {1: 3.0f}   {2: 3.2f}   {3: 3.2f}   {4: 3.3f}    {5: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], costfunction(x)))
    Nfeval +=1

def plot_result(res):
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x) #最適解を計算
    cnum = fmodel.cnum #粒度階の数を取得
    for i in range(cnum):
        plt.subplot(cnum,1,i+1) #粒度階の数だけグラフを準備
        plt.plot(spoints, deposit_o[i,:], marker = 'x', linestyle = 'None', label = "Observation")
        plt.plot(x_dep, deposit_c[i,:], marker = 'o', fillstyle='none', linestyle = 'None', label = "Calculation")
        plt.xlabel('Distance from the shoreline (m)')
        plt.ylabel('Deposit Thickness (m)')
        d = fmodel.Ds[i]*10**6
        gclassname='{0:.0f} $\mu$m'.format(d[0])
        plt.title(gclassname)
        plt.legend()

    plt.subplots_adjust(hspace=0.7)
    plt.show()

def inv_multistart():
    """
    マルチスタート法による逆解析を行う
    """
    
    #初期値と上限・下限を設定
    read_setfile('config.ini') 

    #観測データを読み込む
    (spoints, deposit_o) = readdata(observation_x_file, observation_deposit_file) #観測データを読み込む
    
    #初期値のリストを設定    
    initU = [2, 4, 6]
    initH = [2, 5, 8]
    initC = [[0.001, 0.001, 0.001,0.001], [0.004, 0.004, 0.004,0.004], [0.01, 0.01, 0.01, 0.01]]
    
    #逆解析を行うための初期値のリストを作る
    res = []
    initparams = []
    for i in range(len(initU)):
        for j in range(len(initH)):
            for k in range(len(initC)):
                init = [initial_params[0],initU[i],initH[j]]
                init.extend(initC[k])
                initparams.append(init)

    #複数の初期値で逆解析を行う
    for l in range(len(initparams)):
        res.append(optimization(initparams[l], bound_values))
    
    return res, initparams

if __name__=="__main__":
#    #初期値と上限・下限を設定
#    read_setfile('config_sendai.ini') 
#
#    #観測データを読み込む
#    (spoints, deposit_o) = readdata(observation_x_file, observation_deposit_file) #観測データを読み込む
#
#    #最適化計算を行う
#    res = optimization(initial_params, bound_values)
#
#    #結果を保存する
#    save_result(inversion_result_file, inversion_x_file, inversion_deposit_file, res)
#
#    #結果を表示
#    plot_result(res)

    res, initparams = inv_multistart()
    print(initparams)
    print(res)