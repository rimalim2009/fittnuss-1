import numpy as np
from scipy import interpolate as ip
from fittnuss import forward_model as fmodel
import time as tm
import scipy.optimize as opt
import matplotlib.pyplot as plt
from configparser import ConfigParser as scp

Nfeval = 1 #number of epochs in optimization calculation
deposit_o = [] #Observed thickness of a tsunami deposit
spoints = [] #Location of sampling points
observation_x_file=[]
observation_deposit_file=[]
inversion_result_file=[]
inversion_x_file=[]
inversion_deposit_file=[]
initial_params = []
bound_values = []

def read_setfile(configfile):
    """
    read setting file (config.ini) and set parameters to the inverse model
    """
    global observation_x_file, observation_deposit_file, inversion_result_file, inversion_x_file, inversion_deposit_file, initial_params, bound_values
    parser = scp()
    parser.read(configfile)#read a setting file

    #set file names
    observation_x_file = parser.get("Import File Names", "observation_x_file")
    observation_deposit_file = parser.get("Import File Names", "observation_deposit_file")
    inversion_result_file = parser.get("Export File Names", "inversion_result_file")
    inversion_x_file = parser.get("Export File Names", "inversion_x_file")
    inversion_deposit_file = parser.get("Export File Names", "inversion_deposit_file")

    #Set starting values
    Rw0 = parser.getfloat("Inversion Options", "Rw0")
    U0 = parser.getfloat("Inversion Options", "U0")
    h0 = parser.getfloat("Inversion Options", "h0")
    C0_text = parser.get("Inversion Options", "C0")

    #Convert strings (comma separated) to float
    C0 = [float(x) for x in C0_text.split(',') if len(x) !=0]
    initial_params = [Rw0, U0, h0]
    initial_params.extend(C0)
    initial_params = np.array(initial_params)

    #Set ranges of possible solutions
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
    
    #Set the variales in the forward model
    fmodel.read_setfile(configfile)



def costfunction(optim_params):
    """
    Calculate objective function that quantifies the difference between
    field observations and results of the forward model calculation

    Fist, this function runs the forward model using a given set of parameters.
    Then, the mean square error of results was calculated.
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
    Calculate parameter set that minimize the objective function (cost function)
    Optimization is started at the starting values (initial_params). The 
    L-BFGS-B method was used for optimization with parametric boundaries defined
    by bound_values

    """
    if disp_init_cost:
        #show the value of objective function at the starting values
        cost = costfunction(initial_params)
        print('Initial Cost Function = ', cost, '\n')
    
    #Start optimization by L-BFGS-B method
    t0 = tm.clock()
    res = opt.minimize(costfunction, initial_params, method='L-BFGS-B',\
    #res = opt.minimize(imodel.costfunction, optim_params, method='CG',\
                   bounds=bound_values,callback=callbackF,\
                   options={'disp': True})
    print('Elapsed time for optimization: ', tm.clock() - t0, '\n')

    #Display result of optimization
    if disp_result:
        print('Optimized parameters: ')
        print(res.x)
    
    return res

def readdata(spointfile, depositfile):
    """
    Read measurement dataset
    """
    global deposit_o, spoints
    
    #Set variables from data files
    spoints = np.loadtxt(spointfile, delimiter=',')
    deposit_o = np.loadtxt(depositfile, delimiter=',')
    
    return (spoints, deposit_o)

def save_result(resultfile, spointfile, depositfile, res):
    """
    Save the inversion results
    """
    #Calculate the forward model using the inversion result
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x)

    #Save the best result
    fmodel.export_result(spointfile, depositfile)
    np.savetxt(resultfile, res.x)
    

def callbackF(x):
    """
    A function to display progress of optimization
    """
    global Nfeval
    print('{0: 3d}  {1: 3.0f}   {2: 3.2f}   {3: 3.2f}   {4: 3.3f}    {5: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], costfunction(x)))
    Nfeval +=1

def plot_result(res):
    (x, C, x_dep, deposit_c) = fmodel.forward(res.x) #Calculate optimized result
    cnum = fmodel.cnum #Number of grain-size classes
    for i in range(cnum):
        plt.subplot(cnum,1,i+1) #Prepare plot of each grain-size class
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
    Perform inversion using the multi-start method
    """
    
    #Set initial conditions
    read_setfile('config.ini') 

    #Read measurement data set
    (spoints, deposit_o) = readdata(observation_x_file,\
                                    observation_deposit_file)
    
    #Set a list of starting values
    initU = [2, 4, 6]
    initH = [2, 5, 8]
    initC = [[0.001, 0.001, 0.001,0.001], [0.004, 0.004, 0.004,0.004], [0.01, 0.01, 0.01, 0.01]]
    
    #Make a list of sets of starting values
    res = []
    initparams = []
    for i in range(len(initU)):
        for j in range(len(initH)):
            for k in range(len(initC)):
                init = [initial_params[0],initU[i],initH[j]]
                init.extend(initC[k])
                initparams.append(init)

    #Perform inversion using multiple starting values
    for l in range(len(initparams)):
        res.append(optimization(initparams[l], bound_values))
    
    return res, initparams

if __name__=="__main__":
    #Set initial conditions of inverse model
    read_setfile('config_sendai.ini') 

    #Read measurement data set
    (spoints, deposit_o) = readdata(observation_x_file,\
                                    observation_deposit_file)

    #Conduct optimization
    res = optimization(initial_params, bound_values)

    #Save the reuslt of inversion
    save_result(inversion_result_file, inversion_x_file, inversion_deposit_file, res)

    #Plot results
    plot_result(res)

#    res, initparams = inv_multistart()
#    print(initparams)
#    print(res)
