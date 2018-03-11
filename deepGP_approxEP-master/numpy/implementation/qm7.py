import math
import numpy as np 
import sys
import scipy.io
sys.path.append('../code/')
import AEPDGP_net
import matplotlib.pyplot as plt
import time
import pickle
from utils import *
np.random.seed(1234)

############################## LOAD THE DATASET #####################################
#####################################################################################
dataset = scipy.io.loadmat('dataset/qm7.mat')                                              ##
Columb_matrix = dataset['X']                                                       ##
Atomization = dataset['T'].T                                                       ##
R = dataset['R']                                                                   ##
Z = dataset['Z']                                                                   ##
split = dataset['P']                                                               ##
#####################################################################################
Columb_matrix_self=Columb_matrix_formation(Z,R)
Columb_matrix = Columb_matrix_self
print "Done Creating Columb_matrix_self"

############################## PARAMTER SETUP #######################################
#####################################################################################
params = {}                                                                        ##
params['nolayers' ]        = 2                                                     ##
params['n_hiddens']        = [2,5]                                                   ##
params['M']                = 15                                                    ##
params['no_epochs']        = 200                                                   ##
params['batch_size']       = 50                                                    ##
params['lrate']            = 0.01                                                  ## 
params['representation']   = 'CM_randn_sorted'                                            ## 
params['per_epoch_result'] = True                                                  ##
params['randn_sorted_rep'] = 10
params['randn_sorted_var'] = 1                                                     ##

var_param = {}                                                                     ##
var_param['name']  = 'n_hiddens'                                                   ##
var_param['value'] = [[2,5]]                                                        ##
params.pop(var_param['name'],None)                                                 ##
                                                                                   ##
dump = {}                                                                          ##
dump ['params']    = params     #for dumping purposes                              ##
dump ['var_param'] = var_param  #for dumping purposes                              ##
dump ['result']    = list()                                                        ##
#####################################################################################


for value in var_param['value']:
    
    #Init the local variables
    nolayers         = params.get('nolayers'        ,value)
    n_hiddens        = params.get('n_hiddens'       ,value)
    M                = params.get('M'               ,value)
    no_epochs        = params.get('no_epochs'       ,value)
    no_points_per_mb = params.get('batch_size'      ,value)
    lrate            = params.get('lrate'           ,value)
    representation   = params.get('representation'  ,value) 
    randn_sorted_var = params.get('randn_sorted_var',value)                                                          
    randn_sorted_rep = params.get('randn_sorted_rep',value)                                                          
    per_epoch_result = params.get('per_epoch_result',False)                                                          

    #Creating representations
    CM_eigen = -np.sort(-np.linalg.eigvals(Columb_matrix))
    CM_sorted = matrix_2d_sort_fn(Columb_matrix,random=False)
    CM_randn_sorted =matrix_2d_sort_fn(Columb_matrix,random=True)

#------------------------ Creating mutliple copies of Dataset in case of sorted random --------------------------------#    
#----------------------------------------------------------------------------------------------------------------------#    
    if representation == 'CM_randn_sorted':                                                                           
        for k in range(randn_sorted_rep-1):                                                                             
            split           = np.append(split,dataset['P']+CM_randn_sorted.shape[0],axis=1)                           
            CM_randn_sorted = np.append(CM_randn_sorted,matrix_2d_sort_fn(Columb_matrix,random=True),axis=0)          
            Atomization     = np.append(Atomization,dataset['T'].T,axis=0)                                            
    else:                                                                                                             
            Atomization = dataset['T'].T                                                               
            split = dataset['P']                                             
#----------------------------------------------------------------------------------------------------------------------

    CM_dict = {'CM_eigen':CM_eigen, 'CM_sorted':CM_sorted, 'CM_randn_sorted':CM_randn_sorted}


   
    #Convert CM into 2D matrix and n_pseudos
    CM = CM_dict[representation]
    CM = CM.reshape(CM.shape[0],-1)
    n_pseudos        = [M for _ in range(len(n_hiddens)+1)]
    
    #Create trainning and testing data
    X_train = CM[split[0:4].reshape(-1)]
    X_test = CM[split[4:5].reshape(-1)]
    y_train = Atomization[split[0:4].reshape(-1)]
    y_test =  Atomization[split[4:5].reshape(-1)]
    
    print "X_train.shape = ", X_train.shape
    print "y_train.shape = ", y_train.shape
    print "X_test.shape  = ", X_test.shape
    print "y_test.shape  = ", y_test.shape
    
    # We construct the network
    net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos,normalise_x=True)
    # train
    test_nll, test_rms,test_mae, energy = net.train(x_test=X_test,y_test=y_test,no_epochs=no_epochs,
                                   no_points_per_mb=no_points_per_mb,
                                   lrate=lrate,compute_test=True,compute_logZ=True)
    # We make predictions for the test set
    m, v = net.predict(X_test)
    
    # calculations
    rmse    = np.sqrt(np.mean((y_test - m)**2))
    mae     = np.mean(np.absolute(y_test - m))
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v)) - 0.5 * (y_test - m)**2 / (v))

    #dumping
    result = {'var_val':value, 'log_l':test_ll, 'rmse':rmse, 'mae':mae}
    if per_epoch_result:
        result.update({'rms_list':test_rms,'mae_list':test_mae,'log_ll_list':test_nll})
    dump['result'].append(result)

#Write into the logs/file using pickle   
pickle.dump(dump, open("logs/log_March7/test3."+var_param['name']+".p", "wb"))
