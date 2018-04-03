import math
import numpy as np 
import sys
import scipy.io
sys.path.append('../code/')
import GP
import noisy_GP
import time
import pickle
from utils import *

############################## LOAD THE DATASET #####################################
#####################################################################################
dataset = scipy.io.loadmat('dataset/qm7.mat')                                              ##
Columb_matrix = dataset['X']                                                       ##
Atomization = dataset['T'].T                                                       ##
R = dataset['R']                                                                   ##
Z = dataset['Z']                                                                   ##
split = dataset['P']                                                               ##
#####################################################################################
#Columb_matrix_self=Columb_matrix_formation(Z,R)
#Columb_matrix = Columb_matrix_self
#print "Done Creating Columb_matrix_self"

############################## PARAMTER SETUP #######################################
#####################################################################################
params = {}                                                                        ##
params['num_epochs']       = 5000                                                  ## 
params['representation']   = 'CM_eigen'                                            ## 
params['lrate']            = 1                                                  ##
params['sigma']            = 0                                                  ## 

var_param = {}                                                                     ##
var_param['name']  = 'lrate'                                                   ##
var_param['value'] = [1e-3]                                                        ##
params.pop(var_param['name'],None)                                                 ##
                                                                                   ##
dump = {}                                                                          ##
dump ['params']    = params     #for dumping purposes                              ##
dump ['var_param'] = var_param  #for dumping purposes                              ##
dump ['result']    = list()                                                        ##
#####################################################################################


for value in var_param['value']:
    
    #Init the local variables
    num_epochs            = params.get('num_epochs'           ,value)
    lrate                 = params.get('lrate'                ,value)
    sigma                 = params.get('sigma'           ,value)
    representation        = params.get('representation'       ,value) 

    #Creating representations
    CM_eigen = -np.sort(-np.linalg.eigvals(Columb_matrix))
    CM_sorted = matrix_2d_sort_fn(Columb_matrix,random=False)
    CM_randn_sorted =matrix_2d_sort_fn(Columb_matrix,random=True)

#------------------------ Creating mutliple copies of Dataset in case of sorted random --------------------------------#    
#----------------------------------------------------------------------------------------------------------------------#    
    if representation == 'CM_randn_sorted':                                                                           
        randn_sorted_rep = 10
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
    
	 
    #Create trainning and testing data
    X_train = CM[split[0:2].reshape(-1)]
    X_test = CM[split[4:5].reshape(-1)]
    y_train = Atomization[split[0:2].reshape(-1)]
    y_test =  Atomization[split[4:5].reshape(-1)]
    
   # print "X_train.shape = ", X_train.shape
   # print "y_train.shape = ", y_train.shape
   # print "X_test.shape  = ", X_test.shape
   # print "y_test.shape  = ", y_test.shape
    
    # We construct the network
    net = noisy_GP.GP(X_train = X_train, y_train = y_train)
    # train
    net.train(X_test,y_test,num_epochs=num_epochs,lrate=lrate,sigma=sigma,compute_test=True)

    # We make predictions for the test set
    m_test = net.predict(X_test)
    m_train = net.predict(X_train)

    # calculations
    rmse_test    = np.sqrt(np.mean((y_test - m_test)**2))
    mae_test     = np.mean(np.absolute(y_test - m_test))
    rmse_train   = np.sqrt(np.mean((y_train - m_train)**2))
    mae_train    = np.mean(np.absolute(y_train - m_train))
    #dumping
    result = {'var_val':value, 'rmse_test':rmse_test, 'mae_test':mae_test, 'rmse_train':rmse_train, 'mae_train':mae_train}
    print var_param['name']+':',value, result
    dump['result'].append(result)

#Write into the logs/file using pickle   
pickle.dump(dump, open("logs/log_March7/testK1."+var_param['name']+".p", "wb"))


