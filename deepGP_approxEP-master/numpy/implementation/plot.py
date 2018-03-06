import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import glob

for filename in glob.iglob('logs/test.*'):

    data=pickle.load(open(filename, "rb"))
    filename = filename.replace('.p','.png')
    filename = filename.replace('logs/test.','logs/plots/plot.')
    params = data['params']
    var_param = data['var_param']
    print var_param
    assert len(var_param['value']) == len(data['result']) , "Var_Param and len(result) doesn't match"
    plt.figure()
    plt.suptitle('Study of variation of '+var_param['name'])
    mae = list();
    rmse = list()
    log_l = list()
    for k in range(len(data['result'])):
        k_max = len(data['result'])-1
        k_mid = k_max/2
        var    = data['result'][k]['var_val']
        mae.append( data['result'][k]['mae'])
        rmse.append(data['result'][k]['rmse'])
        log_l.append(-data['result'][k]['log_l'])
        mae_list   = data['result'][k]['mae_list']
        rms_list   = data['result'][k]['rms_list']
        log_ll_list  = data['result'][k]['log_ll_list']
        if k==0 or k ==k_max or k==k_mid or var_param['name']=='lrate':
            plt.subplot(2,2,1)
            plt.plot(mae_list,label = var_param['name']+"="+str(var))
            plt.ylabel('mae')
            plt.xlabel('epochs')
            plt.legend(loc='upper right')
            plt.ylim(ymax=30)
            plt.ylim(ymin=10)
        
            plt.subplot(2,2,2)
            plt.plot(rms_list,label = var_param['name']+"="+str(var))
            plt.ylabel('rmse')
            plt.xlabel('epochs')
            plt.legend(loc='upper right')
            plt.ylim(ymax=40)
            plt.ylim(ymin=10)
        
            plt.subplot(2,2,3)
            plt.plot(log_ll_list,label = var_param['name']+"="+str(var))
            plt.ylabel('log_ll')
            plt.xlabel('epochs')
            plt.legend(loc='upper right')
            plt.ylim(ymax=6)
            plt.ylim(ymin=4)
        
    print var_param['name']+" = ", var,"   mae =",mae
    plt.savefig(filename.replace('.png','_epoch.png'))
    plt.figure()
    plt.suptitle('Study of variation of '+var_param['name'])
    
    plt.subplot(2,2,1)
    x = range(len(var_param['value']))
    plt.xticks(x,var_param['value'])
    plt.plot(x,mae,'ro')
    plt.ylabel('mae')
    plt.xlabel('epochs')
    plt.ylim(ymax=30)
    plt.ylim(ymin=10)
    
    plt.subplot(2,2,2)
    x = range(len(var_param['value']))
    plt.xticks(x,var_param['value'])
    plt.plot(rmse,'ro')
    plt.ylabel('rmse')
    plt.xlabel('epochs')
    plt.ylim(ymax=40)
    plt.ylim(ymin=10)
    
    plt.subplot(2,2,3)
    x = range(len(var_param['value']))
    plt.xticks(x,var_param['value'])
    plt.plot(log_l,'ro')
    plt.ylabel('log_ll')
    plt.xlabel('epochs')
    plt.ylim(ymax=6)
    plt.ylim(ymin=4)
    
    plt.savefig(filename)
