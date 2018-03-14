import math
import numpy as np
import sys as Sys

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")



class Deep_Neural_Network:

# ------------------------------------------------------
# Deep Neural Network Constructor
# ----------------------------------------------------------
    def __init__(self,X_train,y_train,n_hiddens=[2],
                normalise_x=True,normalise_y=True):

        self.X_train     = X_train
        self.y_train     = y_train
        self.num_train   = self.X_train.shape[1]
        self.num_nodes   = [self.X_train.shape[0]] + n_hiddens + [self.y_train.shape[0]]
        self.num_layers  = len(self.num_nodes) - 1
        self.normalise_x = normalise_x
        self.normalise_y = normalise_y
        self.y_mean      = np.mean(self.y_train,axis=1)
        self.y_std       = np.std(self.y_train,axis=1)
    
        if normalise_x:
            self.X_train = (self.X_train - np.mean(self.X_train,axis=1,keepdims=True))/np.std(self.X_train,axis=1,keepdims=True)
        if normalise_y:
            self.y_train = (self.y_train - np.mean(self.y_train,axis=1))/np.std(self.y_train,axis=1) 
        self.params = {}
    
        for layer in range(self.num_layers):
            self.params['W'+str(layer)] = np.random.randn(self.num_nodes[layer+1],self.num_nodes[layer])*(1./np.sqrt(self.num_nodes[layer]))
            self.params['b'+str(layer)] = np.zeros(self.num_nodes[layer+1])
    


# ----------------------------------------------------------
# Forward Pass
# ----------------------------------------------------------
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def sigmoid_backward(self,x):
	return sigmoid(x)*(1-sigmoid(x))

    def relu(self,x):
	return x*(x>0)

    def relu_backward(self,x):
	return 1*(x>0)

# ----------------------------------------------------------
    def forward_step(self,x,W,b):
        z = np.dot(W,x) + b.reshape(-1,1)
        #using sigmoid in output layer
        a = np.tanh(z)
        cache = (a,x)
        return a,cache


# ----------------------------------------------------------
    def forward_propogation(self,X):
        Cache = list()

        for layer in range(self.num_layers):
            output,cache = self.forward_step(X,self.params['W'+str(layer)],self.params['b'+str(layer)])
            X = output
            Cache.append(cache)

        return output,Cache


# ----------------------------------------------------------
# Loss and its derivate computation
# ----------------------------------------------------------
    def compute_loss(self,Y,Ypred):
        #Currently only supports L2 loss
        m = self.num_train
        loss = (0.5/m)*np.sum((Y-Ypred)**2)
        dYpred = (1.0/m)*(Ypred-Y)
        return loss,dYpred



# ----------------------------------------------------------
# Backward Pass
# ----------------------------------------------------------
    def backward_step(self,da,cache,W,b):
        (a,x)=cache
        dz   = (1-a**2)*da
        dx   = np.dot(W.T,dz)
        dW   = np.dot(dz,x.T)
        db   = np.sum(dz,axis=1)
        return dx,dW,db

# ----------------------------------------------------------
    def backward_propogation(self,dy,Cache):
        Grads = {}
        
        for layer in reversed(range(self.num_layers)):
            dx,dW,db = self.backward_step(dy,Cache[layer],self.params['W'+str(layer)],self.params['b'+str(layer)])
	    dy = dx
            Grads['W'+str(layer)] = dW
            Grads['b'+str(layer)] = db

        return Grads

# ----------------------------------------------------------
# Param Update
# ----------------------------------------------------------

    def sgd_update(self,params,grads,lrate,reg):
        for key in grads:
            params[key]   = params[key]-lrate*grads[key]

        return params

    def momentum_update(self,params,grads,velocity,lrate,momentum,reg):
        for key in grads:
            grads[key] = grads[key] + reg*params[key]
            velocity[key] = velocity.get(key,np.zeros_like(params[key]))
            velocity[key] = momentum*velocity[key] - lrate*grads[key]
            params[key]   = params[key]+velocity[key]

        return params,velocity


# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
    def train(self,X_test,y_test,no_epochs=100,batch_size = 50,
			  lrate = 0.01, momentum=0.9,reg=0, compute_test = True):

        velocity = {}
        num_batches = int(math.ceil(self.num_train/batch_size))
        test_loss  = list()
        test_rms   = list()
        test_mae   = list()

	for epoch in range(no_epochs):
            #printProgress(0, num_batches, prefix = 'Epoch %d:' % epoch, suffix = 'Complete', barLength = 20)
            for batch in range(num_batches):
                #batch formation
                batch_mask = np.random.choice(self.num_train, batch_size)
                X_batch = self.X_train[:,batch_mask]
                Y_batch = self.y_train[:,batch_mask]
                #backprop
                Ypred, cache = self.forward_propogation(X_batch)
                loss,dYpred = self.compute_loss(Y_batch,Ypred)
                grads = self.backward_propogation(dYpred,cache)
                #self.params=self.sgd_update(self.params,grads,lrate,reg)
                self.params,velocity=self.momentum_update(self.params,grads,velocity,lrate,momentum,reg)
                #printProgress(batch,num_batches, prefix = 'Epoch %d:' % (epoch/10), suffix = 'Complete', barLength = 20)
		    
            #Compute test after each epoch	
            if compute_test and epoch%100==0:
                ypred_test   = self.predict(X_test)
                test_lossi   = 0.5*np.mean((y_test - ypred_test)**2)
                test_rmsi    = np.sqrt(np.mean((y_test - ypred_test)**2))
                test_maei    = np.mean(np.absolute(y_test - ypred_test)) 
                print "epoch: %.5f, test loss: %.5f, test rms: %.5f, test mae: %.5f" % ((epoch/100),test_lossi,test_rmsi,test_maei)
                test_loss.append(test_lossi)
                test_rms.append(test_rmsi)
                test_mae.append(test_maei)
		
        return test_loss,test_rms,test_mae

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------
    def predict(self,X):
        if self.normalise_x:
            X = (X - np.mean(X,axis=1,keepdims=True))/np.std(X,axis=1,keepdims=True)
        ypred,_ = self.forward_propogation(X)
        if self.normalise_y:
            ypred = ypred*self.y_std + self.y_mean
        return ypred


# ----------------------------------------------------------
# Grad Check
# ----------------------------------------------------------

    def grad_check(self):
        seq_length, num_data = 2,5
        epsilon = 1e-8
        data = np.random.randn(seq_length,num_data,self.num_nodes[0])
        target = np.random.randn(seq_length,num_data,self.num_nodes[-1])

        Ypred,cache = self.forward_propogation(data)
        loss,dYpred = self.compute_loss(target,Ypred)
        grads = self.backward_propogation(dYpred,cache)

        for key in self.params:
            self.params[key]+=epsilon
            Ypred,_ = self.forward_propogation(data)
            loss1,_ = self.compute_loss(target,Ypred)
            self.params[key]-=2*epsilon
            Ypred,_ = self.forward_propogation(data)
            loss2,_ = self.compute_loss(target,Ypred)
            first_difference = (loss1 - loss2)/(2*epsilon)
            print ("First_difference= ",first_difference,"   grad[",key,"] =",grads[key])
            self.params[key]+=epsilon





