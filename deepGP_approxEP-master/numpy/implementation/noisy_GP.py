import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
class GP:

    def compute_covariance(self,X1,X2):
        l2 = self.params['lmbda2']
        s2 = self.params['sigma2']
#	di = np.sum((X1[:,np.newaxis,:] - X2)**2,axis=2)
	di = np.dot((X1[:,np.newaxis,:] - X2)**2,l2).reshape(X1.shape[0],X2.shape[0])
	K = s2*np.exp(-di/2)  
        return K
    
    def compute_distance(self,X1,X2):
        Dist = (X1[:,np.newaxis,:]-X2)**2
	Dist = Dist.transpose(2,0,1)
        return Dist
    
    def compute_mean(self,X):
        B = self.params['Beta']
        X = X	#can be replaced with phiX
        
        u = np.dot(X,B)
        
        return u
    	
    
    def __init__(self,X_train,y_train,normalise_x=True,normalise_y=True):
    	(N,D) = X_train.shape
    
        assert y_train.shape[1] == 1
        assert y_train.shape[0] == N
    
    	self.num_train = X_train.shape[0]
    	self.D         = X_train.shape[1]
    	self.X_train   = X_train
    	self.y_train   = y_train
        self.normalise_x = normalise_x
        self.normalise_y = normalise_y
        self.y_mean      = np.mean(self.y_train,axis=0)
        self.y_std       = np.std(self.y_train,axis=0)
        self.sigman2     = 0

        if normalise_x:
            self.X_train = (self.X_train - np.mean(self.X_train,axis=0,keepdims=True))/np.std(self.X_train,axis=0,keepdims=True)
        print normalise_x,self.y_mean,self.y_std,y_train.shape
        if normalise_y:
            self.y_train = (self.y_train - np.mean(self.y_train,axis=0))/np.std(self.y_train,axis=0) 
    
    	self.params = {}
        self.params['Beta']    = 0.01*np.random.randn(D,1)
    	self.params['sigma2']  = 1
    	self.params['lmbda2']  = np.random.randn(D,1)*0.1+0.5
    
# ----------------------------------------------------------
# Param Update
# ----------------------------------------------------------

    def sgd_update(self,param,grad,lrate):
        param   = param-lrate*grad

        return param

    def momentum_update(self,param,grad,velocity,lrate,momentum,key='x'):
        velocity[key] = velocity.get(key,np.zeros_like(param))
        velocity[key] = momentum*velocity[key] - lrate*grad
        param   = param+velocity[key]

        return param,velocity


    def train(self,X_test=None,y_test=None,lrate = 0.1,sigma = 0,num_epochs=100,compute_test=True):
    	X = self.X_train
    	y = self.y_train
        self.sigman2 = sigma
        velocity = {}
    	for epoch in range(num_epochs):
            s2 = self.params['sigma2']
            l2 = self.params['lmbda2']
            N  = self.num_train
            D  = self.D
            
            u     = self.compute_mean(X)
            K     = self.compute_covariance(X,X)
            R     = K/s2
            K     = K + self.sigman2*np.eye(K.shape[0])
            Dist  = self.compute_distance(X,X) 
            K_inv = inv(K)
            
            assert u.shape     == (N,1)
            assert K.shape     == (N,N)
            assert Dist.shape  == (D,N,N)
            assert K_inv.shape == (N,N)
            assert R.shape     == (N,N)
            
            B      = np.dot(inv(multi_dot([X.T,K_inv,X])),multi_dot([X.T,K_inv,y]))
            ds2    = (1./2)*(np.trace(np.dot(K_inv,R)) - multi_dot([(y-u).T,K_inv,R,K_inv,y-u]))
            dl2 = np.zeros_like(l2)

	    for i in range(D):	
            	dl2[i]    = (1./4)*(-np.trace(np.dot(K_inv,(R*s2)*Dist[i])) + multi_dot([(y-u).T,K_inv,(R*s2)*Dist[i],K_inv,y-u]))
                
            assert B.shape   == (D,1)
            assert dl2.size  == l2.size
            assert ds2.size  == 1
           
            
            l2             = self.sgd_update(l2,dl2,lrate)
            s2             = self.sgd_update(s2,ds2,lrate)
            #l2,velocity     = self.momentum_update(param=l2,grad=dl2,velocity=velocity,lrate=lrate,momentum=0.9,key='l2')
            l2              = np.absolute(l2)
            #s2,velocity     = self.momentum_update(param=s2,grad=ds2,velocity=velocity,lrate=lrate,momentum=0.9,key='s2')
            s2              = np.absolute(s2)
            
            print 'epoch = ', epoch , 'Beta   = ' , B[0] , 'sigma  = ' , s2 , 'dlmda2 = ' , dl2.mean() ,'lmbda2 = ' , l2.mean()
            
            self.params['Beta']   = B
            self.params['sigma2'] = s2
            self.params['lmbda2'] = l2
            if compute_test and epoch%1==0:
                ypred_test   = self.predict(X_test)
                test_rmsi    = np.sqrt(np.mean((y_test - ypred_test)**2))
                test_maei    = np.mean(np.absolute(y_test - ypred_test)) 
                print "epoch: %.5f, test rms: %.5f, test mae: %.5f" % ((epoch),test_rmsi,test_maei)
    
    	 
    
    def predict(self,X_test):

        if self.normalise_x:
            X_test = (X_test - np.mean(X_test,axis=0,keepdims=True))/np.std(X_test,axis=0,keepdims=True)
    
    	X_train = self.X_train
    	y_train = self.y_train
    	Nx  = self.num_train
        D  = self.D
        Ny = X_test.shape[0]
    
        assert X_test.shape[1] == D
    
    	ux      = self.compute_mean(X_train) 
    	uy      = self.compute_mean(X_test)
    	Kxx     = self.compute_covariance(X_train,X_train)
        Kxx     = Kxx + self.sigman2*np.eye(Kxx.shape[0])
    	Kxy     = self.compute_covariance(X_train,X_test)
    	Kyy     = self.compute_covariance(X_test ,X_test)
    	Kxx_inv	= inv(Kxx)		
    
        assert ux.shape      == (Nx,1)
        assert Kxx.shape     == (Nx,Nx)
        assert uy.shape      == (Ny,1)
        assert Kyy.shape     == (Ny,Ny)
        assert Kxy.shape     == (Nx,Ny)
        assert Kxx_inv.shape == (Nx,Nx)
    
    
        uf     = uy + np.dot(np.dot(Kxy.T,Kxx_inv),(y_train - ux))
    	Kf     = Kyy - multi_dot([Kxy.T,Kxx_inv,Kxy])

        assert uf.shape     == (Ny,1)
        assert Kf.shape     == (Ny,Ny)

        if self.normalise_y:
            uf = uf*self.y_std + self.y_mean
    
    	return uf
    

