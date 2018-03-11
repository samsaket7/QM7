import numpy as np
import sys

def matrix_2d_sort_fn(matrix,random=False):
    assert matrix.ndim==3,"Only 3 dims supported"

    temp_matrix = matrix
    
    #Row sorting
    randomness = np.zeros((matrix.shape[0],matrix.shape[1]))
    if random:
        randomness = np.random.randn(*randomness.shape)
    arg_row_sum=np.argsort(-(np.square(temp_matrix).sum(axis=2)+randomness))
    Index = np.zeros_like(arg_row_sum)
    Index[:] = (np.arange(Index.shape[0]).reshape(Index.shape[0],1))
    temp_matrix=temp_matrix[Index,arg_row_sum,:]
    
    #Column sorting
    randomness = np.zeros((matrix.shape[0],matrix.shape[2]))
    if random:
        randomness = np.random.randn(*randomness.shape)
    arg_col_sum=np.argsort(-(np.square(temp_matrix).sum(axis=1)+randomness))
    Index = np.zeros_like(arg_col_sum)
    Index[:] = (np.arange(Index.shape[0]).reshape(Index.shape[0],1))
    temp_matrix=temp_matrix[Index,:,arg_col_sum].transpose(0,2,1)

    return temp_matrix

def Columb_matrix_formation(Z,R):
    epsilon = 10e-30
    num_atoms = Z.shape[1]
    num_molecules = Z.shape[0]
    C = np.zeros((num_molecules,num_atoms,num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            Rd = np.sqrt(np.sum(((R[:,i,:] - R[:,j,:])**2),axis=1))
            Rd[Rd==0]=epsilon
            if(i==j):
                C[:,i,j] = (Z[:,i]**2.4)/2
            else:
                C[:,i,j] = (Z[:,i]*Z[:,j])/Rd


    return C
