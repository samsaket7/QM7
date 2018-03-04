import numpy as np

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

