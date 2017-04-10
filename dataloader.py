import numpy as np
import pickle

def data_loader():
    train_x = []
    val_x = []
    train_y = []
    val_y = []

    with open('train_x.pkl', 'rb') as fid:
        train_x = pickle.load(fid)

    with open('val_x.pkl', 'rb') as fid:
        val_x = pickle.load(fid)

    with open('train_y.pkl', 'rb') as fid:
        train_y = pickle.load(fid)

    with open('val_y.pkl', 'rb') as fid:
        val_y = pickle.load(fid)
    return train_x,train_y,val_x,val_y



'''
#return value in each time windows
def generate_batch(dim1,dim2,batch_size,inputs,targets,num):
    x = np.empty((batch_size, dim1, 1), dtype = int)
    y = np.empty((batch_size, dim2, 1), dtype = int)

    for i in range(batch_size):
        x[i, :, 0] = inputs[num]
        y[i, :, 0] = targets[num]
        num += 1
    return x, y, num
'''
'''
#return the binary value in each time windows
def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res


def generate_batch(dim1,dim2,batch_size,inputs,targets,num,num_bits):
    x = np.empty((num_bits, batch_size, dim1), dtype = int)
    y = np.empty((num_bits, batch_size, dim2), dtype = int)

    for i in range(batch_size):
        for j in range(dim1):
            x[:, i, j] = as_bytes(inputs[num][j],num_bits)
            y[:, i, j] = as_bytes(targets[num][j],num_bits)
        num += 1
    return x, y, num
'''
def generate_batch(dim1,dim2,batch_size,inputs,targets,num,numbits):
    x = np.empty((batch_size, dim1, 1), dtype = float)
    y = np.empty((batch_size, dim2, 1), dtype = float)

    for i in range(batch_size):
        x[i, :, 0] = 1.0 / (1 + np.exp(-inputs[num]))
        y[i, :, 0] = 1.0 / (1 + np.exp(-targets[num]))
        num += 1
    return x, y, num

