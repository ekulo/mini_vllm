import numpy as np

def softmax(x):
    x_max=np.max(x,axis=-1,keepdims=True)
    exp_x=np.exp(x-x_max)
    return exp_x / np.sum(exp_x,axis=-1,keepdims=True)