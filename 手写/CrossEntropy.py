import numpy as np
from softmax import softmax

def crossEntropy(x,y):
    y_hat=softmax(x)
    loss=-np.sum(y * np.log(y_hat + 1e-10),axis=-1,keepdims=True)
    return loss