import numpy as np

class ReLU:
    def forward(self,x):
        self.x=x
        return np.maximum(0,x)
    def backward(self,delta):
        return delta*(self.x>0)