import numpy as np

class SimpleNet:
    def __init__(self,input_size,output_size):

        
        self.w=np.random.rand(input_size,output_size)* 0.01

        self.b=np.zeros(1,output_size)

    def forward(self,x):
        self.input=x
        return np.dot(x,self.w)+self.b

    def backward(self,dz):
        m=self.input.shape[0]
        dw=np.dot(self.input.T,dz)/m
        db=np.sum(dz,axis=0,keepdims=True)/m
        dx=np.dot(dz,self.w.T)
        return dx,dw,db