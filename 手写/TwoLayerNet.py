from softmax import softmax
from CrossEntropy import crossEntropy
from net import  SimpleNet
from relu import ReLU
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,output_size):
        self.layer1=SimpleNet(input_size,2*input_size)
        self.relu=ReLU()
        self.layer2=SimpleNet(2*input_size,output_size)

    def forward(self,x):
        layer1_output=self.layer1.forward(x)
        relu_output=self.relu.forward(layer1_output)
        layer2_output=self.layer2.forward(relu_output)
        return layer2_output
    

    def backward(self,dz,lr=0.1):
        dz,dw2,db2=self.layer2.backward(dz)
        dz=self.relu.backward(dz)
        dz,dw1,db1=self.layer1.backward(dz)
        self.layer2.w=self.layer2.w-dw2*lr
        self.layer2.b=self.layer2.b-db2*lr
        self.layer1.w=self.layer1.w-dw1*lr
        self.layer1.b=self.layer1.b-db1*lr
        return dz
    