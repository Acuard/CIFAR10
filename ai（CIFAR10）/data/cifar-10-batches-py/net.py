import torch
from torch import nn
class Mymodel(nn.Module):
    ##########################神经网络结构定义##########################
    def __init__(self)->None: #重写一个INIT函数
        super().__init__()#继承父类的init函数
        self.conv1= nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)#卷积层(第一层)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)#池化层(第一层)
        self.conv2= nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)#卷积层(第二层)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)#池化层(第二层)
        self.conv3= nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)#卷积层(第三层)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)#池化层(第三层)
        self.flatten=nn.Flatten()#展平层
        #self._initialize_linear_input_size()#动态计算展评后的特征图大小
        self.linear1=nn.Linear(1024,64)#全连接层(第一层)
        self.linear2=nn.Linear(64,10)#全连接层(第二层)
        self.softmax=nn.Softmax(dim=1)#softmax函数,把输出值变成0到1之间的概率值
    ##########################前向传播函数#############################（让数据按照上面定义的层一层一层的传递，最终得到输出）
    def forward(self, x):#重写一个forward函数
       x=self.conv1(x)#第一层卷积
       x=self.maxpool1(x)#第一层池化
       x=self.conv2(x)#第二层卷积
       x=self.maxpool2(x)#第二层池化
       x=self.conv3(x)#第三层卷积
       x=self.maxpool3(x)#第三层池化
       x=self.flatten(x)#展平
       x=self.linear1(x)#第一层全连接
       x=self.linear2(x)#第二层全连接
       x=self.softmax(x)#softmax函数
       return x;

