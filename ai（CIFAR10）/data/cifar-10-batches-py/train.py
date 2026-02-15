import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net import Mymodel#导入自己写的网络模型
#训练代码，首先要得到数据集
from torchvision import datasets,transforms
writer=SummaryWriter(log_dir="logs")#logs是保存日志文件的文件夹
#数据预处理(为什么要进行预处理，又该怎么预处理，可以随便选吗)
transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),#随机水平翻转,数据增强
    transforms.RandomCrop(32,padding=4),#随机裁剪
    transforms.ToTensor(),#把图片转换为Tensor(Tensor是PyTorch中数据的基本结构，可以理解为多维数组)
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化处理,标准化，第一个参数和第二个参数分别是干什么的，又该怎么填数据
])
   
#transform是对数据进行预处理
#训练数据集
train_data_set=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
#测试数据集
test_data_set=datasets.CIFAR10(root="./data",train=False,transform=transform,download=True)
#数据集的大小
train_data_size=len(train_data_set) #训练集的大小
test_data_size=len(test_data_set)  #测试集的大小
#加载数据集
train_data_loader=DataLoader(train_data_set,batch_size=64,shuffle=True)#batch_size是每次训练的图片数量，shuffle是是否打乱数据
test_data_loader=DataLoader(test_data_set,batch_size=64,shuffle=True)#dataloader是用来加载数据的,要怎么才能引用他,打乱不打乱又有什么区别
##########################################定义网络########################################
myModel=Mymodel()#实例化模型
##########################################定义损失函数和优化器########################################
lossfn=torch.nn.CrossEntropyLoss()#交叉熵损失函数,这个函数是干什么的
optimizer=torch.optim.SGD(myModel.parameters(),lr=0.01)#SGD优化器,这个优化器是干什么的,这些参数又是干什么的
##########################################开始训练########################################
#判断是否使用GPU
use_gpu=torch.cuda.is_available()#判断是否有GPU
if(use_gpu):
    print("GPU可用")
    myModel=myModel.cuda()#把模型放到GPU上
#训练轮数
epochs=100#这个训练轮数该怎么选，选多少合适
for epoch in range(epochs):#训练epochs轮
    print("训练轮数{}/{}".format(epoch+1,epochs))
    #损失变量
    train_total_loss=0.0#总的训练损失
    test_total_loss=0.0#总的测试损失
    #准确个数
    train_total_acc=0.0#总的训练准确个数
    test_total_acc=0.0#总的测试准确个数
    #开始训练
    for data in train_data_loader:
        inputs,labels=data#把数据分成输入和标签(为什么要这样做，这样分有什么用)
        if(use_gpu):
            inputs=inputs.cuda()#把数据放到GPU上
            labels=labels.cuda()#把标签放到GPU上
       
        outputs=myModel(inputs)
        loss=lossfn(outputs,labels)#计算实际输出和真是数据间的差距
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播，计算新的梯度
        optimizer.step()#更新参数
        _,index=torch.max(outputs,dim=1)#取出每一行最大的值的索引
        acc=torch.sum(index==labels).item()#计算准确个数
        train_total_acc+=acc#把每次的准确个数加起来
        train_total_loss+=loss.item()#把每次的损失加起来
        
####################验证#####################
    with torch.no_grad():
     for data in test_data_loader:
        inputs,labels=data
        if(use_gpu):
            inputs=inputs.cuda()
            labels=labels.cuda()
        outputs=myModel(inputs)
        loss=lossfn(outputs,labels)
        _,index=torch.max(outputs,dim=1)
        acc=torch.sum(index==labels).item()
        test_total_acc+=acc
        test_total_loss+=loss.item()
    print("train loss:{},acc:{},test loss:{},acc:{}".format(train_total_loss,train_total_acc/train_data_size,test_total_loss,test_total_acc/test_data_size))   
    writer.add_scalar('loss/train',train_total_loss,epoch+1)
    writer.add_scalar('acc/train',train_total_acc/train_data_size,epoch+1)
    writer.add_scalar('loss/test',test_total_loss,epoch+1)
    writer.add_scalar('acc/test',test_total_acc/test_data_size,epoch+1)
    if((epoch+1)%50==0):#每50轮保存一次模型
     torch.save(myModel,"model/model_{}.pth".format(epoch+1))#保存模型
  