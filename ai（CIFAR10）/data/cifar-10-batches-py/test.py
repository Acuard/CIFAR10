import torch
from torchvision import transforms
import os
import cv2     
use_gpu=torch.cuda.is_available()#判断是否有GPU
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
model=torch.load("./model/model_100.pth",map_location=torch.device('cuda'if use_gpu else 'cpu')) #加载模型
model.eval()
#数据预处理
transform=transforms.Compose([
    transforms.ToTensor(),#把图片转换为Tensor(Tensor是PyTorch中数据的基本结构，可以理解为多维数组)
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化处理,标准化，第一个参数和第二个参数分别是干什么的，又该怎么填数据
])
#指定测试用文件夹
folder_path='./testimages'
files=os.listdir(folder_path)#列出文件夹下所有的目录与文件
images_files=[os.path.join(folder_path,f) for f in files]#把文件夹路径和文件名拼接起来,得到所有文件的地址

######测试######
for img in images_files:
    image=cv2.imread(img)#读取图片
    cv2.imshow('image',image)#显示图片
    image=cv2.resize(image,(32,32))#把图片大小统一调整为32*32
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB,)#把图片从BGR格式转换为RGB格式(为什么要转换,这里不太懂
    image=transform(image)#对图片进行预处理
    cv2.waitKey(0)#等待按键,这个具体该怎么用
    image=torch.reshape(image,(1,3,32,32))
    image=image.to('cuda'if use_gpu else 'cpu')
    output=model(image)#把图片输入到网络中,得到输出
    value,index=torch.max(output,1)#得到最大值和索引
    pre_val=classes[index]#得到预测的类别
    print("预测概率:{},预测下标：{},预测结果：{}".format(value.item(),index.item(),pre_val))

   
    







