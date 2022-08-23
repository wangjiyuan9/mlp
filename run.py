import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import MLP
import os
import time
parser=argparse.ArgumentParser()
parser.add_argument("--num_epochs",default=30,type=int)
parser.add_argument("--num_batch_size",default=32,type=int)
args=parser.parse_args()

if not os.path.exists('.\data'):
    os.makedirs('.\data')
image_h=28
image_w=28
image_p=1

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using {}'.format(device))

model=MLP.my_MLP().to(device)
loss_func=torch.nn.CrossEntropyLoss()#loss函数使用交叉熵，我们希望一种分布（可以是概率向量）与另一种分布接近， 而交叉熵和KL散度为我们提供了一种自然的方法测量两个分布之间的差距，这个差距就可以被当作损失函数。
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)#，SGDM在CV里面应用较多，而Adam则基本横扫NLP、RL、GAN、语音合成等领域。
#1970->479
def input_data():
    train_data= torchvision.datasets.MNIST(
        root='.\data',
        download=True,
        train=True,  #训练数据集
        transform=torchvision.transforms.ToTensor()
    )
    test_data = torchvision.datasets.MNIST(
        root='.\data',
        download=True,
        train=False,  # 测试数据集
        transform=torchvision.transforms.ToTensor()
    )
    return train_data,test_data

def train(model):
    train_losses=[]
    test_losses=[]
    print('inputing data...')
    train_data, test_data = input_data()
    
    train_dataloader=DataLoader(train_data,batch_size=args.num_batch_size,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.num_batch_size,shuffle=False)
    print('begin epoches...')
    print('epoch | train: loss | test: loss  Accuracy')
    a=time.time()
    for epoch_num in range(1,args.num_epochs+1):
        train_loss=0
        for i,(images,labels) in enumerate(train_dataloader):#enumerate()返回俩个值一个是序号，也就是在这里的batch地址i，一个是数据train_ids，包括图片和标签
            #images 64*1*28*28  28*28大小，1通道，64个样本一个batch
            images=images.to(device)
            # print(images.shape)
            labels=labels.to(device)
            #print(next(model.parameters()).device)
            labels_hat = model(images)
            #loss=loss_func(labels,labels_hat) #这个会报错，因为labels是B维，labels_hat是B*C维
            loss=loss_func(labels_hat,labels)
            train_loss+=loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()#清空梯度
            #print('{}/'.format(i+1))

        test_loss=0
        test_num=0
        correct=0
        with torch.no_grad():
            for i,(images,labels) in enumerate(test_dataloader):
                images=images.to(device)
                labels=labels.to(device)
                labels_hat=model(images)
                loss=loss_func(labels_hat,labels)
                test_loss+=loss
                _,predict=torch.max(labels_hat,1)
                test_num+=labels.shape[0]
                correct+=(predict==labels).sum().item()
        
        print('{}\t{:.3f}\t{:.3f}\t{:.6f}'.format(epoch_num,test_loss,train_loss,correct/test_num))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    b=time.time()
    print('time: {}'.format(b-a))
    return torch.tensor(train_losses,device='cpu'),torch.tensor(test_losses,device='cpu')

train_losses,test_losses=train(model)
x=range(len(train_losses))
plt.figure(2)
plt.plot(x,train_losses)
plt.plot(x,test_losses)
plt.show()
torch.save(model.state_dict(), 'model.ckpt')
