import torch.nn as nn

class my_MLP(nn.Module):
    def __init__(self):
        super(my_MLP,self).__init__()
        self.input_h=28
        self.input_w=28
        self.passageway=1
        self.hiddendim1=20#隐藏层的维度
        self.hiddendim2=40#隐藏层的维度
        self.classes=1000#drop数
        self.outputdim=10#输出的类别数
        # self.linear1=nn.Linear(self.passageway*self.input_h*self.input_w,self.hiddendim1)
        # self.conv1=nn.Conv2d(1,self.hiddendim1,kernel_size=5,padding=1)#1*28*28->20*24*24
        # self.relu=nn.ReLU()
        # self.conv2=nn.Conv2d(self.hiddendim1,self.hiddendim2,kernel_size=5,padding=1)#20*24*24->40*20*20
        # self.linear1=nn.Linear(self.hiddendim2*4*4,self.classes)
        # self.linear2=nn.Linear(self.classes,self.outputdim)
        # self.linear=nn.Linear(self.hiddendim2*4*4,self.outputdim)
        # self.softmax=nn.Softmax(dim=1)
        # self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # self.dropout=nn.Dropout(0.5)
        self.conv1=nn.Sequential(
            nn.Conv2d(              #--> (1,28,28)
                in_channels=1,      #传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=20,    #输出的图片是几层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),    # 2d代表二维卷积           --> (16,28,28)
            nn.ReLU(),              #非线性激活层
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (16,14,14)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(              #--> (20,14,14)
                in_channels=20,     #传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=40,    #输出的图片是几层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),    # 2d代表二维卷积           --> (40,14,14)
            nn.ReLU(),              #非线性激活层
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (40,7,7)
        )
        self.linear1=nn.Linear(40*7*7,100)
        self.linear2=nn.Linear(100,100)
        self.linear3=nn.Linear(100,10)
        self.softmax=nn.Softmax(dim=1)
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()




        '''self.net=nn.Sequential(
            nn.Linear(self.input_h*self.input_w,256),
            nn.ReLu(),
            nn.Linear(256,10)
        )'''

    def forward(self,input):
        #input  B*(P*H*W) -> B*F
        # input=input.view(input.shape[0],-1)
        # input_linear1=self.linear1(input)
        # input_relu1=self.relu1(input_linear1)
        # input_linear2=self.linear2(input_relu1)
        # input_relu2=self.relu1(input_linear2)
        # output=self.linear3(input_relu2)
        
        '''
        input_conv1=self.conv1(input)
        input_pool1=self.pool(input_conv1)
        input_relu1=self.relu(input_pool1)
        input_conv2=self.conv2(input_relu1)
        input_pool2=self.pool(input_conv2)
        input_relu2=self.relu(input_pool2)

        input_dropout=self.dropout(input_relu2)
        input_linear1=self.linear1(input_dropout)
        # input_linear1=self.linear1(input_relu2)
        input_relu=self.relu(input_linear1)
        input_dropout=self.dropout(input_relu)

        input_linear2=self.linear2(input_dropout)
        input_relu=self.relu(input_linear2)
        output=self.softmax(input_relu)
        #output=self.softmax(input_relu)
        '''

        x=self.conv1(input)
        x=self.conv2(x)
        x=x.view(x.shape[0],-1)

        x=self.linear1(x)
        x=self.linear2(x)
        output=self.linear3(x)
        # output=self.softmax(x)
        # x=self.relu(x)
        # output=self.softmax(x)
        return output