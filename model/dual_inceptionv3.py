import torch
import torch.nn as nn
import torchvision

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvBNReLUFactorization(in_channels,out_channels,kernel_sizes,paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InceptionV3ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=5, padding=2),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleB(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1,7],paddings=[0,3]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[7,1],paddings=[3, 0]),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[1, 7], paddings=[0, 3]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[7, 1], paddings=[3, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[1, 7], paddings=[0, 3]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3,kernel_sizes=[7, 1], paddings=[3, 0]),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleC(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1,3],paddings=[0,1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3,1],paddings=[1, 0])

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3,stride=1,padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1],paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3],paddings=[0, 1])

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleD(nn.Module):
    def __init__(self, in_channels,out_channels1reduce,out_channels1,out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3,stride=2)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3,stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):
    def __init__(self, in_channels, out_channels1reduce,out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce,kernel_sizes=[1, 7], paddings=[0, 3]),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce,kernel_sizes=[7, 1], paddings=[3, 0]),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

class InceptionAux(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=5,stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out

class Dual_InceptionV3(nn.Module):
    def __init__(self, stage='train'):
        super(Dual_InceptionV3, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=80, kernel_size=3, stride=1),
            ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block3 = nn.Sequential(
            InceptionV3ModuleA(in_channels=192, out_channels1=64,out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV3ModuleA(in_channels=256, out_channels1=64,out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV3ModuleA(in_channels=288, out_channels1=64,out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64)
        )

        self.block4 = nn.Sequential(
            InceptionV3ModuleD(in_channels=288, out_channels1reduce=384,out_channels1=384,out_channels2reduce=64, out_channels2=96),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=128,  out_channels2=192, out_channels3reduce=128,out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160,  out_channels2=192,out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192,out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=192, out_channels2=192,out_channels3reduce=192, out_channels3=192, out_channels4=192),
        )
        if self.stage=='train':
            self.aux_logits1 = InceptionAux(in_channels=768,out_channels=1)
            self.aux_logits2 = InceptionAux(in_channels=768,out_channels=2)

        self.block5 = nn.Sequential(
            InceptionV3ModuleE(in_channels=768, out_channels1reduce=192,out_channels1=320, out_channels2reduce=192, out_channels2=192),
            InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384,  out_channels2=384, out_channels3reduce=448,out_channels3=384, out_channels4=192),
            InceptionV3ModuleC(in_channels=2048, out_channels1=320, out_channels2reduce=384, out_channels2=384,out_channels3reduce=448, out_channels3=384, out_channels4=192),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=8,stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, 32)
        self.fc1 = nn.Linear(32, 1)
        self.fc2 = nn.Linear(32, 2)
        self.fc3 = nn.Linear(1000, 32)
        self.fc4 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(p=0.75)
        self.dropout2 = nn.Dropout(p=0.75)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x2 = self.block1(x)
        x2 = self.block2(x2)
        x2 = self.block3(x2)
        aux2 = x2 = self.block4(x2)
        x2 = self.block5(x2)
        x2 = self.max_pool(x2)
        #x2 = self.dropout(x2)
        x2 = x2.view(x2.size(0),-1)
        x2 = self.linear(x2)

        x2=self.dropout2(x2)
        x2 = self.relu(x2)
        output2 = self.fc2(x2)  #32-2

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x1 = self.block3(x1)
        aux1 = x1 = self.block4(x1)
        x1 = self.block5(x1)
        x1 = self.max_pool(x1)
        #x1 = self.dropout(x1)
        x1 = x1.view(x1.size(0),-1)
        x1 = self.linear(x1)

        x1 = torch.cat((x1, x2), 1)  # 64
        x1 = self.fc4(x1)  # 64-32
        x1 =self.dropout1(x1)
        x1 = self.relu(x1)
        output1 = self.fc1(x1)   #32-1
        output1 = output1.squeeze(-1)

        if self.stage == 'train':
            aux1 = self.aux_logits1(aux1)
            aux1 = aux1.squeeze(-1)
            aux2 = self.aux_logits2(aux2)
            return output1,output2,aux1,aux2
        else:
            return output1,output2

if __name__=='__main__':
    model = Dual_InceptionV3()
    print(model)

    input = torch.randn(1, 3, 299, 299)
    aux,out = model(input)
    print(aux.shape)
    print(out.shape)
