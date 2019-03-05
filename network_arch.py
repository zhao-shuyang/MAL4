import torch
import torch.nn as nn
import torch.nn.functional as Fx
import torch.nn.init as init
from torch import optim

class EmbNet(nn.Module):
    def __init__(self):
        super(EmbNet,self).__init__()
        self.embedding_length = 1024
        self.layer1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer3 = nn.MaxPool2d(2)
        self.layer4 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(32,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer6 = nn.MaxPool2d(2)
        self.layer7 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer9 = nn.MaxPool2d(2)
        self.layer10 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer12 = nn.MaxPool2d(2) #
        self.layer13 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer15 = nn.MaxPool2d(2) #
        self.layer16 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        self.layer17 = nn.MaxPool2d(2) # 
        #self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.ReLU())
        self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2), nn.BatchNorm2d(1024),nn.ReLU())
        
        #self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2), nn.ReLU(), nn.Tanh())
        #self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2), nn.Sigmoid())
        
        #self.layer19 = nn.Sequential(nn.Conv2d(128,128,kernel_size=1),nn.BatchNorm2d(128),nn.ReLU())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out).squeeze(3)
        
        #out = Fx.avg_pool2d(out, kernel_size=out.size()[2:]).squeeze()
        out = torch.mean(out, dim=2)
        #print (out.size())
        return out
    
    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        #print (state_dict)
        self.load_state_dict(state_dict)

class ASclassifier(nn.Module):
    def __init__(self,nclass):
        super(ASclassifier,self).__init__()
        #self.fc1 = nn.Sequential(nn.Linear(1024,1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(nn.Linear(1024,nclass), nn.Sigmoid())

    def forward(self, x) :
        #out = self.fc1(x)
        out = self.fc2(x)
        #out = Fx.avg_pool2d(out,kernel_size=out.size()[2:])
        #out = out.view(out.size(0),-1)
        return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

class CascadeNet(nn.Module):
    def __init__(self,nclass):
        super(CascadeNet,self).__init__()
        self.emb_net = EmbNet()
        self.cls_net = ASclassifier(nclass)
        
    def forward(self, x) :
        out = self.emb_net(x)
        out = self.cls_net(out)
        return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

        
