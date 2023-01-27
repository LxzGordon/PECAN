import numpy as np
import torch as th

shapes={'simple':[5,4,20],
    'unident_s':[9,5,20],
    'random0':[5,5,20],
    'random1':[5,5,20],
    'random3':[8,5,20]}
class self_attn(th.nn.Module):
    def __init__(self):
        super(self_attn,self).__init__()
        self.Q=th.nn.Linear(128,128)
        self.K=th.nn.Linear(128,128)
        self.V=th.nn.Linear(128,128)
    
    def forward(self,x):
        q=self.Q(x).view(-1,10,128)
        k=self.K(x).view(-1,10,128)
        k=th.cat([th.t(key).view(-1,128,10) for key in k],dim=0)
        v=self.V(x).view(-1,10,128)
        weights=th.softmax(th.bmm(q,k)/np.sqrt(128),dim=1)
        x=th.bmm(weights,v).reshape(-1,10,128)
        return x


class Net(th.nn.Module):
    def __init__(self,layout):
        super(Net,self).__init__()
        self.sp=shapes[layout]
        self.conv1=th.nn.Conv2d(
            in_channels=self.sp[0],
            out_channels=25,
            kernel_size=(5,5),
            padding=(2,2)
        )
        self.conv2=th.nn.Conv2d(
            in_channels=25,
            out_channels=25,
            kernel_size=(3,3),
            padding=(1,1)
        )
        self.conv3=th.nn.Conv2d(
            in_channels=25,
            out_channels=25,
            kernel_size=(3,3)
        )
        # transition encoding
        self.fc1=th.nn.Linear(25*(self.sp[1]-2)*(self.sp[2]-2),64)
        self.fc2=th.nn.Linear(64+6,128)
        self.attn=[self_attn().cuda() for _ in range(5)]
        #self.attn=self_attn().cuda()
        # transition sum encoding
        self.fc4=th.nn.Linear(128,128)

        # trajectory sum encoding
        self.fc5=th.nn.Linear(128,4)

        self.activation=th.nn.LeakyReLU()
    def forward(self,obs,act,eval=False,pbt_train=False):
        x=self.conv1(obs.view(-1,self.sp[0],self.sp[1],self.sp[2]))
        x=self.activation(x)

        x=self.conv2(x)
        x=self.activation(x)

        x=self.conv3(x)
        x=self.activation(x).view(x.shape[0],-1)

        x=self.fc1(x)
        x=self.activation(x)
        
        x=th.cat([x,act.reshape(-1,6)],dim=1)
        x=self.fc2(x)
        x=self.activation(x)
        for i in range(len(self.attn)):
            x=self.attn[i](x)
        
        if not eval:
            x=th.sigmoid(x).reshape(-1,obs.shape[1],10,128)
            
            x=x.sum(dim=2).reshape(-1,128)
            x=self.fc4(x)
            x=th.sigmoid(x).reshape(-1,obs.shape[1],128)
            x1=x.sum(dim=1).reshape(-1,128)
            x1=self.fc5(x1)
            if pbt_train:
                return x1
            #different combinaiton of the trajs
            x2=x[:,0]+x[:,1]
            x2=self.fc5(x2)
            x3=x[:,0]+x[:,2]
            x3=self.fc5(x3)
            x4=x[:,2]+x[:,1]
            x4=self.fc5(x4)
            x5=x[:,0]
            x5=self.fc5(x5)
            x6=x[:,0]
            x6=self.fc5(x6)
            x7=x[:,0]
            x7=self.fc5(x7)
            return [x1,x2,x3,x4,x5,x6,x7]
        if eval:
            x=th.sigmoid(x).reshape(-1,obs.shape[0],10,128)
            
            x=x.sum(dim=2).reshape(-1,128)
            x=self.fc4(x)
            x=th.sigmoid(x).reshape(-1,obs.shape[0],128)
            x1=x.sum(dim=1).reshape(-1,128)
            x1=self.fc5(x1)
            return x1