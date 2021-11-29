import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNet(nn.Module):
    def __init__(self,num_classes=6):
        super(RNNet,self).__init__()

        # Defining some parameters
        self.feature_size=101
        self.hidden_size = 32 #define yourself 
        self.n_layers = 2
        self.num_classes=num_classes

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(
            input_size=self.feature_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.n_layers, 
            batch_first=True,
            dropout=0.2
        )   
        # Fully connected layer
        self.fc1 = nn.Sequential(
           nn.Linear(in_features=5152, out_features=128),
           nn.LeakyReLU()
        )

        self.fc2=nn.Sequential(
            nn.Linear(in_features=128, out_features=self.num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self,x):
        x=x.squeeze(dim=1) #删除第一维，得到[100,161,101]
        batch_size=x.size(0)
        #Initializing hidden state for first input using method defined below
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size,device=x.device) #[2,100,32]

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, h_0) #[100,161,32] #[2,100,32]

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out=torch.flatten(out,1) #[100,5152]
        out = self.fc1(out)      #[100,128]
        out = self.fc2(out)      #[100,6]

        return out,hidden
        """
        torch.Size([100, 161, 101])
        h_0 torch.Size([2, 100, 32])
        out torch.Size([100, 161, 32])
        hidden torch.Size([2, 100, 32])
        out1 torch.Size([100,5152])
        out2 torch.Size([100, 128])
        out3 torch.Size([100, 6])
        """

class ConvNet(nn.Module):
    # CNN
    def __init__(self,num_classes=6):
        '''
        ######################################
        define your nerual network layers here
        ######################################
        '''
        self.num_classes=num_classes
        self.feature_size=101
        self.frame_num=161
        #[100,1,161,101]
        super(ConvNet,self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
		# initialize second set of CONV => RELU => POOL layers
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32000, out_features=800),
            nn.ReLU()
        )

		# initialize our softmax classifier
        self.fc2=nn.Sequential(
            nn.Linear(in_features=800, out_features=self.num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''
        
        x=self.conv1(x)
        x=self.conv2(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.fc2(x)
        """
        torch.Size([100, 16, 80, 50])
        torch.Size([100, 32, 40, 25])
        torch.Size([100, 32000])
        torch.Size([100, 800])
        torch.Size([100, 6])
        """
        return x

class FcNet(nn.Module):
    # DNN
    def __init__(self,num_classes=6):
        '''
        ######################################
        define your nerual network layers here
        ######################################
        '''
        super(FcNet,self).__init__()
        self.num_classes=num_classes
        self.feature_size=101
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size//2),
            nn.LeakyReLU(),
            nn.Linear(self.feature_size//2, self.feature_size//2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(self.feature_size//2, self.feature_size//4),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4025,128),
            nn.LeakyReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128,self.num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''
        
        """
        [batch_size,1,frame_num,feature_size][100,1,161,101]
        用四张卡跑后为[25,1,161,101]
        """
        x=self.layers(x)        #[100,1,161,25]
        x=torch.flatten(x,1)    #[100,4025]
        x=self.fc1(x)           #[100,128]
        x=self.fc2(x)           #[100,6]

        return x
