import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 第一层卷积（1D卷积）：
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation1 = nn.ReLU()
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
        
        # 第二层卷积（1D卷积）：
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.batchnorm5 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation5 = nn.ReLU()
        self.pooling5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 第三层深度可分离卷积（1D卷积）：
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, groups=64)
        self.batchnorm6 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation6 = nn.ReLU()
        self.pooling6 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        #第四层普通卷积（1D卷积）：
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation8 = nn.ReLU()
        self.pooling8 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        # 第五层全连接层：
        self.fc1 = nn.Linear(in_features=7424, out_features=1024)
        self.batchnorm9 = nn.BatchNorm1d(num_features=1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation9 = nn.ReLU()
        # 第六层全连接层：
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm10 = nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation10 = nn.ReLU()
        # 第七层全连接层：
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.batchnorm11= nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation11= nn.ReLU()
        # 第八层全连接层：
        self.out = nn.Linear(in_features=128, out_features=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # print("输入维度，====，",x.shape)
        x = x.unsqueeze(1)  # 添加一个维度，将输入变为(batch_size, channels, seq_len)
        
        # 第一层卷积：
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        
        # 第二层普通卷积：
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.activation5(x)
        x = self.pooling5(x)
        
        # 第三层深度可分离卷积：
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.activation6(x)
        x = self.pooling6(x)

        # 第四层普通卷积：
        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.activation8(x)
        x = self.pooling8(x)

        # 展开成一维向量
        x = torch.flatten(x, start_dim=1)

        # 第五层全连接层：
        x = self.fc1(x)
        x = self.batchnorm9(x)
        x = self.activation9(x)
        
        # 第六层全连接层：
        x = self.fc2(x)
        x = self.batchnorm10(x)
        x = self.activation10(x)
        # 第七层全连接层：
        x = self.fc3(x)
        x = self.batchnorm11(x)
        x = self.activation11(x)
        # 第八层输出层：
        x = self.out(x)

        x = self.sigmoid(x)
        #x=F.softmax(x, dim=1)
        
        return x