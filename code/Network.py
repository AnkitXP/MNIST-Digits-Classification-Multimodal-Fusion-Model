import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MNIST_Classifier, self).__init__()

        WR_embed = int(input_size * 0.6)
        SP_embed = input_size - WR_embed

        self.wr_encoder = WR_Encoder(WR_embed)
        self.sp_encoder = SP_Encoder(SP_embed)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear1 = nn.Linear(hidden_size + input_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)

    def forward(self, x_wr, x_sp):
        x_wr = self.wr_encoder(x_wr)
        x_sp = self.sp_encoder(x_sp) 
        x = torch.cat((x_wr, x_sp), dim=1) 
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        x = torch.cat((x_wr, lstm_out[:, -1, :], x_sp), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.log_softmax(x)
        
        return x

class WR_Encoder(nn.Module):

    def __init__(self, WR_embed):
        super(WR_Encoder, self).__init__()

        self.dropout = nn.Dropout2d(0.25)

        self.conv_1_32_5x5 = nn.Conv2d(1, 32, 5) # Bx1x28x28 -> Bx32x24x24
        self.conv_32_32_5x5 = nn.Conv2d(32, 32, 5, bias=False) # Bx32x20x20
        self.batchnorm_32 = nn.BatchNorm2d(32) # Bx32x20x20
        self.maxpool_2x2 = nn.MaxPool2d(2, 2) # Bx32x10x10    
        self.conv_32_64_3x3 = nn.Conv2d(32, 64, 3, bias=False) # Bx64x8x8 
        self.conv_64_64_3x3 = nn.Conv2d(64, 64, 3, bias = False) # Bx64x6x6
        self.batchnorm_64 = nn.BatchNorm2d(64) # Bx64x6x6
        self.linear_576_256 = nn.Linear(576, 256, bias=False)
        self.linear_256_128 = nn.Linear(256, 128, bias=False)
        self.linear_128_WR_embed = nn.Linear(128, WR_embed, bias=False)
        self.batchnorm_WR_embed = nn.BatchNorm1d(WR_embed)
        

    def forward(self, x):
        x = self.conv_1_32_5x5(x)
        x = self.conv_32_32_5x5(x)
        x = self.batchnorm_32(x)
        x = F.relu(x)
        x = self.maxpool_2x2(x)
        x = self.dropout(x)
        x = self.conv_32_64_3x3(x)
        x = self.conv_64_64_3x3(x)
        x = self.batchnorm_64(x)
        x = F.relu(x)
        x = self.maxpool_2x2(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.linear_576_256(x)
        x = self.linear_256_128(x)
        x = self.linear_128_WR_embed(x)
        x = self.batchnorm_WR_embed(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x
    

class SP_Encoder(nn.Module):
    def __init__(self, SP_embed):
        super(SP_Encoder, self).__init__()

        self.batch_norm_input = nn.BatchNorm1d(507)

        self.linear_507_384 = nn.Linear(507, 384, bias=False)
        self.linear_384_256 = nn.Linear(384, 256, bias=False)
        self.linear_256_128 = nn.Linear(256, 128, bias=False)
        self.linear_128_64 = nn.Linear(128, 64, bias=False)
        self.linear_64_SP_embed = nn.Linear(64, SP_embed, bias=False)

        self.dropout = nn.Dropout(0.25)
        self.dropout_input = nn.Dropout(0.1)

    def forward(self, x):
        x = self.batch_norm_input(x)
        x = self.dropout_input(x)
        x = F.relu(self.linear_507_384(x))
        x = self.dropout(x)
        x = F.relu(self.linear_384_256(x))
        x = self.dropout(x)
        x = F.relu(self.linear_256_128(x))
        x = self.dropout(x)
        x = F.relu(self.linear_128_64(x))
        x = self.dropout(x)
        x = F.relu(self.linear_64_SP_embed(x))
        x = self.dropout(x)

        return x