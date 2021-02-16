import torch
import torch.nn as nn


# LeNet
class LeNet(nn.Module):
    def __init__(self, dp_ratio=0.2):
        super(LeNet, self).__init__()
        # linear layer (784 -> 300)
        self.fc1 = nn.Linear(28 * 28, 300)
        # linear layer (300 -> 100)
        self.fc2 = nn.Linear(300, 100)
        # linear layer (100 -> 10)
        self.fc3 = nn.Linear(100, 10)
        # dropout layer
        self.dropout = nn.Dropout(dp_ratio)
        # activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # flatten input
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Conv-4
class Conv_4(nn.Module):
    def __init__(self, dp_ratio=0.5):
        super(Conv_4, self).__init__()
        # convolution layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (128 * 8 * 8 -> 256)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        # dropout layer
        self.dropout = nn.Dropout(dp_ratio)
        # activation
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# VGG-19
class VGG(nn.Module):
    def __init__(self, dp_ratio=0.5):
        super(VGG, self).__init__()
        # convolution layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # linear layer (512 * 7 * 7 -> 4096)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        # dropout layer
        self.dropout = nn.Dropout(dp_ratio)

        # activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # 64
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        # 128
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        # 256
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x = self.pool(x)
        # 512
        x = self.conv9(x)
        x = self.activation(x)
        x = self.conv10(x)
        x = self.activation(x)
        x = self.conv11(x)
        x = self.activation(x)
        x = self.conv12(x)
        x = self.activation(x)
        x = self.pool(x)
        # 512
        x = self.conv13(x)
        x = self.activation(x)
        x = self.conv14(x)
        x = self.activation(x)
        x = self.conv15(x)
        x = self.activation(x)
        x = self.conv16(x)
        x = self.activation(x)
        # average pool
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x