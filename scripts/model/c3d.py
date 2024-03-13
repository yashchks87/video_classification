import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride = 1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride = 1, padding = 'same')
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3 = nn.Conv3d(128, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv4 = nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv5 = nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv6 = nn.Conv3d(256, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.linear1 = nn.Linear(2304, 2048)
        self.linear2 = nn.Linear(2048, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.pool4(x)
        x = self.conv6(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        # b5.view(b5.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    @torch.no_grad()
    def predictions(self, x):
        x = self.forward(x)
        return x



# Some notes
# N -> number of sequences (mini batch) -> regular batch size
# Cin -> number of channels (3 for rgb) -> 3 if RGB 1 if grayscale
# D -> Number of images in a sequence -> We are going to use 20 as number of frames in single video (Temporal info)
    # Also we will shuffle based on video on frames, order should be maintained in frames
# H -> Height of one image in the sequence -> Regular image height(Spatial info)
# W -> Width of one image in the sequence -> Regular image width(Spatial info)
    
# temp_data = torch.rand(1, 3, 20, 128, 128)
# Here 1 is batch size, 3 is number of channels, 20 is number of frames, 128 is height and 128 is width