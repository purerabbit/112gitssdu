import torch
import torch.nn as nn
import fastmri

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Each RB comprised of two convolutional layers.  All layers had a kernel size of 3×3 and 64 channels.

    The first layer is followed by a ReLU activation.  The second is followed by a constant multiplication, with factor
    (by default) equal to 0.1.
    """
    def __init__(self, in_channels=64, out_channels=64, stride=1, const_multiple=0.1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self._scalar_mult_2 = const_multiple

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out *= self._scalar_mult_2
        identity = identity if self.downsample is None else self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, input_channels=2, output_channels=64):
        super().__init__()
        # layer of input and output convolution layers
        # 15 residual blocks (RB) with skip connections
            # Each RB comprised of two convolutional layers
                # first layer is followed by a rectified linear unit (ReLU)
                # second layer is followed by a constant multiplication layer, with factor C = 0.1 (55).
                # All layers had a kernel size of 3×3 and 64 channels
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3),padding=1)
        self.residual_blocks = nn.ModuleList()
        for i in range(15):
            self.residual_blocks.append(ResidualBlock())
        self.conv_last = nn.Conv2d(output_channels, 2, kernel_size=(3, 3), padding=1)
    def forward(self, x, z0, mask, x0):
        """

        Parameters
        ----------
        x
            A volume for an MRI image.

        Returns
        -------

        """
        z = self.conv1(x)
        for rb in self.residual_blocks:
            z = rb(z)
        z = self.conv_last(z)
        return x, z, mask, x0

class DC(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, x, z0, mask, x0):
        return (fastmri.ifft2c(fastmri.fft2c(z0.permute(0, 2, 3, 1)) 
        * (torch.ones_like(mask.float())-mask.float()) + fastmri.fft2c(x0.permute(0, 2, 3, 1)))).permute(0, 3, 1,2), z0, mask, x0


class MriSelfSupervised(nn.Module):
    def __init__(self, input_channels=2, output_channels=64):
        super().__init__()
        # layer of input and output convolution layers
        # 15 residual blocks (RB) with skip connections
            # Each RB comprised of two convolutional layers
                # first layer is followed by a rectified linear unit (ReLU)
                # second layer is followed by a constant multiplication layer, with factor C = 0.1 (55).
                # All layers had a kernel size of 3×3 and 64 channels
        self.net = nn.ModuleList()
        for i in range(10):
            self.net.append(ResNet())
            self.net.append(DC())
    def forward(self, x, z0, mask, x0):
        for n in self.net:
            x, z0, mask, x0 = n(x, z0, mask, x0)
        return x, z0, mask, x0

