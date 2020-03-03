import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, kernel_size, z_dim, channels, stride, padding):
        super().__init__()
        self.network = nn.Sequential()
        depth = len(channels)
        for i in range(depth):
            layer = nn.Sequential()
            in_channel = z_dim if i == 0 else channels[i - 1]
            layer.add_module(f'DeConv{i + 1}', nn.ConvTranspose2d(
                in_channel, channels[i], kernel_size[i], stride[i], padding[i], bias=False
            ))
            if i == depth - 1:
                layer.add_module(f'Tanh', nn.Tanh())
            else:
                layer.add_module(f'BatchNorm{i + 1}', nn.BatchNorm2d(channels[i]))
                layer.add_module(f'ReLU{i + 1}', nn.ReLU())

            for m in layer.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight.data, 0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight.data, 1, 0.02)
                    nn.init.constant_(m.bias.data, 0)

            self.network.add_module(f'Layer{i}', layer)

    def forward(self, x):
        """
        :param x: normal noise with shape [batch_size, z_dim, 1, 1]
        :return: float-point image of shape [batch_size, channels, height, width], values from [0, 1].
        """
        return (self.network(x) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, kernel_size, channels, stride, padding):
        super().__init__()
        self.network = nn.Sequential()
        depth = len(channels)
        for i in range(depth)[::-1]:
            layer = nn.Sequential()
            layer_idx = depth - i
            out_channel = 1 if i == 0 else channels[i - 1]
            layer.add_module(f'Conv{layer_idx}', nn.Conv2d(
                channels[i], out_channel, kernel_size[i], stride[i], padding[i], bias=False
            ))
            if i == 0:
                layer.add_module('Sigmoid', nn.Sigmoid())
            else:
                layer.add_module(f'BatchNorm{layer_idx}', nn.BatchNorm2d(out_channel))
                layer.add_module(f'LeakyReLU{layer_idx}', nn.LeakyReLU(.2))

            for m in layer.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight.data, 0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight.data, 0, 0.02)
                    nn.init.constant_(m.bias.data, 0)

            self.network.add_module(f'Layer{layer_idx}', layer)

    def forward(self, x):
        """
        :param x: float-point image of shape [batch_size, channels, height, width], values from [0, 1].
        :return: [batch_size, discriminated], where discriminated range from [0, 1]
        """
        return self.network(x).view(-1, 1)


def dcgan(kernel_size, z_dim, channels, stride, padding):
    return (
        Generator(kernel_size, z_dim, channels, stride, padding).cuda(),
        Discriminator(kernel_size, channels, stride, padding).cuda(),
    )
