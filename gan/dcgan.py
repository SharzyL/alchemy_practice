import torch.nn as nn


def dcgan(kernel_size, z_dim, channels, stride, padding):
    layers = len(channels)

    generator = nn.Sequential()
    for i in range(layers):
        in_channel = z_dim if i == 0 else channels[i - 1]
        generator.add_module(f'DeConv{i + 1}', nn.ConvTranspose2d(
            in_channel, channels[i], kernel_size[i], stride[i], padding[i], bias=False
        ))
        if i == layers - 1:
            generator.add_module(f'Tanh', nn.Tanh())
        else:
            generator.add_module(f'BatchNorm{i + 1}', nn.BatchNorm2d(channels[i]))
            generator.add_module(f'ReLU{i + 1}', nn.ReLU())

    for m in generator.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1, 0.02)
            nn.init.constant_(m.bias.data, 0)

    discriminator = nn.Sequential()
    for i in range(layers)[::-1]:
        layer = layers - i
        out_channel = 1 if i == 0 else channels[i - 1]
        discriminator.add_module(f'Conv{layer}', nn.Conv2d(
            channels[i], out_channel, kernel_size[i], stride[i], padding[i], bias=False
        ))
        if i == 0:
            discriminator.add_module('Sigmoid', nn.Sigmoid())
        else:
            discriminator.add_module(f'BatchNorm{layer}', nn.BatchNorm2d(out_channel))
            discriminator.add_module(f'LeakyReLU{layer}', nn.LeakyReLU(.2))

    for m in discriminator.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    return generator.cuda(), discriminator.cuda()
