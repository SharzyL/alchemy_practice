import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
from PIL import Image


class Waifu(data.Dataset):
    def __init__(self, path='../_data/waifu'):
        self.path = Path(path)
        self.waifu_paths = list(self.path.iterdir())
        self.transform = transforms.Compose((
            transforms.Resize(96),
            transforms.ToTensor(),
        ))

    def __getitem__(self, item):
        return self.transform(
            Image.open(self.waifu_paths[item])
        )

    def __len__(self):
        return len(self.waifu_paths)


if __name__ == '__main__':

    import torch
    import torch.utils.data as data

    batch_size = 64

    WaifuLoader = data.DataLoader(
        dataset=Waifu(),
        batch_size=batch_size,
        shuffle=True
    )

    from gan.dcgan import dcgan

    z_dim = 32
    channels = [512, 256, 128, 64, 3]
    kernel_size = [4, 5, 4, 4, 4]
    stride = [2, 3, 2, 2, 2]
    padding = [0, 1, 1, 1, 1]

    generator, discriminator = dcgan(kernel_size, z_dim, channels, stride, padding)

    def get_noise(size):
        return torch.randn(size=(size, z_dim, 1, 1)).cuda()

    # for i, images in enumerate(WaifuLoader):
    #     noise = get_noise(batch_size)
    #     generated = generator(noise)
    #     print(generated.shape)
    #     discriminated = discriminator(generated)
    #     print(discriminated.shape)
    #     print(discriminator(images.cuda()))
    #     break

    print(generator)
    print()
    print(discriminator)

    generator(get_noise(1))
