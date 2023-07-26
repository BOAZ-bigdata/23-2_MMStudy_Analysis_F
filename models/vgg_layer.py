
import torch.nn as nn
import torch

class VGGLayer():
    def __init__(self):
        self.vgg11_config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        self.vgg13_config = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        self.vgg16_config = [
                            64,
                            64,
                            "M",
                            128,
                            128,
                            "M",
                            256,
                            256,
                            256,
                            "M",
                            512,
                            512,
                            512,
                            "M",
                            512,
                            512,
                            512,
                            "M",
                        ]

        self.vgg19_config = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]

    def get_vgg_layers(self,config, batch_norm):
        layers = []
        in_channels = 3

        for c in config:
            assert c == "M" or isinstance(c, int)
            if c == "M":
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = c

        return nn.Sequential(*layers)