import torch
import torch.nn as nn


architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: [(tuples), (tuples), int]
    # tuples and then last integer represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S, self.B, self.C = split_size, num_boxes, num_classes
        self.cnn = self._create_conv_layers(self.architecture)
        self.fcs = self._create_linear_layers()

    def forward(self, x):
        out = self.cnn(x)
        out = self.fcs(out)     # out = torch.flatten(out, start_dim=1)
        out = out.view(out.size(0), self.S, self.S, self.C + self.B * 5)
        return out

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                # Tuple: (kernel_size, num_filters, stride, padding)
                layers += [CNNBlock(in_channels=in_channels, out_channels=x[1],
                                    kernel_size=x[0], stride= x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                # List: [(tuples), (tuples), int]
                for _ in range(x[2]):
                    layers += [CNNBlock(in_channels=in_channels, out_channels=x[0][1],
                                        kernel_size=x[0][0], stride=x[0][2], padding=x[0][3])]
                    layers += [CNNBlock(in_channels=x[0][1], out_channels=x[1][1],
                                        kernel_size=x[1][0], stride=x[1][2], padding=x[1][3])]
                    in_channels=x[1][1]
        return nn.Sequential(*layers)

    def _create_linear_layers(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.architecture[-1][1] * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )


def test():
    model = Yolov1()
    x = torch.randn(16, 3, 448, 448)
    model(x)


# test()

