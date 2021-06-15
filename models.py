from torch import nn
import torchvision
import torch


class SimpleConvNet(nn.Module):
    """
    Simple
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return nn.functional.softmax(self.layer4(out), dim=1)


class ComplexConvNet(nn.Module):
    """
    Complex
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ELU(),
            #nn.MaxPool2d(2),
            #nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(p=0.1)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1)
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 7)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return nn.functional.softmax(self.output(out), dim=1)


class InceptionConvNet(nn.Module):
    """
    Inception
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),  # 24, 24, 8
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        # first inception layer
        self.layer2single = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1),  # 24, 24, 16
            nn.ELU()
        )
        self.layer2a = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 24, 24, 32
            nn.ELU()
        )
        self.layer2b = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 24, 24, 32
            nn.ELU()
        )
        self.layer2c = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d(8, 16, kernel_size=1),  # 24, 24, 16
            nn.ELU()
        )
        # conv 2-3
        self.layer_conv23 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(p=0.2)
        )

        # second inception layer
        self.layer3single = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ELU()
        )
        self.layer3a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.layer3b = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.layer3c = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ELU()
        )

        # conv 3-4
        self.layer_conv34 = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2)
        )

        # third inception layer
        self.layer4single = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ELU()
        )
        self.layer4a = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.layer4b = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.layer4c = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ELU()
        )

        # # layer 4-5
        # self.layer_conv45 = nn.Sequential(
        #     nn.Conv2d(768, 256, kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ELU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(p=0.1)
        # )
        # # fourth inception layer
        # self.layer5single = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ELU()
        # )
        # self.layer5a = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ELU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ELU()
        # )
        # self.layer5b = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ELU(),
        #     nn.Conv2d(512, 512, kernel_size=5, padding=2),
        #     nn.ELU()
        # )
        # self.layer5c = nn.Sequential(
        #     nn.MaxPool2d(3),
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     nn.ELU()
        # )
        # final layers
        self.layer6 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(2048, 768, kernel_size=1),
        #     nn.BatchNorm2d(768),
        #     nn.ELU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(768, 256, kernel_size=1),
        #     nn.BatchNorm2d(256),
        #     nn.ELU(),
        #     nn.MaxPool2d(2)
        # )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 64, 32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            # nn.Linear(64, 16),
            # nn.ELU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        out_first_conv = self.layer1(x)
        # first inception
        single = self.layer2single(out_first_conv)
        a = self.layer2a(out_first_conv)
        b = self.layer2b(out_first_conv)
        out = torch.nn.functional.pad(out_first_conv, (24, 24, 24, 24), 'constant', 0)
        c = self.layer2c(out)
        out = torch.cat((single, a, b, c), 1)
        # conv 2-3
        out = self.layer_conv23(out)
        # second inception
        single = self.layer3single(out)
        a = self.layer3a(out)
        b = self.layer3b(out)
        out = torch.nn.functional.pad(out, (24, 24, 24, 24), 'constant', 0)
        c = self.layer3c(out)
        out = torch.cat((single, a, b, c), 1)
        # skip connection
        # out_first_conv = out_first_conv.repeat(1, 48, 1, 1)
        # out = torch.add(out_first_conv, out)
        # out = nn.ELU(out)
        # conv 3-4
        out = self.layer_conv34(out)
        # third inception
        single = self.layer4single(out)
        a = self.layer4a(out)
        b = self.layer4b(out)
        out = torch.nn.functional.pad(out, (12, 12, 12, 12), 'constant', 0)
        c = self.layer4c(out)
        out = torch.cat((single, a, b, c), 1)
        # # conv 4-5
        # out = self.layer_conv45(out)
        # # fourth inception
        # single = self.layer5single(out)
        # a = self.layer5a(out)
        # b = self.layer5b(out)
        # out = torch.nn.functional.pad(out, (6, 6, 6, 6), 'constant', 0)
        # c = self.layer5c(out)
        # out = torch.cat((single, a, b, c), 1)
        # final layers
        out = self.layer6(out)
        return nn.functional.softmax(self.output(out), dim=1)


def extend_vgg():
    model = torchvision.models.vgg16(pretrained=True)
    first_conv_layer = [ nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)

    model.classifier[-1] = nn.Linear(4096, 1000)

    model.classifier.add_module('7', nn.ReLU())

    model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))

    model.classifier.add_module('9', nn.Linear(1000, 7))

    model.classifier.add_module('10', nn.LogSoftmax(dim=1))

    for param in model.features[1:].parameters():  # disable grad for trained layers
        param.requires_grad = False
    return model
