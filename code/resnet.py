"""
Minimal ResNet-18 identical to torchvision.models.resnet18 (same state_dict
keys), so the official ImageNet weights load without installing torchvision
(which could replace the XPU build of PyTorch).
Weights: IMAGENET1K_V1, https://download.pytorch.org/models/resnet18-f37072fd.pth
"""
import torch
import torch.nn as nn

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = None
        if stride != 1 or inp != out:
            self.downsample = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                            nn.BatchNorm2d(out))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + idt)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def resnet18(pretrained=True):
    m = ResNet18()
    if pretrained:
        sd = torch.hub.load_state_dict_from_url(WEIGHTS_URL, progress=True, map_location="cpu")
        m.load_state_dict(sd)
    return m
