from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNetAT(ResNet):
    """Attention maps of ResNeXt-101 32x8d.

    Overloaded ResNet model to return attention maps.
    """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]


def resnet18():
    base = models.resnet18(pretrained=True)
    model = ResNetAT(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(base.state_dict())
    return model


def resnet34():
    base = models.resnet34(pretrained=True)
    model = ResNetAT(BasicBlock, [3, 4, 6, 3])
    model.load_state_dict(base.state_dict())
    return model


def resnet50():
    base = models.resnet50(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(base.state_dict())
    return model


def resnet101():
    base = models.resnet101(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 23, 3])
    model.load_state_dict(base.state_dict())
    return model


def resnet152():
    base = models.resnet152(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 8, 36, 3])
    model.load_state_dict(base.state_dict())
    return model


def resnext50_32x4d():
    base = models.resnext50_32x4d(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
    model.load_state_dict(base.state_dict())
    return model


def resnext101_32x8d():
    base = models.resnext101_32x8d(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
    model.load_state_dict(base.state_dict())
    return model


def wide_resnet50_2():
    base = models.wide_resnet50_2(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 6, 3], width_per_group=64*2)
    model.load_state_dict(base.state_dict())
    return model


def wide_resnet101_2():
    base = models.wide_resnet101_2(pretrained=True)
    model = ResNetAT(Bottleneck, [3, 4, 23, 3], width_per_group=64*2)
    model.load_state_dict(base.state_dict())
    return model
