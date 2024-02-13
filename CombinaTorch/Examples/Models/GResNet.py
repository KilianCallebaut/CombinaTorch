from typing import Type, Union, List, Optional, Callable, Tuple

import torch
from torch import nn, Tensor


def conv(in_planes: int, out_planes: int, stride: int = 1, kernel_size: Type[Union[int, Tuple[int]]] = 3) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            kernel1: Type[Union[int, Tuple[int]]] = (1, 1),
            kernel2: Type[Union[int, Tuple[int]]] = (3, 3),
            activation1=nn.ReLU(inplace=True),
            activation2=nn.ReLU(inplace=True)
    ) -> None:
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.F1conv1 = conv(inplanes, planes, stride, kernel1)
        self.F1bn1 = norm_layer(planes)
        self.F1relu = activation1
        self.F1conv2 = conv(planes, planes, kernel_size=kernel1)
        self.F1bn2 = norm_layer(planes)

        self.F2conv1 = conv(inplanes, planes, stride, kernel2)
        self.F2bn1 = norm_layer(planes)
        self.F2relu = activation2
        self.F2conv2 = conv(planes, planes, kernel_size=kernel2)
        self.F2bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        outF1 = self.F1conv1(x)
        outF1 = self.F1bn1(outF1)
        outF1 = self.F1relu(outF1)
        outF1 = self.conv2(outF1)
        outF1 = self.bn2(outF1)

        outF2 = self.F2conv1(x)
        outF2 = self.F2bn1(outF2)
        outF2 = self.F2relu(outF2)
        outF2 = self.conv2(outF2)
        outF2 = self.bn2(outF2)

        out = outF1 * outF2
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GResNet(nn.Module):

    def __init__(
            self,
            block: Type[BasicBlock],
            layers: List[int],
            kernel1: Type[Union[int, Tuple[int]]],
            kernel2: Type[Union[int, Tuple[int]]],
            activation1,
            activation2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,

    ) -> None:
        super(GResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(3, 3), stride=2,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        nn.ModuleList([self._make_layer(block, self.inplanes, filt, stride=2,
                                        kernel1=kernel1, kernel2=kernel2, activation1=activation1,
                                        activation2=activation2) for filt in layers])

        self.avgpool = nn.AvgPool2d((9, 9), stride=2)
        self.fc = nn.Linear(200)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block: Type[BasicBlock],
                    planes: int,
                    out: int,
                    stride: int = 2,
                    kernel1: Type[Union[int, Tuple[int]]] = (1, 1),
                    kernel2: Type[Union[int, Tuple[int]]] = (3, 3),
                    activation1=nn.ReLU(inplace=True),
                    activation2=nn.ReLU(inplace=True)
                    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1:
            downsample = nn.Sequential(
                conv(planes, out, stride, kernel_size=(1, 1)),
                norm_layer(out),
            )

        layers = []
        layers.append(block(inplanes=planes, planes=out, stride=stride, downsample=downsample,
                            activation1=activation1, activation2=activation2, kernel1=kernel1, kernel2=kernel2))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class GResNetBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_s, activation):
        f1 = nn.ModuleList([nn.Conv2d(in_channels=in_c,
                                      out_channels=out_c,
                                      kernel_size=(kernel_s, kernel_s),
                                      # stride=1,
                                      # padding=1,
                                      bias=False),
                            nn.BatchNorm2d(out_c),
                            activation])
        f2 = nn.ModuleList([nn.Conv2d(in_channels=in_c,
                                      out_channels=out_c,
                                      kernel_size=(kernel_s, kernel_s),
                                      # stride=1,
                                      # padding=1,
                                      bias=False),
                            nn.BatchNorm2d(out_c),
                            activation])


class GResNet(nn.Module):

    def __init__(
            self,
            task_list
    ):
        super().__init__()
        self.name = 'GResNet'

    def forward(self, x):
        return tuple()
