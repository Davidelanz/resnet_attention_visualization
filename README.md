# Visualize ResNet Attention Layers 

Tool for attention visualization in ResNets inner layers.
The models used are the ``torchvision`` pretrained ones 
(see [this link](https://pytorch.org/docs/stable/torchvision/models.html)
for further details)

A modified ``ResNet`` class, called ``ResNetAT``, is available
at ``resnet_at.py``, along with the finctions to initialize
the different ResNet architectures. ``ResNetAT``'s ``forward``
method is defined sucht that the inner layers' outputs
are available as model's outputs.

## Results

|Summary|   |   |   |   |
|---|---|---|---|---|
|[ResNet18](#resnet18)|[ResNet34](#resnet34)|[ResNet50](#resnet50)|[ResNet101](#resnet101)|[ResNet152](#resnet152)|
|![ResNet18](./results_png/ResNet-18.png)|![ResNet34](./results_png/ResNet-34.png)|![ResNet50](./results_png/ResNet-50.png)|![ResNet101](./results_png/ResNet-101.png)|![ResNet152](./results_png/ResNet-152.png)|
|[ResNeXt50_32x4d](#resnext50_32x4d)|[ResNeXt101_32x8d](#resnext101_32x8d)|[WideResNet50_2](#wideresnet50_2)|[WideResNet101_2](#wideresnet101_2)|
|![ResNeXt50_32x4d](./results_png/ResNeXt-50(32x4d).png)|![ResNeXt101_32x8d](./results_png/ResNeXt-101(32x8d).png)|![WideResNet50_2](./results_png/WideResNet-50(64*2).png)|![WideResNet101_2](./results_png/WideResNet-101(64*2).png)|

## Details

### ResNet18[↑](#results)
![ResNet18](./results_png/ResNet-18.png)
### ResNet34[↑](#results)
![ResNet34](./results_png/ResNet-34.png)
### ResNet50[↑](#results)
![ResNet50](./results_png/ResNet-50.png)
### ResNet101[↑](#results)
![ResNet101](./results_png/ResNet-101.png)
### ResNet152[↑](#results)
![ResNet152](./results_png/ResNet-152.png)

### ResNeXt50_32x4d[↑](#results)
![ResNeXt50_32x4d](./results_png/ResNeXt-50(32x4d).png)
### ResNeXt101_32x8d[↑](#results)
![ResNeXt101_32x8d](./results_png/ResNeXt-101(32x8d).png)

### WideResNet50_2[↑](#results)
![WideResNet50_2](./results_png/WideResNet-50(64*2).png)
###  WideResNet101_2[↑](#results)
![WideResNet101_2](./results_png/WideResNet-101(64*2).png)

