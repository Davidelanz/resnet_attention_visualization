import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from . import resnet_at
except ImportError:
    import resnet_at


def image_folder(folder_dir):
    images = []
    for f in glob.iglob(os.path.join(folder_dir, "*")):
        images.append(np.asarray(Image.open(f)))
    images = np.array(images, dtype=object)
    return images


def plot_attention(model: resnet_at.ResNetAT,
                   images_dir: str, out_dir: str,
                   title: str):

    imgs = image_folder(images_dir)

    fig, ax = plt.subplots(nrows=len(imgs), ncols=5, figsize=(15, 3*len(imgs)))
    fig.suptitle(title)

    for row, im in enumerate(imgs):

        tr_center_crop = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model.eval()
        with torch.no_grad():
            x = tr_center_crop(im).unsqueeze(0)
            gs = model(x)

        ax[row][0].imshow(im)

        for i, g in enumerate(gs):
            ax[row][i+1].imshow(g[0], interpolation='bicubic', cmap='gray')
            ax[row][i+1].set_title(f'g{i}')

    # Save pdf versions
    Path(f"{out_dir}_pdf").mkdir(parents=True, exist_ok=True)
    fig_filename = os.path.join(f"{out_dir}_pdf", f"{title}.pdf")
    fig.savefig(fig_filename, bbox_inches='tight')

    # Save png versions
    Path(f"{out_dir}_png").mkdir(parents=True, exist_ok=True)
    fig_filename = os.path.join(f"{out_dir}_png", f"{title}.png")
    fig.savefig(fig_filename, bbox_inches='tight')


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_dir = os.path.join(dir_path, "inputs")
    out_dir = os.path.join(dir_path, "results")

    model = resnet_at.resnet18()
    plot_attention(model, img_dir, out_dir, "ResNet-18")
    model = resnet_at.resnet34()
    plot_attention(model, img_dir, out_dir, "ResNet-34")
    model = resnet_at.resnet50()
    plot_attention(model, img_dir, out_dir, "ResNet-50")
    model = resnet_at.resnet101()
    plot_attention(model, img_dir, out_dir, "ResNet-101")
    model = resnet_at.resnet152()
    plot_attention(model, img_dir, out_dir, "ResNet-152")

    model = resnet_at.resnext50_32x4d()
    plot_attention(model, img_dir, out_dir, "ResNeXt-50(32x4d)")
    model = resnet_at.resnext101_32x8d()
    plot_attention(model, img_dir, out_dir, "ResNeXt-101(32x8d)")

    model = resnet_at.wide_resnet50_2()
    plot_attention(model, img_dir, out_dir, "WideResNet-50(64*2)")
    model = resnet_at.wide_resnet101_2()
    plot_attention(model, img_dir, out_dir, "WideResNet-101(64*2)")
