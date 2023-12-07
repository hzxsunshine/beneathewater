import torch
import torch.nn as nn

# import torch.nn.functional as F
from configs.constants import contrastive_pairs
import numpy as np
import clip_custom
import math

import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, ToTensor


def transform_():
    """
    Transform for CLIP
    """
    return Compose([
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip_custom.load("ViT-B/32", device=self.device)
        self.model.float()
        for para in self.model.parameters():
            para.requires_grad = False
        self.model.eval()

    def forward(self, text):
        text = clip_custom.tokenize(text).to(self.device)
        return self.model.encode_text(text)


class CLIPContrastive(nn.Module):
    """
    CLIP contrastive loss
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip_custom.load("ViT-B/32", device=self.device)
        self.model.float()
        self.preprocess = transform_()
        for para in self.model.parameters():
            para.requires_grad = False
        self.model.eval()

    def forward(self, images):
        # images should be 4D tensor of shape (N, C, H, W)
        batch_size = images.shape[0]
        score = 0
        for i in range(batch_size):
            image = self.preprocess(images[i]).unsqueeze(0)
            text = clip_custom.tokenize(np.ravel(contrastive_pairs)).to(self.device)

            logits_per_image, _ = self.model(image, text)
            logits_per_image = logits_per_image.view(len(contrastive_pairs), 2)
            probs = logits_per_image.softmax(dim=-1)

            # mse:
            score += torch.mean((probs[:, 1] * 255) ** 2)
            
        return score / batch_size
    
class CLIPFeature(nn.Module):
    """
    CLIP Perceptual Loss
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip_custom.load("RN101", device=self.device)
        self.model.float()
        for para in self.model.parameters():
            para.requires_grad = False
        self.preprocess = transform_()
        self.model.eval()

    def forward(self, outputs, raws, weights=[1., 1., 1., 1., .5]):
        # images should be 4D tensor of shape (N, C, H, W)
        batch_size = outputs.shape[0]
        score = 0
        mse_loss = nn.MSELoss()
        for i in range(batch_size):
            image = self.preprocess(outputs[i]).unsqueeze(0)
            output_features = self.model.encode_image(image)[1]

            image = self.preprocess(raws[i]).unsqueeze(0)
            raw_features = self.model.encode_image(image)[1]

            for i in range(len(weights)):
                score += weights[i] * mse_loss(255 * output_features[i], 255 * raw_features[i])

        return score / batch_size / sum(weights)


### Waternet Base ###
class ConfidenceMapGenerator(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        # Confidence maps
        # Accepts input of size (N, 3*4, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=128, kernel_size=7, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=5, dilation=1, padding="same"
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding="same"
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=1, dilation=1, padding="same"
        )
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, dilation=1, padding="same"
        )
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, dilation=1, padding="same"
        )
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding="same"
        )
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        # print(out.shape)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.sigmoid(self.conv8(out))
        out1, out2, out3 = torch.split(out, [1, 1, 1], dim=1)
        return out1, out2, out3


class Refiner(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=7, dilation=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        # print(x.shape, xbar.shape)
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    waternet = WaterNet()
    in = torch.randn(16, 3, 112, 112)
    waternet_out = waternet(in, in, in, in)
    waternet_out.shape
    # torch.Size([16, 3, 112, 112])
    """

    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)
        refined_wb = self.wb_refiner(x, wb)
        refined_ce = self.ce_refiner(x, ce)
        refined_gc = self.gc_refiner(x, gc)
        return (
                torch.mul(refined_wb, wb_cm)
                + torch.mul(refined_ce, ce_cm)
                + torch.mul(refined_gc, gc_cm)
        )

if __name__ == "__main__":
    pass
