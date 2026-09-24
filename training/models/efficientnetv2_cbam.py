import torch
from torch import nn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel=7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                     nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        attention = self.channel(x.mean((2, 3), keepdim=True)) + self.channel(x.amax((2, 3), keepdim=True))
        x = x * attention.sigmoid()
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.spatial(spatial).sigmoid()


class RimpangClassifier(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        import timm
        if config["architecture"] != "tf_efficientnetv2_b0.in1k":
            raise ValueError("This contract requires EfficientNetV2-B0.")
        self.backbone = timm.create_model(config["architecture"], pretrained=pretrained, num_classes=0, global_pool="")
        features = self.backbone.num_features
        self.attention = CBAM(features, config["cbamReduction"], config["cbamKernel"]) if config["cbam"] else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 10))

    def forward(self, x):
        x = self.attention(self.backbone.forward_features(x))
        return self.head(self.pool(x).flatten(1))
