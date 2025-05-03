import torch
import torch.nn.functional as F

# Define a custom segmentation model using DINOv2
class MySegmentationModel(torch.nn.Module):
    def __init__(self, dinov2_model):
        super(MySegmentationModel, self).__init__()
        self.dinov2_model = dinov2_model
        # per-pixel linear layer
        self.linear1 = torch.nn.Conv2d(
            in_channels=dinov2_model.embed_dim,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        features = self.dinov2_model.forward_features(x)
        B, P, C = features["x_norm_patchtokens"].shape
        S = int(P**0.5)
        x = features["x_norm_patchtokens"].reshape(B, S, S, C)
        x = x.permute(0, 3, 1, 2)
        x = self.linear1(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x
