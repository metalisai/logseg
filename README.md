# Files

Report - report.pdf  
Baseline training code - train\_baseline.py  
DINOv2 training code - train\_dinov2.py  
Training images - images_[allbands|falsecolor|truecolor]  
Training masks - masks  

# Data

118 images were labeled with binary segmentation masks. The images are 128x128 pixels. The data was split into train (80%), validation (10%) and test (10%).  
The data labeling was done in CVAT using the true color images.
  
The images were fetched from Sentinel-2 using the SentinelHub API.  
False color images use (Band 8, Band 4, Band 3) as RGB.  
True color images use (Band 2, Band 3, Band 4) as RGB.  
All band images use all 13 bands.
  
The images are cut forests in Viljandi county, Estonia. The regions with cut forests were found using "Metsaregister" (Forest Register). All images are from 2020+, usually in May.  

# Baseline

## Model architecture

The model is a U-NET with pre-trained ResNet50 backbone from torchgeo.  

## Choosing color bands
True color images consistently performed worse, so they were excluded from further analysis.  
  
When training 40 epochs, the following test metrics were obtained:  
  
allbands  
│    test\_BinaryAccuracy    │    0.9827552437782288     │  
│  test\_BinaryJaccardIndex  │    0.5457581281661987     │  
│         test\_loss         │    0.06099435314536095    │  
  
│    test\_BinaryAccuracy    │    0.9844782948493958     │  
│  test\_BinaryJaccardIndex  │    0.5794427990913391     │  
│         test\_loss         │    0.05389002338051796    │  
  
│    test\_BinaryAccuracy    │    0.9826331734657288     │  
│  test\_BinaryJaccardIndex  │    0.5329545736312866     │  
│         test\_loss         │   0.057594358921051025    │  
  
validation jaccard indices:  
0.573 0.585 0.601  
  
falsecolor  
│    test\_BinaryAccuracy    │    0.9841026663780212     │  
│  test\_BinaryJaccardIndex  │     0.571663498878479     │  
│         test\_loss         │   0.050392430275678635    │  
  
│    test\_BinaryAccuracy    │      0.986083984375       │  
│  test\_BinaryJaccardIndex  │    0.6456664800643921     │  
│         test\_loss         │    0.03748343884944916    │  
  
│    test\_BinaryAccuracy    │     0.985365629196167     │  
│  test\_BinaryJaccardIndex  │    0.6254956126213074     │  
│         test\_loss         │    0.04146193712949753    │  
  
validation jaccard indices:  
0.512 0.546 0.524  
  
The most important metric here is BinaryJaccardIndex(IoU). Training IoU reached 0.80-0.90, but the model was overfitting with so little data.

Average IoU with all bands on test data was 0.55 and on validation data 0.59.
Average IoU with with false color on test data was 0.61 and on validation data 0.53.

## Results
  
**The IoU on unseen data generally seems to be around 0.55-0.60.**  
  
Using all bands seems to perform better on validation, but worse on test. The image count is too low to draw any conclusions.  
The results are still close enough that it's better to continue with false color images, as it's easier to adapt to DINOv2.
  
Some prediction masks:  
![predictions](img/predictions.png)

# DINOv2

A simple linear layer with bilinear interpolation on top of DINOv2 patch tokens seemed to perform best. Using more linear layers or convTranspose for upsampling did not improve the results.  

dinov2\_vits14  
**IoU on test data was usually around 0.45. On validation data best was usually around 0.55.**  
dinov2\_vitl14  
**IoU on test data was usually around 0.50. On validation data best was usually around 0.47.**  
  
```python
class MySegmentationModel(torch.nn.Module):
    def __init__(self, dinov2_model):
        super(MySegmentationModel, self).__init__()
        self.dinov2_model = dinov2_model
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
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x
```


![predictions](img/predictions_dino.png)
  
## More things to try
Combine U-NET and DINOv2 patch tokens.
