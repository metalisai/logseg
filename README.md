# Data

118 images were labeled with binary segmentation masks. The images are 128x128 pixels. The data was split into train (80%), validation (10%) and test (10%).  
The data labeling was done in CVAT using the true color images.

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
  
**The IoU on unseen data generally seems to be around 0.55-0.60.**  
  
Using all bands seems to perform better on validation, but worse on test. The image count is too low to draw any conclusions.  
The results are still close enough that it's better to continue with false color images, as it's easier to adapt to DINOv2.
  
Some prediction masks:  
![predictions](img/predictions.png)

# DINOv2

TODO
