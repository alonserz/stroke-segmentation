# stroke-segmentation
# Quick Start

## Install dependencies
- Install [pytorch](https://pytorch.org/get-started/locally/)
```
pip install -r requirements.txt
```

## Model training and inference 
`Brain_MRI_start_M.py` - file to train model.
`inference.py` - trained model inference.

## Models' weights
Will publish them later.

# Data description
Dataset consists of ~3500 256x256x2 (HeightxWidthxChannels) MRI images of brain.

# Augmentations
[Resize](https://explore.albumentations.ai/transform/Resize)                                                                                          
[D4](https://explore.albumentations.ai/transform/D4)                                                                                                  
[Rotate](https://explore.albumentations.ai/transform/Rotate)                                                                                          
[PixelDropout](https://explore.albumentations.ai/transform/PixelDropout)                                                                              
[RandomBrightnessContrast](https://explore.albumentations.ai/transform/RandomBrightnessContrast)                                                      
[ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform)                                                                      
[SymmetricElasticTransform](https://github.com/alonserz/stroke-segmentation/blob/adee728e2ac611d86b6dbb7e7532d64607b1d0ee/MRIDataset.py#L13C7-L13C32)

SymmetricElasticTransform splits image on two parts, mirrors one part and concatenate original part and mirrored.
Now i'm not really sure how SymmetricElasticTransform affects on final result. It increased values of Dice and Jaccard scores by 7 percents once, but must be tested more times with different encoders.
