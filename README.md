# Car segmentation

## File descriptions:

- imgSize.py: outputs image sizes, Just used for checking.
- dataset.py: dataset for DataLoader, batchtraining (getitems) and augment both image and mask.
- unet.py: architecture of the unet model.
- train.py: 


whould it be better just to do this? ( maybe easier)
`
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=5)
`


## Tasks:

### 1: Data split

- train: 70%
- test: 15%
- validation: 15%

Maybe 80/10/10 had been better considering its quite small dataset, but i think its okay, at least after augmentation.

### 2: Model (Unet)

I chose to use U-Net. After a bit of research I found out that unet was developed for image segmentation and it is not very complex and works well on small dataset.
Since I dont have too much experience with segmentation but more with classification, I felt it was a good idea to learn U-net, and could from there make changes e.g.encoders to change the Model.


### 3: Augmentation/transformation

- Resize: uniform size
- Normailize: [1,0]

- 50% flip,
- 50% random rotation,


I actually wanted to test without any augmentation first to check the difference. and then add some more than what I did, but i haven't had time yet.

