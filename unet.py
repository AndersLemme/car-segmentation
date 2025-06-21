import torch
import torch.nn as nn
import torchvision.transforms.functional as TF #resize, tensor -> PIL, rotate, etc


class DoubleConv(nn.Module): #2 conv in each down/up block, encoding/decoding
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1), #kernel = 3x3, stride = 1, padding = same
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), #kernel = 3x3, stride = 1, padding = same
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 5, features=[64, 128, 256, 512]): #define all parts of unet
        super(unet, self).__init__()
        self.downs =nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))  # Feature* 2 because of concat connection
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.OutputLayer = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] #reverse list, to eaily connect to up bolcks

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection= skip_connections[i//2]

            #if shape of x mismatch skip_connection => resize x. (the concat layers have to be equal)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) #height + width 

            concat_skip =torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.OutputLayer(x)

def test():
    x = torch.randn((3, 1, 160, 160 ))
    model = unet(in_channels=1, out_channels=1)
    pred = model(x)
    print(pred.shape)
    print(x.shape)
    assert pred.shape == x.shape

if __name__ == "__main__":
    test()







"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ENCODER = "resnet34"
ENC_WT  = "imagenet"
CLASSES = ["bg","car","wheel","light","window"]

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENC_WT,
    in_channels=3,
    classes=len(CLASSES),
)

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss("multiclass")
    def forward(self, preds, targets):
        return 0.7*self.ce(preds, targets) + 0.3*self.dice(preds, targets)

criterion = ComboLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.004)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

# Training loop omitted for brevity – run 30 epochs, early‑stop on val mIoU.
"""


#....................................................................
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class DoubleConv(nn.Module):
"""
