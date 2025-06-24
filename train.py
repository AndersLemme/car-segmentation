import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import unet
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from dataset import carData
import glob
import os
import matplotlib.pyplot as plt

from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex
#from torchmetrics.classification import MulticlassJaccardIndex, MulticlassDice

#Hyperparameters
LEARNINGRATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #cpu on my system
BATCH_SIZE = 32
N_EPOCHS = 35
N_WORKERS = 0 #2
IMAGE_HEIGHT = 256 #736
IMAGE_WIDTH = 256 #992
PIN_MEMORY = True
LOAD_MODEL = False
IMAGE_DIR = "data/images/"
MASK_DIR = "data/masks/"


def train(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for i, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.long().to(device=DEVICE)
        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()             
        optimizer.step() 

        # Progress bar info
        loop.set_postfix(loss=loss.item())
    
def eval(model, loader, loss_fn, device=DEVICE):
    model.eval()
    total_loss = 0
    num_batches = 0

    # Metrics
    num_classes = 5  # Match your model’s out_channels
    dice_metric = DiceScore(num_classes=num_classes, average='macro').to(device)
    iou_metric = JaccardIndex(task="MULTICLASS", num_classes=num_classes, average='macro').to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.long().to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()

            preds_classes = torch.argmax(preds, dim=1)

            # Accumulate for metrics
            all_preds.append(preds_classes)
            all_targets.append(masks)
            num_batches += 1

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    avg_dice = dice_metric(all_preds, all_targets).item()
    avg_iou = iou_metric(all_preds, all_targets).item()
    avg_loss = total_loss / num_batches

    model.train()
    return avg_loss, avg_dice, avg_iou


def inference(model, test_loader, device=DEVICE):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            images, _ = batch  # we ignore true masks here
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)  # shape: (B, H, W)
            preds.extend(predicted.cpu().numpy())  # append each predicted mask
    return preds  # list of 2D numpy arrays (predicted masks)



def main(): #needed for workers on windows
    
    # Transformations
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # List all image files
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))

    # Ensure they align
    assert len(image_paths) == len(mask_paths)

    # Split into train (70%) and temp (30%)
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(image_paths, mask_paths, test_size=0.30, random_state=42)
    # Split temp into val (15%) and test (15%) ... I thought I could do this in one line with train_test_split?
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = carData(train_imgs, train_masks, transform=train_transform)
    val_dataset = carData(val_imgs, val_masks, transform=val_test_transform)
    test_dataset = carData(test_imgs, test_masks, transform=val_test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=N_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=N_WORKERS, pin_memory=PIN_MEMORY)

    # Model, loss, optimizer
    model = unet(in_channels=3, out_channels=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)

    #scaler = torch.amp.GradScaler()

    for epoch in range(N_EPOCHS):
        train(train_loader, model, optimizer, loss_fn)
        
        #eval(model, val_loader,loss_fn)
        # Optionally run eval on val_loader here
        
        val_loss, val_dice, val_iou = eval(model, val_loader, loss_fn)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    
    #Call test function - Inference
    predicted_masks = inference(model, test_loader)

    plt.imshow(predicted_masks[0])
    plt.title("Example predicted mask from test set 0")
    plt.show()
    plt.imshow(predicted_masks[1])
    plt.title("Example predicted mask from test set 1")
    plt.show()
    plt.imshow(predicted_masks[2])
    plt.title("Example predicted mask from test set 2")
    plt.show()




    ## Finally store model.
    model_path = "./models/model2_35epochs.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main() #to avoid issues while running non workde




