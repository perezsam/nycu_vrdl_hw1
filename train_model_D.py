import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data'
NUM_CLASSES = 100
EPOCHS = 30
BATCH_SIZE = 32           
LEARNING_RATE = 2e-4      

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class AttentionHead(nn.Module):
    def __init__(self, in_features, reduction=16):
        super(AttentionHead, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features // reduction, in_features)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc2(self.relu(self.fc1(x))))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha, self.ls = gamma, alpha, label_smoothing
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce_loss) 
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

def main():
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(448, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.TrivialAugmentWide(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.avgpool = GeM()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(AttentionHead(in_features), nn.Dropout(p=0.3), nn.Linear(in_features, NUM_CLASSES))
    model = model.to(DEVICE)

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    print("Training Model D (SOTA ResNeXt-50)...")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")
        
    torch.save(model.state_dict(), 'model_D_resnext50_sota.pth')

if __name__ == "__main__":
    main()