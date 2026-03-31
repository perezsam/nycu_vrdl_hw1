import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.amp import autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = './data/test'
TRAIN_DIR = './data/train' 

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p, self.eps = nn.Parameter(torch.ones(1)*p), eps
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

class RawImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg'))])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        return Image.open(os.path.join(self.root_dir, self.image_files[idx])).convert('RGB'), self.image_files[idx]

def custom_collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]

def main():
    train_set = datasets.ImageFolder(TRAIN_DIR)
    idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}

    print("Loading Models...")
    ma = models.resnet101(); ma.fc = nn.Linear(ma.fc.in_features, 100)
    ma.load_state_dict(torch.load('model_A_resnet101.pth'))
    
    mb = models.resnext50_32x4d(); mb.fc = nn.Linear(mb.fc.in_features, 100)
    mb.load_state_dict(torch.load('model_B_resnext50.pth'))
    
    md = models.resnext50_32x4d(); md.avgpool = GeM()
    md.fc = nn.Sequential(AttentionHead(2048), nn.Dropout(0.3), nn.Linear(2048, 100))
    md.load_state_dict(torch.load('model_D_resnext50_sota.pth'))

    ma, mb, md = ma.to(DEVICE).eval(), mb.to(DEVICE).eval(), md.to(DEVICE).eval()

    scales = [(512, 448), (576, 512), (664, 600)]
    tta_transforms = []
    for resize, crop in scales:
        tta_transforms.append(transforms.Compose([
            transforms.Resize(resize), transforms.TenCrop(crop),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(c) for c in crops])),
            transforms.Lambda(lambda imgs: torch.stack([transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])(i) for i in imgs]))
        ]))

    test_loader = DataLoader(RawImageDataset(TEST_DIR), batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate)
    results = []

    print("Running Inference...")
    with torch.no_grad():
        for imgs, filenames in test_loader:
            batch_size = len(imgs)
            prob_a, prob_b, prob_d = 0, 0, 0
            
            for t in tta_transforms:
                scale_inputs = torch.stack([t(img) for img in imgs]).to(DEVICE)
                bs, n_crops, c, h, w = scale_inputs.size()
                flat_inputs = scale_inputs.view(-1, c, h, w)
                
                with autocast('cuda'):
                    prob_a += torch.softmax(ma(flat_inputs), 1).view(bs, n_crops, -1).mean(1)
                    prob_b += torch.softmax(mb(flat_inputs), 1).view(bs, n_crops, -1).mean(1)
                    prob_d += torch.softmax(md(flat_inputs), 1).view(bs, n_crops, -1).mean(1)
                    
            final_blend = (0.15 * (prob_a / 3)) + (0.15 * (prob_b / 3)) + (0.70 * (prob_d / 3))
            _, preds = torch.max(final_blend, 1)
            
            for i in range(batch_size):
                results.append({'image_name': os.path.splitext(filenames[i])[0], 'pred_label': int(idx_to_class[preds[i].item()])})

    pd.DataFrame(results).to_csv('prediction.csv', index=False)
    print("Saved to prediction.csv")

if __name__ == "__main__":
    main()