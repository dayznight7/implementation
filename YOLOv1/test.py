import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import PascalVOC2012
from loss import Yolov1Loss


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from utils import intersection_over_union, calculate_tp_fp_fn_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Yolov1().to(device)
state_dict = torch.load('checkpoint/100epoch.pth')
model.load_state_dict(state_dict)
model.eval()
criterion = Yolov1Loss()


transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])
train_dataset = PascalVOC2012(
    csv_file='data/train.csv',
    img_dir='data/images',
    label_dir='data/labels',
    transform=transforms,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
)
test_dataset = PascalVOC2012(
    csv_file='data/test.csv',
    img_dir='data/images_test',
    label_dir='data/labels_test',
    transform=transforms,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=5,
    shuffle=False,
    drop_last=False,
)


tp, fp, fn, loss = calculate_tp_fp_fn_loss(train_loader, model, criterion=Yolov1Loss())
print(f"True Positives: {tp} False Positives: {fp}, False Negatives: {fn}")
print(f"Precision = {tp / (tp + fp)}")
print(f"Recall = {tp / (tp + fn)}")

