import torch
import os
import pandas as pd
from PIL import Image


class PascalVOC2012(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform=None):

        self.transform = transform
        self.annotations = pd.read_csv(csv_file, header=None)
        self.images_folder = os.path.join(img_dir)
        self.labels_folder = os.path.join(label_dir)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.annotations.iloc[idx, 0])
        label_name = os.path.join(self.labels_folder, self.annotations.iloc[idx, 1])

        image = Image.open(img_name)
        label = self.read_label_file(label_name)

        if self.transform:
            image = self.transform(image)

        return image, label

    def read_label_file(self, label_file):
        boxes = []
        with open(label_file, 'r') as file:
            for line in file.readlines():
                class_label, xmin, ymin, xmax, ymax = [float(x) for x in line.strip().split()]
                x, y, w, h = (xmin+xmax)/2., (ymin+ymax)/2., xmax-xmin, ymax-ymin
                boxes.append([int(class_label), x, y, w, h])

        label = torch.zeros(7, 7, 30)

        for box in boxes:
            class_label, x, y, w, h = box
            i, j = int(7*x), int(7*y)
            x_cell, y_cell, w_cell, h_cell = 7*x-i, 7*y-j, 7*w, 7*h
            if label[i, j, 20] == 0:
                label[i, j, 20] = 1
                label[i, j, 21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label[i, j, class_label] = 1

        return label

