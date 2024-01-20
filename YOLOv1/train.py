import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import PascalVOC2012
from loss import Yolov1Loss


seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Yolov1().to(device)
criterion = Yolov1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)


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

loss_list = []
for epoch in range(100):

    loss_tmp = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_tmp += loss.item()


    loss_list.append(loss_tmp)
    print(str(epoch)+'epoch loss = '+str(loss_list[-1]))
    if epoch % 10 == 9:
        torch.save(model.state_dict(), 'checkpoint/'+str(epoch+1)+'epoch.pth')
