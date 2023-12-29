import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# from mobileNet_CIFAR10 import MobileNet
# from mobileNet_ImageNet import MobileNet
from CustomMobileNet import CustomMobileNet


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=100,
                                           shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1000,
                                          shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CustomMobileNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))


for epoch in range(100):

    for _, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            _, predict = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
        accuracy = 100 * (correct / total)

        print('Accuracy of the network on the 50000 train images: %.2f %%' % (accuracy))


print('Finished Training')


# torch.save(model.state_dict(), 'checkpoint.pth')
# state_dict = torch.load('checkpoint.pth')
# model = CustomMobileNet().to(device)
# model.load_state_dict(state_dict)
#
# with torch.no_grad():
#     total = 0
#     correct = 0
#     for i, data in enumerate(test_loader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model.forward(inputs)
#         _, predict = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predict == labels).sum().item()
#     accuracy = 100 * (correct / total)
#
# print('Accuracy of the network on the 10000 test images: %.2f %%' % (accuracy))