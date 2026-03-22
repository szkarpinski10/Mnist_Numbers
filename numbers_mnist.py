import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets 
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.optim as optim

train_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(f"train_dataloader: {len(train_data)}, test_dataloader={len(test_data)}")
BATCH_SIZE=64

train_dataloader=DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)
test_dataloader=DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"train_dataloader: {len(train_dataloader)}, test_dataloader={len(test_dataloader)}")

#visualization
fig=plt.figure(figsize=(9,9))
rows,cols=4,4
class_name=train_data.classes

for i in range(1,rows*cols+1):
    random_idx=torch.randint(0,len(train_data),(1,)).item()
    img, label=train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap="grey")
    plt.title(class_name[label])
    plt.axis(False)

plt.show()

class MNIST_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3136,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128,out_features=10)
        )

    def forward(self,x):
        x=self.conv_layer1(x)
        x=self.conv_layer2(x)
        x=self.classifier(x)
        return x
        
device="cuda" if torch.cuda.is_available() else "cpu"
Model_0=MNIST_Model().to(device)

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=Model_0.parameters(),lr=0.001)

def learning_loop(model, dataloader, loss_fn,optimizer):
    train_loss=0
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device), y.to(device)
        y_preds=model(X)
        loss=loss_fn(y_preds,y)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss/len(dataloader)

def test_loop(model,dataloader,loss_fn):
    test_loss=0
    correct=0
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            y_preds=model(X)
            test_loss+=loss_fn(y_preds,y).item()
            correct+=(y_preds.argmax(dim=1)==y).sum().item()
    
    accuracy=correct/len(dataloader.dataset)*100
    return test_loss/len(dataloader),accuracy

EPOCHS=9

for epoch in range(EPOCHS):
    training_loss=learning_loop(Model_0,train_dataloader,loss_fn,optimizer)
    test_loss, test_acc=test_loop(Model_0,test_dataloader,loss_fn)
    print(f"Epoch: {epoch+1} | train loss: {training_loss:.4f} | test loss: {test_loss:.4f} | accuracy: {test_acc:.2f}%")


# Wykresy
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
