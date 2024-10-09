import torch
import torchvision
from torchvision.datasets import MNIST
#import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.optim as optim
from math import cos

f = open('C:/Programs/Disk_C/CodeBlocks_projects/for_tests/Formula.txt')
try:
    formula = f.read()
    print(formula)
finally:
    f.close()

# In[10]:


batch_size = 100
num_epochs = 10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
traindt = MNIST(root='data/', train=True, transform=transform, download=True)
testdt = MNIST(root='data', train=False, transform=transform, download=True)
train_loader = DataLoader(traindt, batch_size, shuffle=True)
test_loader = DataLoader(testdt, batch_size, shuffle=False)

# In[11]:


"""хорошая модель, пока что сделаю похуже, чтобы возможно нагляднее было улучшение или ухудшение
class Classifier(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout1 = nn.Dropout2d(0.25) 
        self.dropout2 = nn.Dropout2d(0.5) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.dropout1(x) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.dropout2(x) 
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x"""


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, 28 * 28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a


model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


# In[12]:


def adjust_learning_rate(optimizer, epoch, num_epochs, formula):
    """initial_lr = 0.01
    fin_lr = 0.001
    step = (initial_lr-fin_lr)/num_epochs"""
    e = epoch
    lr = eval(formula)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[15]:


losses = []
accuracies = []
test_losses = []
test_accuracies = []
# Train the model
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, num_epochs, formula)
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
    acc = (predicted == labels).sum().item() / labels.size(0)
    accuracies.append(acc)
    losses.append(loss.item())

    # Evaluate the model on the validation set
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        test_acc += correct / total
        test_accuracies.append(acc)
        test_losses.append(loss.item())
    current_lr = optimizer.param_groups[0]['lr']

    print('Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Test Loss:{:.4f}, Accuracy:{:.2f}, Test Accuracy:{:.2f}'.format(
        epoch + 1, num_epochs, current_lr, loss.item(), test_loss, acc, test_acc))

# In[18]:


f_write = open('C:/Programs/Disk_C/CodeBlocks_projects/for_tests/Accuracy.txt', 'w')
try:
    f_write.write(str(test_acc))
finally:
    f_write.close()
