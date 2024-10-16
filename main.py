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
from math import cos, exp, sin, log
import numpy as np
#import numexpr as ne Не поддерживает log с двумя аргументами
import sys


"""def check_formula(formula):
    notfin = True
    expr = formula
    while(notfin):
        if 'log' in expr:
            ind = expr.find('log')
            arg = expr[ind+4:]
            print(arg)
            ind2 = arg.find('(')
            if (ind2 != -1):"""



"""def adjust_learning_rate(optimizer, epoch, num_epochs, formula):
    initial_lr = 0.01
    fin_lr = 0.001
    step = (initial_lr-fin_lr)/num_epochs

    x00 = epoch
    lr = eval(formula)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr"""

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


batch_size = 100
num_epochs = 3


print('start Python')
f = open('C:/Programs/Disk_C/CodeBlocks_projects/gp_with_neural_network/Lrs.txt', 'r')
try:
    lines = f.readlines()
    n_individs = len(lines)
    lrs = np.zeros((n_individs, num_epochs))
    for i,line in enumerate(lines):
        line = line.rstrip()
        line = line.split('\t')
        line = list(map(float, line))
        lrs[i] = np.array(line)
    print(lrs)

finally:
    f.close()

# In[10]:
"""for k in range(n_individs):
    check_formula(formulas[k])"""


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



model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

#max_int = sys.maxsize
#min_int = -sys.maxsize-1


results = np.zeros(n_individs)
for k in range(n_individs):
    # Train the model
    for epoch in range(1, num_epochs+1):
        running_loss = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[k][epoch-1]
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        #accuracies.append(acc)
        #losses.append(loss.item())
        running_loss /= len(train_loader)
        # Evaluate the model on the validation set
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        num_batches = len(test_loader)
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss /= num_batches
        test_acc = correct / total
            #test_accuracies.append(acc)
            #test_losses.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']

        print('Tree {}: Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Test Loss:{:.4f}, Test Accuracy:{:.2f}'.format(
        k, epoch, num_epochs, current_lr, running_loss, test_loss, test_acc))
        results[k] = round(test_loss, 4)
# In[18]:


f_write = open('C:/Programs/Disk_C/CodeBlocks_projects/gp_with_neural_network/Losses.txt', 'w')
try:
    for i in range(n_individs):
        f_write.write(str(results[i])+'\n')
finally:
    f_write.close()
