import torch
import torchvision
from torchvision.datasets import MNIST
#import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import numexpr as ne Не поддерживает log с двумя аргументами
import sys
from tqdm import tqdm
import time


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

def modified_lrs(lrs, ignore):
    n = len(lrs)
    num_epochs = len(lrs[0])
    if ignore == True:
        for i in range(n):
            for j in range(num_epochs):
                if lrs[i][j] == -50000:
                    for k in range(num_epochs):
                        lrs[i][k] = -50000
                    break
    else:
        for i in range(n):
            for j in range(num_epochs):
                if lrs[i][j] == -50000:
                    if j != 0:
                        lrs[i][j] = lrs[i][j-1]
                    else:
                        for k in range(1, num_epochs):
                            if lrs[i][k] != -50000:
                                lrs[i][j] = lrs[i][k]
                                break


def main():
    start = time.time()
    ignore = False # если True, то игнорируется целый индивид, если False, то вместо непосчитанного lr подставляется предыдущий или последующий
    batch_size = 100
    max_int = sys.maxsize
    # min_int = -sys.maxsize-1
    print('start Python')
    f = open('C:/Programs/Disk_C/CodeBlocks_projects/gp_with_neural_network/Lrs.txt', 'r')
    #f = open('C:/Programs/Disk_C/My projects/genetic_programming_with_nn/genetic_programming_with_nn/Lrs.txt', 'r')
    try:
        lines = f.readlines()
        n_individs = len(lines)
        num_epochs = lines[0].count('\t')
        lrs = np.zeros((n_individs, num_epochs))
        for i,line in enumerate(lines):
            line = line.rstrip()
            line = line.split('\t')
            line = list(map(float, line))
            lrs[i] = np.array(line)
        print(lrs)

    finally:
        f.close()

    time_prepar = time.time()
    num_workers = 0
    modified_lrs(lrs, ignore)
    transform = transforms.Compose([transforms.ToTensor(), ]) # transforms.ToTensor() автоматически нормализует данные в случае картинок
    traindt = MNIST(root='data/', train=True, transform=transform, download=True)
    testdt = MNIST(root='data/', train=False, transform=transform, download=True)
    train_loader = DataLoader(traindt, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testdt, batch_size, shuffle=False, num_workers=num_workers)
    results = np.zeros(n_individs)
    print('Time_data_preparation: {:.4f}'.format(time.time()-time_prepar))
    for k in range(n_individs):
        time_individ = time.time()
        res = False
        model = Classifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        model.train() # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
        for epoch in range(1, num_epochs+1):
            time_epoch = time.time()
            #train_tqdm = tqdm(train_loader, leave=True)
            lr = lrs[k][epoch-1]
            if lr == -50000.0:
                results[k] = max_int
                res = True
                break
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            running_loss = 0
            for images, labels in train_loader: # цикл по батчам
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #train_tqdm.set_description('Tree {}: Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}'.format(
                    #k, epoch, num_epochs, lr, running_loss))

            running_loss /= len(train_loader)
            print('Tree {}: Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}'.format(
                    k, epoch, num_epochs, lr, running_loss, time.time()-time_epoch))




        if (res == False):
            test_loss = 0.0
            correct = 0.0
            model.eval() # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1) # получаем индекс
                    correct += torch.sum(predicted == labels).item()

            test_loss /= len(test_loader) #делим на количество батчей
            test_acc = correct / len(testdt) #делим на количество наблюдений в тестовой выборке
            results[k] = round(test_loss, 8)
            print('Tree {}: TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}'.format(k, test_loss, test_acc, time.time()-time_individ))
        else: print('Tree {} is not valid'.format(k))



    print("Открытие файла для записи losses")
    f_write = open('C:/Programs/Disk_C/CodeBlocks_projects/gp_with_neural_network/Losses.txt', 'w')
    #f_write = open('C:/Programs/Disk_C/My projects/genetic_programming_with_nn/genetic_programming_with_nn/Losses.txt', 'w')
    try:
        for i in range(n_individs):
            f_write.write(f"{results[i]:.8f}\n")
    finally:
        f_write.close()
    print("Закрытие файла для записи losses")
    print("Time_whole_program: {:.4f} seconds".format(time.time() - start))  # 76 85 почему-то, надо попробовть когда процессор будет не так занят

if __name__ == '__main__':
    main()