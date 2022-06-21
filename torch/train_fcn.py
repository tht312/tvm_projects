# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MyDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None, target_transform=None):
        self.imglist = [os.path.join(img_path,item) for item in os.listdir(img_path)]
        self.labellist = [os.path.join(label_path,item) for item in os.listdir(label_path)]
        self.imglist.sort()
        self.labellist.sort()
        self.transform = ToTensor()
        self.target_transform = None
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image, label = Image.open(self.imglist[idx]), Image.open(self.labellist[idx])
        image, label = self.transform(image)*255, self.transform(label)*255
        return image, label

img_path = '../dataset/cut_img'
label_path = '../dataset/cut_label'
training_data = MyDataset(img_path, label_path)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_features[0])
# print(train_labels[0])

import torch
from torch.autograd import Variable 
from fcn_for_train import FCN32s
model = FCN32s(n_class=6)
# checkpoint = torch.load('torch_model/fcn32s_49.pth')
# model.load_state_dict(checkpoint)
model.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for epoch in range(100):
    total_loss = 0
    for i, (img, label) in enumerate(train_dataloader):
        img, label = Variable(img).to(device), Variable(label.long().squeeze(0)).to(device)
        optimizer.zero_grad() # 梯度清0
        output = model(img)
        calLoss = torch.nn.CrossEntropyLoss().to(device)
        loss = calLoss(output,label)
        loss.backward() # 误差反向传播
        optimizer.step() # 用优化器去更新权重参数
        total_loss = total_loss + loss
        
        # if epoch < 5:
        #    print(i,loss,flush=True)
    print("epoch:",epoch,"total loss:",total_loss,flush=True)
    torch.save(model.state_dict(), 'torch_model/fcn32s_{}.pth'.format(epoch))

