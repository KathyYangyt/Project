from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from allocate import Cmnist_iid, Cmnist_noniid,Cmnist_noniid_unequal

def GetDataset(args):
       
       trans_chinese = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
       
       path_dir="../Project/Chinese_digit/data/data"

       class MyDataset(Dataset): 
          def __init__(self, path_dir, transform=None): 
              self.path_dir = path_dir  
              self.transform = transform  
              self.images = os.listdir(self.path_dir)  
 
          def __len__(self): 
              return len(self.images)
 
          def __getitem__(self, index):  
              image_index = self.images[index]  
              image_path = os.path.join(self.path_dir, image_index)  
              image = Image.open(image_path)  
            
              label = int(image_path.split('/')[-1].split('.')[0].split('_')[-1]) - 1
 
              image = self.transform(image)
              return image, label

       dataset_train_all = MyDataset(path_dir, transform=trans_chinese)
       dataset_train,dataset_test= torch.utils.data.random_split(dataset_train_all,[12000,3000])
       if args.iid == 'iid':
          group_clients = Cmnist_iid(dataset_train, args.num_clients)
       elif args.iid == 'non_iid_equal':
          group_clients = Cmnist_noniid(dataset_train, args.num_clients)
       elif args.iid == 'non_iid_unequal':
          group_clients = Cmnist_noniid_unequal(dataset_train, args.num_clients)
       else:
          exit('Error: unrecognized type') 

       return dataset_train , dataset_test , group_clients
      