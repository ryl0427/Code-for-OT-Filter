from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

# CIFAR Dataset
# train/test/labeled/unlabeled
class cifar_dataset(Dataset): 
    def __init__(self, dataset, transform, mode, root_dir, selected, noisy_label):
        self.transform = transform
        self.mode = mode
        self.selected = selected
                
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            self.train_data = train_data
            self.train_label = train_label
            if noisy_label:
                self.train_label = noisy_label
                        
    def __getitem__(self, index):                 
        if self.mode=='train':
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img) 
            return img, target, index
         
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        
        elif self.mode=='labeled':
            img, target = self.train_data[int(self.selected[index])], self.train_label[int(self.selected[index])]
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            return img1, img2, target
        
        elif self.mode=='unlabeled':
            img = self.train_data[int(self.selected[index])]
            img = Image.fromarray(img)
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            img4 = self.transform[3](img)
            return img1, img2, img3, img4
           
    def __len__(self):
        if self.mode=='train':
            return len(self.train_data)
        elif self.mode=='test':
            return len(self.test_data)
        else:
            return len(self.selected)

# generate symmetric/asymmetric noisy
def symmetrical_noisy(label, noisy_rate, num_class):
    noisy_label = []
    for i in range(len(label)):
        a=random.uniform(0,1)
        if a<noisy_rate:
            noisy_label.append(random.randint(0,num_class-1))
        else:
            noisy_label.append(label[i])
    return noisy_label

def asymmetrical_cifar10(label, noisy_rate):
    noisy_label = []
    for i in range(len(label)):
        a=random.uniform(0,1)
        
        if label[i]==2 and a<noisy_rate:
            noisy_label.append(0)
        
        elif label[i]==4 and a<noisy_rate:
            noisy_label.append(7)

        elif label[i]==9 and a<noisy_rate:
            noisy_label.append(1)
            
        elif label[i]==3 and a<noisy_rate:
            noisy_label.append(5)

        elif label[i]==5 and a<noisy_rate:
            noisy_label.append(3)
        
        else:
            noisy_label.append(label[i])
    
    return noisy_label

def asymmetrical_cifar100(label, noisy_rate):
    noisy_label = []
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    for i in range(len(label)):
        a = random.uniform(0,1)
        if a<noisy_rate:
            change = np.where(coarse_labels==coarse_labels[label[i]])
            changed_label = change[0][random.randint(0,4)]
            noisy_label.append(changed_label)
        else:
            noisy_label.append(label[i])
    
    return noisy_label
