import argparse
import os
import os.path as osp
import sys
import numpy as np
import time
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNet(Dataset):

    def __init__(self, root='./', train=True):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        
    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets
        
    def getAllDataAndPaths(self):
      return self.data, self.targets


def selectUpToKShotsPerIncrementalClass(data, targets, K_shots_split):
  
  selected_data = []
  selected_targets = []
  
  for index in range(40):
    indexOfIncrementalClass = index+60
    temp_data = []
    temp_targets = []
    for j in range(len(data)):
      if targets[j] == indexOfIncrementalClass:
        temp_data.append(data[j])
        temp_targets.append(targets[j])
    
    chosen_indices = np.random.choice(range(len(temp_data)), K_shots_split[index], replace=False)
    
    for k in chosen_indices:
      selected_data.append(temp_data[k])
      selected_targets.append(temp_targets[k])
    
  return selected_data, selected_targets


def main(dataroot, N_ways, K_shots, Num_increments, Num_classes, exp_dir):
  #dataroot = '/scratch/tahmad/FSCIL_datasets'
    
  trainset = MiniImageNet(root=dataroot, train=True)
  print('Total number of examples in dataset')
  print(trainset.__len__())
  
  data, targets = trainset.getAllDataAndPaths()
  
  #for k in range(trainset.__len__()):
  #  print(data[k], targets[k])
  
  N_sum = 0 
  while (N_sum != 40):
    N_ways_split = [np.random.randint(1,N_ways) for i in range(Num_increments)]
    N_sum = np.sum(N_ways_split)
  print('N_ways_split in incremental sessions')
  print(N_ways_split)
  print('Number of total incremental classes')
  print(np.sum(N_ways_split))
  
  classes_so_far = []
  classes_so_far.append(60)
  for j in range(len(N_ways_split)):
    classes_so_far.append(60+np.sum(N_ways_split[:j+1]))
  print('classes_so_far array')
  print(classes_so_far)
  fileName = exp_dir + 'class_dist_per_session.npy'
  np.save(fileName,classes_so_far)
  
  
  K_shots_split = [np.random.randint(1,K_shots) for i in range(Num_classes)]
  print('K_shots_split')
  print(K_shots_split)
  print('Number of total incremental shots')
  print(np.sum(K_shots_split))
  
  
  selected_data, selected_targets = selectUpToKShotsPerIncrementalClass(data, targets, K_shots_split)
  
  fileName = exp_dir + 'all_inc_sessions.txt'
  file1 = open(fileName, 'w')
  for k in range(len(selected_data)):
    line = selected_data[k] + ',' + str(selected_targets[k])
    file1.write(line)
    file1.write('\n')
    #print(selected_data[k], selected_targets[k])
  file1.close()
  
  
  for index in range(Num_increments):
    
    num_class_so_excluding_this_session = np.sum(N_ways_split[:index])
    num_class_so_including_this_session = np.sum(N_ways_split[:index+1])
    #print('incremental session: ', index, 'classes:', int(num_class_so_excluding_this_session), num_class_so_including_this_session)
    print('incremental session # ', index+1, ': classes up to this session:', num_class_so_including_this_session)
    
    
    if num_class_so_excluding_this_session == 0:
      start_line_from = 0
      end_line_to = np.sum(K_shots_split[:num_class_so_including_this_session]) 
    else:
      start_line_from = np.sum(K_shots_split[:num_class_so_excluding_this_session])
      end_line_to = np.sum(K_shots_split[:num_class_so_including_this_session])
    
    #print(index, 'lines', start_line_from, end_line_to)  
    print('incremental session # ', index+1, ': shots for these ', N_ways_split[index] ,'classes: ', K_shots_split[int(num_class_so_excluding_this_session):int(num_class_so_including_this_session)])
    
    if index+2 < 10:
      fileName = exp_dir + 'session_0' + str(index+2) + '.txt'
    else:
      fileName = exp_dir + 'session_' + str(index+2) + '.txt'
    
    file1 = open(fileName, 'w')
    
    for k in range(start_line_from, end_line_to):
      line = selected_data[k][51:]
      line = 'MINI-ImageNet/train/' + line[:9] + '/' + line
      file1.write(line)
      file1.write('\n')
    file1.close()
    
if __name__ == '__main__':
  # arguments that need to be changed according to experimental setting
  dataroot = '/scratch/tahmad/FSCIL_datasets'
  N_ways = 6 #11
  K_shots = 11 #6 #11 
  Num_increments = 14 #7 
  Num_classes = 40
  exp_dir = './experiments_mini_imagenet/experiment_21/'
  main(dataroot, N_ways, K_shots, Num_increments, Num_classes, exp_dir)
  
