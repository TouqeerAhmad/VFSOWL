import argparse
import os
import sys
import numpy as np
import time
import torch
from PIL import Image
from torch.utils.data import Dataset

class CUB200(Dataset):

    def __init__(self, root='./', train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)
    
    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)
    
    def getAllDataAndPaths(self):
      return self.data, self.targets
    
    def __len__(self):
      return len(self.data)
    



def selectUpToKShotsPerIncrementalClass(data, targets, K_shots_split):
  
  selected_data = []
  selected_targets = []
  
  for index in range(100):
    indexOfIncrementalClass = index+100
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
  
  #dataroot = '/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200'
    
  trainset = CUB200(root=dataroot, train=True)
  print('Total number of examples in dataset')
  print(trainset.__len__())
  
  data, targets = trainset.getAllDataAndPaths()
  
  #for k in range(trainset.__len__()):
  #  print(data[k], targets[k])
  
  
  N_sum = 0 
  while (N_sum != 100):
    N_ways_split = [np.random.randint(1,N_ways) for i in range(Num_increments)]
    N_sum = np.sum(N_ways_split)
  print('N_ways_split in incremental sessions')
  print(N_ways_split)
  print('Number of total incremental classes')
  print(np.sum(N_ways_split))
  
  classes_so_far = []
  classes_so_far.append(100)
  for j in range(len(N_ways_split)):
    classes_so_far.append(100+np.sum(N_ways_split[:j+1]))
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
      #line = selected_data[k] + ',' + str(selected_targets[k])
      line = selected_data[k][39:]
      file1.write(line)
      file1.write('\n')
    file1.close()

if __name__ == '__main__':
  # arguments that need to be changed according to experimental setting
  dataroot = '/scratch/tahmad/FSCIL_datasets/CUB_200'
  N_ways = 6 #11
  K_shots = 11 #11
  Num_increments = 30 #15
  Num_classes = 100
  exp_dir = './experiments_cub200/experiment_21/'
  main(dataroot, N_ways, K_shots, Num_increments, Num_classes, exp_dir)
  
