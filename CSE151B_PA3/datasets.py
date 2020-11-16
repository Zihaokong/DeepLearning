import torch
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from torchvision import transforms
import numpy as np
# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        #the direct path name storing picture
        self.pic_dir = os.path.join(os.getcwd(),root,"images")
        
        #open txt file and read files
        f_train = open(root+'/'+file_path,'r')
        self.lines = f_train.readlines()
        
        #place holder for dataset labels
        self.labels = []
        
        #store each image's absolute path for loading, and store their labels as well
        for i in range(len(self.lines)):
            line = self.lines[i].split()
            self.lines[i] = os.path.join(self.pic_dir,line[0])
            
            self.labels.append(int(line[1]))
        
        #throw error when number of image is not the same as number of labels
        if len(self.lines) != len(self.labels):
            raise Exception("y size aren't match x size")
        shuffle = np.random.permutation(len(self.lines))


    def __len__(self):
        #return the length of dataset
        return len(self.lines)

    
    def __getitem__(self, item):
        # turn PIL to tensor, and normalize use ImageNet mean and SD
        train_transforms = transforms.Compose([transforms.ToTensor(), \
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
        #read an image dynamically and center crop it, then transform it
        image = Image.open(self.lines[item]).convert("RGB")
        image = transforms.CenterCrop(224)(image)
        image = train_transforms(image)
        return image,self.labels[item]

        
            

            