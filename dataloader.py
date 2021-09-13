import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io
import torch
def addLayer(img, dirct):
    leftorright = dirct[17]
    additionLayer = np.ones((1,512,512))
    #additionLayer = torch.from_numpy(np.ones((1,512,512)))
    #img = torch .cat([img, additionLayer],0)
    #img = np.concatenate((img, additionLayer), axis=0)
    dis = int(dirct[18:20])/10.0
    #print(dirct[18:20])
    if leftorright =='0':
        img = np.concatenate((img, additionLayer*dis), axis=0)

    elif leftorright =='l':


        img = np.concatenate((img, additionLayer*dis), axis=0)


    elif leftorright =='r':

        img = np.concatenate((img, additionLayer*(-1*dis)), axis=0)
        


    return img



class DepthDatasetLoader(Dataset):
    """"For loading RGB images and corresponding depth"""
    
    def __init__(self, dataset_directory):
        
        self.dataset_directory = dataset_directory
        self.rgb_images = sorted(glob.glob(self.dataset_directory + '/**/*'+'.png', recursive=True))
        #print(self.rgb_images)
        #self.depth_images = sorted(glob.glob(self.dataset_directory +"Depth" + '/**/*'+".png", recursive=True))
        
    def __len__(self):
        return 600
    
    def __getitem__(self, idx):
        rgb_img_file = self.rgb_images[idx]
        #depth_img_file = self.depth_images[idx]
        #print(rgb_img_file[20::])
        rgb_img = cv2.imread(rgb_img_file)[...,::-1] # bgr to rgb. Since opencv loads images as bgr
        #depth_img = cv2.imread(depth_img_file, 0)    # Load as grayscale 
        #depth_img = np.expand_dims(depth_img, 2)     # Add an extra channel dimension. Converts 
                                                     # (height, width) to (height, width, channel)
        org_img_file = 'CameraGTRGB000'+rgb_img_file[20::]
        org_img = cv2.imread(org_img_file)[...,::-1]
        rgb_img = np.transpose(rgb_img, (2,0,1))     # Since Pytorch models take tensors 
        org_img = np.transpose(org_img, (2,0,1)) 
        #print(rgb_img.type())
        #print(addLayer(rgb_img,dirct=rgb_img_file))
        #rgb_img = addLayer(rgb_img,dirct=rgb_img_file)
        org_img = addLayer(org_img,dirct=rgb_img_file)
        #depth_img = np.transpose(depth_img, (2,0,1)) # in (channel, width, height) format
        data = dict()
        data['rgb'] = rgb_img.copy()
        data['org'] = org_img.copy()
        #data['depth'] = depth_img.copy()
        data['filenames'] = rgb_img_file
        return data

# dd_object = DepthDatasetLoader(dataset_directory = "dataset/")
# print(dd_object[10].keys())
# print(dd_object[10]['rgb'].shape)
# print(dd_object[10]['filenames'])

# mydataloader = DataLoader(dd_object, batch_size=7, shuffle=True, num_workers=1, drop_last=True)

# for sub_iteration, minibatch in enumerate(mydataloader):
#     print(sub_iteration)
#     print(minibatch['rgb'].shape, minibatch['filenames'])