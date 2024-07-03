import os
import io
import numpy as np
import torch
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
from utils.utils_ds import *
import matplotlib.pyplot as plt
import utils.vis as vis
import utils.utils as ut


output_dir="/home/edabk/Sleeping_pos/data/output_keypoint"
sv_dir=os.path.join(output_dir,"image_vis")
if not os.path.exists(sv_dir): os.makedirs(sv_dir)
root_dir = "/home/edabk/Sleeping_pos/data/IR_9class_merge_raw/train"
csv_file_name = os.path.join(output_dir,"batch_keypoint_path.csv")
opts_model="HRpose"
exec('from model.{} import get_pose_net'.format(opts_model)) 
opts_sz_pch=(256, 256) 
opts_out_shp=[64, 64, -1]
n_jt=14 #num_node
opts_input_nc=1 #num_channel
li_mean=[0.1924838]
li_std=[0.077975444]
bb=np.array([-20.,   0., 160., 160.])
scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
batch_size=64
model = get_pose_net(in_ch=opts_input_nc, out_ch=n_jt) # lay input la batch,1,256,256 tuc la mot cai anh
                                                       # cho output la batch,14,64,64
#checkpoint_file = "/home/edabk/new_SLP/gcn_classification_SLP/SLP-Dataset-and-Code/output_old/SLP_IR_u12_HRpose_ts1/model_dump/model_best.pth"
checkpoint_file="/home/edabk/new_SLP/gcn_classification_SLP/pretrained_HRpose_models/checkpoint_model_best.pth"
print("=> loading checkpoint '{}'".format(checkpoint_file))
checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['state_dict'])  # here should be cuda setting
print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))



# Define your custom dataset class
class IRDataset(Dataset):
    def __init__(self, root_dir, csv_file_name, transform=None,classes=[0,1,2,3,4,5,6,7,8]):
        self.root_dir = root_dir
        self.csv_file_name = csv_file_name
        self.transform = transform
        self.data = []  # List to store image paths and corresponding class labels
        self.classes = classes
        # Populate self.data with image paths and class labels
        for class_i in self.classes:
            class_dir = os.path.join(self.root_dir, str(class_i+1))
            for filename in os.listdir(class_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(class_dir, filename)
                    self.data.append((img_path, class_i))  # Tuple: (image path, class label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_label = self.data[idx]
        img = self.get_img(img_path)
        #print(img.shape)
        img=np.mean(img,axis=2)
        img=np.expand_dims(img,axis=2)
        img_patch, trans = generate_patch_image(img, bb, do_flip, scale, rot, do_occlusion, input_shape=opts_sz_pch[::-1])
        if img_patch.ndim<3:
            img_channels = 1 # add one channel
            img_patch = img_patch[..., None]
        else:
            img_channels = img_patch.shape[2]   # the channels
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
        trans_tch = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=li_mean, std=li_std)])
        pch_tch = trans_tch(img_patch) #1,256,256
        #inputs = np.expand_dims(pch_tch,axis=0)
        return pch_tch, class_label,img_path

    def get_img(self, img_pth):
        read_func = io.imread
        img = read_func(img_pth)  # Should be a 2D array
        img = np.array(img)
        return img

# Example usage

custom_transforms = transforms.Compose([transforms.ToTensor()])  # Add any other transformations you need

IR_dataset = IRDataset(root_dir, csv_file_name, transform=custom_transforms)
dataloader = DataLoader(IR_dataset, batch_size=batch_size, shuffle=False)


with open(csv_file_name, mode='a', newline='') as file_:
    writer = csv.writer(file_)
# Now you can iterate over batches of images and class labels using the dataloader
    iter=0
    for batch_img, batch_class,batch_img_path in dataloader:
        # Your model inference or training code here
        iter+=1
        #img_patch, trans = generate_patch_image(img_cb, bb, do_flip, scale, rot, do_occlusion, input_shape=opts_sz_pch[::-1])
        outputs = model(batch_img)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs
        output=output.detach().numpy()
        pred, _ = get_max_preds(output)
        pred_hm=pred
        pred_hm=pred_hm/opts_out_shp[0] * opts_sz_pch[1] #normalize
        if iter%1==0:
        # This is for visualization 1 sample later on
            for vs_index in range(len(batch_img)):
                #vs_index=np.random.randint(0,len(batch_img)) # which sample for visualization
                pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis  # shape 14x3  
                pred2d_patch[:,:2] = pred_hm[vs_index] # only first   # shape 14x3
                x_coords = pred2d_patch[:, 0]
                y_coords = pred2d_patch[:, 1]
                mod0 = "IR"
                mean = [0.1924838]
                std = [0.077975444]
                img_patch_vis = ut.ts2cv2(batch_img[vs_index], mean, std) # to CV BGR
                    # pseudo change
                cm = 11
                img_patch_vis = cv2.applyColorMap(img_patch_vis, cm) # shape 256x256x3

                # original version get img from the ds_rd , different size , plot ing will vary from each other
                # warp preds to ori
                # draw and save  with index.
                idx_test = batch_img_path[vs_index].split("/")[-1] # image index
                skels_idx = ((12, 13), (12, 8), (8, 7), (7, 6), (12, 9), (9, 10), (10, 11), (2, 1), (1, 0), (3, 4), (4, 5))
                # get pred2d_patch
                vis.save_2d_skels_242_simple(img_patch_vis, pred2d_patch, skels_idx, os.path.join(sv_dir,str(int(batch_class[vs_index]))), suffix='-'+mod0,
                                    idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
                # write to csv part
                data_row=[]
                x_coords=pred_hm[vs_index,:,0]
                y_coords=pred_hm[vs_index,:,1]
                for index in range(0,14):
                    data_row.append(x_coords[13-index])
                    data_row.append(y_coords[13-index])
                data_row.append(int(batch_class[vs_index]))
                data_row.append(batch_img_path[vs_index])
                writer.writerow(data_row)
            print("iter: ",iter)
            print("Batch shape",batch_img.shape)
            print("Batch size:", batch_img.shape[0])
            print("Class labels:", batch_class)
