from skimage import io
import os
import numpy as np
import torchvision.transforms as transforms
from utils.utils_ds import *
import matplotlib.pyplot as plt
import csv
csv_file_name="/home/edabk/Sleeping_pos/data/keypoint_extra_8_to_9.csv"
opts_model="HRpose"
root_dir="/home/edabk/Sleeping_pos/data/IR_9class_merge_raw/train"
exec('from model.{} import get_pose_net'.format(opts_model)) 
opts_sz_pch=(256, 256) 
opts_out_shp=[64, 64, -1]
n_jt=14 #num_node
opts_input_nc=1 #num_channel
li_mean=[0.1924838]
li_std=[0.077975444]
bb=np.array([-20.,   0., 160., 160.])

scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

def getImg(img_pth):
	readFunc = io.imread
	img = readFunc(img_pth)  # should be 2d array
	img = np.array(img)
	return img 
model = get_pose_net(in_ch=opts_input_nc, out_ch=n_jt) # lay input la batch,1,256,256 tuc la mot cai anh
                                                       # cho output la batch,14,64,64
checkpoint_file = "/home/edabk/new_SLP/gcn_classification_SLP/SLP-Dataset-and-Code/output_old/SLP_IR_u12_HRpose_ts1/model_dump/model_best.pth"
print("=> loading checkpoint '{}'".format(checkpoint_file))
checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['state_dict'])  # here should be cuda setting
print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))






with open(csv_file_name, mode='a', newline='') as file_:
    writer = csv.writer(file_)
    for class_i in range(8,10):
        print(os.listdir(os.path.join(root_dir,str(class_i))))
        for filename in os.listdir(os.path.join(root_dir,str(class_i))):
            if filename.endswith(".png"):
                li_img=[]
                img_path=os.path.join(root_dir,str(class_i),filename)
                print(img_path)
                img=getImg(img_path)
				#print("Shpe of img is",img.shape) #(160,120)
				#print(img)
                img = img[..., None]        # add one dim
                li_img.append(img) # len is always 1
                img_cb = np.concatenate(li_img, axis=-1) 
                img_cb=np.average(img_cb,axis=2)
                #print(img_cb.shape)
                img_patch, trans = generate_patch_image(img_cb, bb, do_flip, scale, rot, do_occlusion, input_shape=opts_sz_pch[::-1])
                if img_patch.ndim<3:
                    img_channels = 1        # add one channel
                    img_patch = img_patch[..., None]
                else:
                    img_channels = img_patch.shape[2]   # the channels
                for i in range(img_channels):
                    img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
                #print(img_patch.shape)
                trans_tch = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=li_mean, std=li_std)]
							)
                pch_tch = trans_tch(img_patch) #1,256,256
                print("Shape of pch_tch is: ",pch_tch.shape)
                inputs=np.expand_dims(pch_tch,axis=0)
                #print(inputs)
                inputs=torch.from_numpy(inputs)
                outputs = model(inputs)
                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs
					
                output=output.detach().numpy()
                pred, _ = get_max_preds(output)
                pred_hm=pred # 1,14,2
                print(pred_hm.shape)

                pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis  # shape 14x3  
                pred2d_patch[:,:2] = pred_hm[0] / opts_out_shp[0] * opts_sz_pch[1]  # only first   # shape 14x3
                x_coords = pred2d_patch[:, 0]
                y_coords = pred2d_patch[:, 1]
                data_row=[]
                for index in range(0,14):
                    data_row.append(x_coords[13-index])
                    data_row.append(y_coords[13-index])
                data_row.append(int(class_i)-1)
                #print("This is pred2d_patch: ",pred2d_patch)
                writer.writerow(data_row)



"""
data=pred2d_patch
x_coords = data[:,0]
y_coords = data[:, 1]
confidences = data[:, 2]

# Create a scatter plot
plt.scatter(x_coords, y_coords, c=confidences, cmap='viridis', label='Data Points')

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot of Data Points with Confidence')

# Show the colorbar for confidence values
cbar = plt.colorbar()
cbar.set_label('Confidence')
plt.savefig('infer.png')
# Display the plot
plt.show()
"""
