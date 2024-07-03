import os
import csv
# Define the source and destination directories
source_dir = '../data'  # Assuming the script runs in the directory containing the folders
csv_file = "/home/edabk/Sleeping_pos/skeleton-based-code/pose_classifier/preprocess/gt_keypoint/gt_reverse.csv"
def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]
def extract_subject_frame_idx(file_name):
    subj_idx=str(file_name).split(".")[0].split("_")[2]
    frame_idx=str(file_name).split(".")[0].split("_")[1]
    return subj_idx,frame_idx
modals_lst=[]
# Iterate over the folders and merge .png files
for folder in os.listdir(source_dir):
    if folder.startswith('IR_9class_5_FOLD_'): # uncover,cover2,cover1
        modality=folder.split("_")[-1]
        print(modality)
        modal_classi_lst=[]
        for fold in range(1, 6):  # FOLD_1 to FOLD_5
            for set_type in ['train', 'val']:
                for class_num in range(1, 10):  # Class 1 to 9
                    source_path = os.path.join(source_dir,folder, f'FOLD_{fold}', set_type, str(class_num))
                    if os.path.exists(source_path):
                        for file_name in os.listdir(source_path):
                            modal_classi_lst.append((extract_subject_frame_idx(file_name)[0],extract_subject_frame_idx(file_name)[1],str(class_num-1)))
        omitdup_modal_classi_lst=removeDuplicates(modal_classi_lst)
        modals_lst.append(omitdup_modal_classi_lst)
for lst in modals_lst:
    print(len(lst))
final_lst=[]
for i,v in enumerate(modals_lst[0]):
    if ((v in modals_lst[1]) and (v in modals_lst[2])):
        final_lst.append(v)
final_lst=removeDuplicates(final_lst)
# print(len(final_lst))
print(final_lst[0:9])




from data.SLP_RD import SLP_RD
import opt
# opts outside?
opts = opt.parseArgs()
opts = opt.aug_opts(opts)
SLP_rd_all = SLP_RD(opts, phase='all')
print(len(SLP_rd_all.li_joints_gt_IR))
print(len(SLP_rd_all.li_joints_gt_IR[0]))
print(len(SLP_rd_all.li_joints_gt_IR[0][0]))
print(len(SLP_rd_all.li_joints_gt_IR[0][0][0]))

with open(csv_file,"w",newline='') as csvf:
    writer=csv.writer(csvf)
    for v in final_lst:
        subj_idx,frame_idx,classi=v
        data=SLP_rd_all.li_joints_gt_IR[int(subj_idx)-1][int(frame_idx)-1]
        data_row=[]
        for node_idx in range(14):
            # data_row.append(data[13-node_idx][0])
            # data_row.append(data[13-node_idx][1])
            data_row.append(data[node_idx][0])
            data_row.append(data[node_idx][1])
        data_row.append(classi)
        writer.writerow(data_row)

    

