#!/bin/bash

# Command 1: COVER2
python batch_infer_kp16_vis.py --output_dir /home/edabk/Sleeping_pos/SLP-Dataset-and-Code/output_infer_kp16/cover2 --csv_name cover2.csv --image_folder /home/edabk/Sleeping_pos/data/IR_9class_merge_raw_cover_modes/IR_9class_merge_raw_COVER2 --vis_name vis --vis_interval 10 --ckpt_path /home/edabk/Sleeping_pos/pretrained_HRpose_models/checkpoint_model_best.pth --batch_size 32 &&

# Command 2: COVER1
python batch_infer_kp16_vis.py --output_dir /home/edabk/Sleeping_pos/SLP-Dataset-and-Code/output_infer_kp16/cover1 --csv_name cover1.csv --image_folder /home/edabk/Sleeping_pos/data/IR_9class_merge_raw_cover_modes/IR_9class_merge_raw_COVER1 --vis_name vis --vis_interval 10 --ckpt_path /home/edabk/Sleeping_pos/pretrained_HRpose_models/checkpoint_model_best.pth --batch_size 32 &&

# Command 3: UNCOVER
python batch_infer_kp16_vis.py --output_dir /home/edabk/Sleeping_pos/SLP-Dataset-and-Code/output_infer_kp16/uncover --csv_name uncover.csv --image_folder /home/edabk/Sleeping_pos/data/IR_9class_merge_raw_cover_modes/IR_9class_merge_raw_UNCOVER --vis_name vis --vis_interval 10 --ckpt_path /home/edabk/Sleeping_pos/pretrained_HRpose_models/checkpoint_model_best.pth --batch_size 32 &&

# Command 4: MIX
python batch_infer_kp16_vis.py --output_dir /home/edabk/Sleeping_pos/SLP-Dataset-and-Code/output_infer_kp16/mixed --csv_name mixed.csv --image_folder /home/edabk/Sleeping_pos/data/IR_9class_merge_raw/train --vis_name vis --vis_interval 30 --ckpt_path /home/edabk/Sleeping_pos/pretrained_HRpose_models/checkpoint_model_best.pth --batch_size 32
