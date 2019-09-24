train:

python main.py multi_pose --exp_id hg_Virtual__3x --dataset coco_hp --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3 --num_workers 4 --Train_file_path Virtual_V0_real_v2_real_v1_real_v0_home_3_home_2_home_1_output.json --num_epochs 140 --val_intervals 1