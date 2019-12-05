train:

python main.py multi_pose --exp_id test_on_726_hr --dataset coco_hp --arch hourglass --batch_size 22 --master_batch 2 --lr 2.5e-4 --gpus 0,1,2,3 --num_workers 2 --Train_file_path home_1_home_2_home_3_real_v0_real_v1_real_v2_output.json --num_epochs 140 --val_intervals 1 --resume



ctdet --exp_id hg_Virtual_4 --dataset coco --arch hourglass --batch_size 22 --master_batch 2 --lr 2.5e-4 --gpus 0,1,2,3 --num_workers 2 --Train_file_path home_1_home_2_home_3_real_v0_real_v1_real_v2_real_v3_human_label_kobeF2_Virtual_V4_output.json --num_epochs 140 --val_intervals 1 --resume