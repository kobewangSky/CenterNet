cd src
# train
python main.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 1,2 --num_epochs 50 --lr_step 140
# test
python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume
# flip test
python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..
