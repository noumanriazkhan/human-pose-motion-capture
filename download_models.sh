pip install gdown

gdown https://drive.google.com/uc?id=1naZPyA0s5nB862XzcR4ZL84vKX0TxXlA
unzip trained_models.zip
mv ./backup_large_files backup_models
cd backup_models

mv 256x192_resnet50_coco.zip ../TF-SimpleHumanPose/output/model_dump/
mv 256x192_resnet50_coco.zip ../TF-SimpleHumanPose/output/model_dump/COCO/
mv snapshot_140.ckpt.data-00000-of-00001 ../TF-SimpleHumanPose/output/model_dump/COCO/
mv 256x192_resnet50_coco.zip ../TF-SimpleHumanPose/data/imagenet_weights/
mv snapshot_140.ckpt.data-00000-of-00001 ../TF-SimpleHumanPose/data/imagenet_weights/
mv person_keypoints_train2017.json ../TF-SimpleHumanPose/data/COCO/annotations/
mv pose_hrnet_w32_256x192.pth ../
mv model.ckpt-667589.data-00000-of-00001 ../hmr/models/
mv pose_iter_440000.caffemodel ../3dpose_gan/
