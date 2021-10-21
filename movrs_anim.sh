#!/bin/bash

echo "Computing Pose"

conda activate pose

rm human_detection.json
rm result.json

rm -r ./videoframes
mkdir ./videoframes

# choose right conda enviornment for programs to run
python vid2img.py

cd models/research/object_detection
python person_detection.py

cd ~/inzamam

cp models/research/object_detection/valimage_dict.json TF-SimpleHumanPose/main
cp models/research/object_detection/human_detection.json TF-SimpleHumanPose/data/COCO/dets

if [ -d "TF-SimpleHumanPose/data/COCO/images/val2017" ]; then
  # Control will enter here if $DIRECTORY exists.
  rm -r TF-SimpleHumanPose/data/COCO/images/val2017
fi

cp -r videoframes/ TF-SimpleHumanPose/data/COCO/images

mv TF-SimpleHumanPose/data/COCO/images/videoframes TF-SimpleHumanPose/data/COCO/images/val2017

cd TF-SimpleHumanPose/main
python test-Copy1.py --gpu 0 --test_epoch 140

cd ~/inzamam

cp models/research/object_detection/human_detection.json .
cp TF-SimpleHumanPose/output/result/COCO/result.json .

conda activate chainer

cd 3dpose_gan

rm ./csv_out/*
rm ./estimated_animation.bvh

python bin/pose_to_csv.py ../result.json --lift_model gen_epoch_500.npz --model2d pose_iter_440000.caffemodel --proto2d openpose_pose_coco.prototxt --mode json --img_folder ../videoframes

blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py

blender --background -noaudio -P bvh_to_fbx.py
echo "Finished computing pose, results are save in human_detection.json & result.json"
