#!/bin/bash

echo "Computing Pose"

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

# run filter
python moving_avg_filter.py

python coco_pose_vis.py --img_folder ./videoframes/ --json_path ./filtered_result.json

python img2vid.py --img_folder ./videoframes/results/

echo "Finished computing pose, results are save in human_detection.json & result.json"
