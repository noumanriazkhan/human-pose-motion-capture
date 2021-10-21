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
# python post_process.py

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

cd ./videoframes/results/

ffmpeg -i ../../video/input.mp4 -i output_blank.avi -filter_complex hstack=inputs=2 output_comb.avi

cd ~/inzamam

cd 3dpose_gan

rm ./csv_out/*
rm ./estimated_animation.bvh
rm ./output_anim.fbx

python bin/pose_to_csv.py ../result.json --lift_model gen_epoch_500.npz --model2d pose_iter_440000.caffemodel --proto2d openpose_pose_coco.prototxt --mode json --img_folder ../videoframes

blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py

cp ./csv_out/csv_joined.csv ../videoframes/results/

blender --background -noaudio -P bvh_to_fbx.py

cd ~/inzamam
mkdir ./videoframes/hmr
python crop_frames_for_hmr.py

cd ./hmr
rm output/bvh_animation/estimated_animation.bvh 
rm output/csv/*
rm output/csv_joined/*
python ../run_hmr_demo.py
cd ../
blender --background hmr/csv_to_bvh.blend -noaudio -P hmr/csv_to_bvh.py
blender --background -noaudio -P bvh_to_fbx.py
echo "Finished computing pose, results are save in human_detection.json & result.json"
