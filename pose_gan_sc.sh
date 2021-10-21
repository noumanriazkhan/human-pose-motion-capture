#!/bin/bash
echo "Starting ...    ..   ."
cd 3dpose_gan

rm ./csv_out/*
rm ./estimated_animation.bvh
rm ./output_anim.fbx

python bin/pose_to_csv.py ../result.json --lift_model gen_epoch_500.npz --model2d pose_iter_440000.caffemodel --proto2d openpose_pose_coco.prototxt --mode json

blender --background csv_to_bvh.blend -noaudio -P csv_to_bvh.py

cp ./csv_out/csv_joined.csv ../videoframes/results/

cd ~/inzamam

echo "Completed"
