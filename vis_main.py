import subprocess

p1 = subprocess.Popen(['./movrs.sh'])

p1.wait();

p2 = subprocess.Popen(['python', 'coco_pose_vis.py', '--img_folder', './videoframes/', '--json_path', './result.json'])

p2.wait();

p3 = subprocess.Popen(['python', 'img2vid.py', '--img_folder', './videoframes/results/'])

p3.wait()