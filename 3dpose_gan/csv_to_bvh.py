import csv
import os
import bpy

def sub(x):
    return x[0]-x[1]
objects = bpy.context.scene.objects

empties = []

for object in objects:
    if object.type == 'EMPTY':
        empties.append(object)
        
print(empties)

filename = 'csv_joined.csv'
directory = './csv_out/'

fullpath = os.path.join(directory, filename)

with open(fullpath, 'r', newline='') as csvfile:
    ofile = csv.reader(csvfile, delimiter=',')
    next(ofile) # <-- skip the x,y,z header
    for line in ofile:
        f, *pts = line
        # these things are still strings (that's how they get stored in the file)
        # here we recast them to integer and floats
        frame_num = int(f)
        print(frame_num)
        fpts = [float(p) for p in pts]
        coordinates = [fpts[0:3], fpts[3:6], fpts[6:9], fpts[9:12],
                       fpts[12:15], fpts[15:18], fpts[18:21], fpts[21:24],
                       fpts[24:27], fpts[27:30], fpts[30:33], fpts[33:36],
                       fpts[36:39], fpts[39:42],fpts[42:45], [fpts[42]+0.3,fpts[43]-0.3,fpts[44]+0.3], [fpts[42]-0.3,fpts[43]-0.3,fpts[44]+0.3], [fpts[42]+0.5,fpts[43]+0.3,fpts[44]+0.3], [fpts[42]-0.5,fpts[43]+0.3,fpts[44]+0.3], fpts[45:48]]
        bpy.context.scene.frame_set(frame_num)
        for ob, position in zip(empties, coordinates):
            #ob.location = [sub(x) for x in zip(position, [35.8460513,-22.5729021,-2.83607928])]
            ob.location = position
            ob.keyframe_insert(data_path="location", index=-1)

bpy.data.objects['rig'].select = True

target_file = './estimated_animation.bvh'

bpy.ops.export_anim.bvh(filepath=target_file, frame_start=0, frame_end=frame_num)
#bpy.ops.export_scene.fbx('./bvh_animation/estimated_scene.fbx')

