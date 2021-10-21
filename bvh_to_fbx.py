import bpy

objs = bpy.data.objects

for obj in objs:
        if obj.name == 'Cube':
                obj.select = True

bpy.ops.object.delete()

bpy.ops.import_anim.bvh(filepath='./hmr/output/bvh_animation/estimated_animation.bvh')

bpy.ops.export_scene.fbx(filepath='./videoframes/results/output_anim_hmr.fbx')
