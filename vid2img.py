import cv2
import os
import skvideo.io


def getFrames(video_name, folder_name, verbose = False):
    if os.path.isfile(video_name):
        videogen = skvideo.io.vreader(video_name)
        count = 0
        for frame in videogen:
             # save frame as JPEG file
            frame = cv2.resize(frame, (640,360))
            cv2.imwrite(os.path.join(folder_name, "frame%06d.jpg" % count), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
            count += 1
        return count, 1
    else:
        return 0, 0
#     vidcap = cv2.VideoCapture(video_name)
#     success,image = vidcap.read()
#     if success == False:
#         print('No video file found or no supported format!')
#         return 0, -1
#     '''
#     #Following operation is now added to movrs.sh
#     if(len(os.listdir(folder_name))) > 0:
#         if verbose == True:
#             print('Deleting existing frames in folder ....')
#         for one_file in [os.path.join(folder_name, x) for x in os.listdir('videoframes')]:
#             os.unlink(one_file)
#         if verbose == True:
#             print('Deleted!')
#     else:
#         if verbose == True:
#             print('no file remaining')
#     '''
#     count = 0
    
#     while success:
#         cv2.imwrite(os.path.join(folder_name, "frame%06d.jpg" % count), image)     # save frame as JPEG file      
#         success,image = vidcap.read()
#         if verbose == True:
#             print('Read a new frame: ', success)
#         count += 1
#     return count, 1

count, status = getFrames('video/input.mp4', 'videoframes')
print(count, status)

