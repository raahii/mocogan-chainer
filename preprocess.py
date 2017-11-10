"""
Perform following preprocessing for MUG Facial Dataset.
 1. discard videos of neutral facial expression
 2. extract subsequences of video clips
 3. crop the face region
 4. resize 64 x 64

To use this script, pass the arguments
MUG dataset directory and save directory.

I suppose that dataset path is 'subject3' directory
in MUG dataset

<dataset path (subject3)> 
     |
     |--- 001
     |     |
     |     |--- anger
     |     |--- disgust
     |            :
     |
     |--- 002
           :
"""

import sys, os, glob
import cv2

CASCADE_PATH = "data/cv_models/haarcascade_frontalface_default.xml"

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_face(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(96, 96))

    # # detected rectangle
    # grid_color = (255, 0, 0)
    # for rect in facerect[0]:
    #     cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), grid_color, thickness=2)
    # show_img(image)
    rect = facerect[0] if len(facerect) > 0 else None

    return rect

def main(dataset_path, output_path):
    # list all video clips
    facial_expressions = ["anger", "disgust", "happiness",
                          "fear", "sadness", "surprise"]
    video_paths = []
    for exp in facial_expressions:
        takes = glob.glob(os.path.join(dataset_path,'*',exp,'*'))
        good_takes = list(filter(lambda name: name.find('nsg') < 0, takes))
        video_paths.extend(good_takes)

    print(">>> {} video clips found.".format(len(video_paths)))
    
    edge   = 0.20 # dont use 20% of the end of the video
    speeds = [1, 2] # use 1 frame per speed frames ( to change speed )
    length = 16 # video length (frame num)
    stride = length // 2 # stride width

    # perform preprocess
    saved_num = 0
    os.makedirs(output_path, exist_ok=True)
    for i, in_dir in enumerate(video_paths):
        images = glob.glob(os.path.join(in_dir, '*.jpg'))

        edge_frames = int(len(images) * edge)
        start = edge_frames
        end   = len(images) - edge_frames
        for speed in speeds:
            frame_width = speed*length
            for seq_n, j in enumerate(range(start, end-frame_width, stride)[0:-1]):
                out_dir = os.path.join(output_path, "{:04d}_{:d}_{:02d}".format(i+1, speed, seq_n+1))
                os.makedirs(out_dir, exist_ok=True)
            
                # detect face region of the video clip 
                mid_frame = j + frame_width//2
                image = cv2.imread(images[mid_frame])
                rect = detect_face(image)
                if rect is None: continue

                for k in range(length):
                    # read img
                    image = cv2.imread(images[j+speed*k])

                    # crop face part
                    x, y = rect[0], rect[1]
                    w, h = rect[2], rect[3]
                    image = image[y:y+h, x:x+w]

                    # resize 64, 64
                    image = cv2.resize(image,(64, 64))

                    # save
                    image_name = "{:02d}.jpg".format(k+1)
                    cv2.imwrite(os.path.join(out_dir, image_name) , image)

                saved_num += 1
                print("{}({} frames) --> {}(offset:{}, speed:{})".format(
                                                        '/'.join(in_dir.split('/')[-3:]),
                                                        len(images),
                                                        '/'.join(out_dir.split('/')[-2:]),
                                                        j, speed))
        print("")

    print(">>> preprocess finished and {} video clips saved".format(saved_num))

if __name__=="__main__":
    if len(sys.argv) < 3:
        print("usage: python preprocess.py <dataset path> <output path>")
        sys.exit()

    main(sys.argv[1], sys.argv[2])
