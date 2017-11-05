"""
Perform following preprocessing for MUG Facial Dataset.
 1. discard videos of neutral facial expression 
 2. discard videos containing fewer than 64 frames
 3. crop the face region
 4. resize 96 x 96

To use this scriptm pass the arguments
MUG dataset directory and save directory.

I suppose that dataset path is 'subject' directory
in MUG dataqset

<dataset path> 
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
        video_paths.extend(takes)

    print(">>> {} video clips found.".format(len(video_paths)))
    
    # perform preprocess

    os.makedirs(output_path, exist_ok=True)
    for i, in_dir in enumerate(video_paths):
        images = glob.glob(os.path.join(in_dir, '*.jpg'))
        if len(images) < 64: continue
        
        # detect face region of the video
        image = cv2.imread(images[len(images)//2])
        rect = detect_face(image)

        out_dir = os.path.join(output_path, "{:04d}".format(i+1))
        for image_path in images:
            # read img
            image = cv2.imread(image_path)

            # crop face part
            x, y = rect[0], rect[1]
            w, h = rect[2], rect[3]
            image = image[y:y+h, x:x+w]

            # resize 96x96
            image = cv2.resize(image,(96, 96))

            # save
            image_name = os.path.basename(image_path)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, image_name) , image)

        print("{} --> {}".format(in_dir, out_dir))

if __name__=="__main__":
    if len(sys.argv) < 3:
        print("usage: python preprocess.py <dataset path> <output path>")
        sys.exit()

    main(sys.argv[1], sys.argv[2])
