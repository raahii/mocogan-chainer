"""
Perform following preprocessing for ** MUG Facial Dataset **.
 1. discard videos of neutral facial expression
 2. extract subsequences of video clips
 3. crop the face region
 4. resize 64 x 64
 5. save samples for each category

To use this script, pass the arguments
MUG dataset directory and save directory.

I suppose that dataset path is 'subject3' directory
in MUG dataset

<dataset path (e.g. subject3)> 
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
import argparse
import sys, os, glob, shutil
import cv2
import re

import multiprocessing 

CASCADE_PATH = "data/cv_models/haarcascade_frontalface_default.xml"

frame_name_regex = re.compile(r'([0-9]+).jpg')

def frame_number(name):
    match = re.search(frame_name_regex, name)
    return match.group(1)

video_info_regex = re.compile(r'.*?\/(?P<user_num>[0-9]+)\/(session(?P<session_num>[0-9])\/)?(?P<expression>.*?)\/take(?P<take_num>[0-9]+)')

def extract_video_info(name):
    match = re.search(video_info_regex, name)
    user_num    = int(match.group('user_num'))
    session_num = match.group('session_num')
    session_num = 0 if session_num is None else int(session_num)
    expression  = match.group('expression')
    take_num    = int(match.group('take_num'))

    return user_num, session_num, expression, take_num

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

def perform_preprocess_multi(args):
    perform_preprocess(*args)

def perform_preprocess(in_dir, save_path, options):
    images = glob.glob(os.path.join(in_dir, '*.jpg'))
    images = sorted(images, key=frame_number)

    edge, speeds, length, stride = options

    edge_frames = int(len(images) * edge)
    start = 0
    end   = len(images) - edge_frames

    num_created_samples = 0
    for speed in speeds:
        frame_width = speed*length
        for seq_n, j in enumerate(range(start, end-frame_width, stride)[0:-1]):
            user_num, session_num, expression, take_num = extract_video_info(in_dir)
            out_dir = os.path.join(save_path, "user{:03d}_sess{}_take{:03d}_{:02d}".format(user_num, session_num, take_num, seq_n))
            os.makedirs(out_dir, exist_ok=True)
        
            # detect face region of the video clip
            mid_frame = j + frame_width//2
            image = cv2.imread(images[mid_frame])
            rect = detect_face(image)
            if rect is None:
                shutil.rmtree(out_dir)
                continue
            
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

            num_created_samples += 1
            print("{}({} frames) --> {}(offset:{}, speed:{})".format(
                                                    '/'.join(in_dir.split('/')[-3:]),
                                                    len(images),
                                                    '/'.join(out_dir.split('/')[-2:]),
                                                    j, speed))
    print("")

    return num_created_samples

def main():
    parser = argparse.ArgumentParser(description='Preprocessing script for MUG Facial Expression Database')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('save_path', type=str)
    # parser.add_argument('--use_labels', action='store_true')
    args = parser.parse_args()

    edge   = 0.10 # dont use end of frames of the video
    speeds = [2] # use 1 frame per speed frames ( to change speed )
    length = 16 # video length (frame num)
    stride = length // 2 # stride width

    # list all video clips
    facial_expressions = ["anger", "disgust", "happiness",
                          "fear", "sadness", "surprise"]
    num_orginal_samples = 0
    for label, exp in enumerate(facial_expressions):
        video_paths = glob.glob(os.path.join(args.dataset_path, '*', exp, '*'))
        # good_takes = list(filter(lambda name: name.find('nsg') < 0, takes))
        # video_paths.extend(good_takes)
        num_orginal_samples += len(video_paths)
        print(">>> {}: {} video clips found.".format(exp, len(video_paths)))
    
        # perform preprocess
        saved_num = 0
        category_path = os.path.join(args.save_path, str(label))
        os.makedirs(category_path, exist_ok=True)

        # for i, in_dir in enumerate(video_paths):

        # multi
        print('mode multi')
        video_num = len(video_paths)
        num_orginal_samples += video_num
        args_iter = zip(video_paths, [category_path]*video_num, [(edge, speeds, length, stride)]*video_num)
        pool = multiprocessing.Pool(4)
        pool.map(perform_preprocess_multi, args_iter)
        pool.close()

        # print(">>> preprocess finished, original {} video clips were converted to {} new video clips".format(num_orginal_samples, num_created_samples))
        print(">>> preprocess finished, original {} video clips were converted.".format(num_orginal_samples))

if __name__=="__main__":
    main()
    # if len(sys.argv) < 3:
    #     print("usage: python preprocess.py <dataset path> <output path>")
    #     sys.exit()
    #
    # main(sys.argv[1], sys.argv[2])
