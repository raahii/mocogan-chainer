import sys, os, glob
import re
from pathlib import Path

import chainer
import numpy as np
from PIL import Image
from tqdm import tqdm

frame_name_regex = re.compile(r'([0-9]+).jpg')

def frame_number(name):
    match = re.search(frame_name_regex, str(name))
    return match.group(1)

def read_video(paths):
    video = []
    for path in paths:
        f = Image.open(path)
        try:
            frame = np.asarray(f, dtype=np.float32)
            video.append(frame)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return np.asarray(video, dtype=np.float32)

class MugDataset(chainer.dataset.DatasetMixin):
    # {{{
    def __init__(self, root_path, video_length=16):
        self.root_path = Path(root_path)
        self.video_length = video_length
        self.extract_speed = 2

        self.video_categories = list(self.root_path.glob("*"))
        self.num_labels = len(self.video_categories)

        category2num = {
            "anger":     0,
            "disgust":   1,
            "happiness": 2,
            "fear":      3,
            "sadness":   4,
            "surprise":  5,
        }

        self.videos = []
        for category_path in self.video_categories:
            if not category_path.is_dir():
                continue

            num_categ = category2num[category_path.name]
            for video_path in category_path.glob("*"):
                if not video_path.is_dir():
                    continue
                
                video_len = len(list(video_path.glob("*.jpg")))
                if video_len >= video_length:
                    self.videos.append((video_path, num_categ))
                else:
                    print(">> discarded {} (video length {} < {})\n".
                            format(video_path.parent.name, video_len, video_length))

    def __len__(self):
        return len(self.videos)

    def get_example(self, i):
        """return video shape: (ch, frame, width, height)"""
        video_path, categ = self.videos[i]

        frame_paths = np.array(sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=frame_number))

        # videos can be of various length, we randomly sample sub-sequences
        video_len = len(frame_paths)
        if video_len < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                .format(len(frame_paths), self.video_length))
        elif video_len > self.video_length * self.extract_speed:
            needed = self.extract_speed * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
            frame_paths = frame_paths[subsequence_idx]
        else:
            gap = video_len - self.video_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.arange(start, start+self.video_length)
            frame_paths = frame_paths[subsequence_idx]
    
        # read video
        video = read_video(frame_paths)
        if len(video.shape) != 4:
            raise ValueError('invalid video shape: {}'.format(video.shape))
        video = (video - 128.) / 128.
        
        # # concat label data as feature maps
        # t, y, x, c = video.shape
        # label_video = -1.0 * np.ones((t, y, x, self.num_labels), dtype=np.float32)
        # label_video[:,:,:,categ] = 1.0
        #
        # # concat video and label
        # video = np.concatenate((video, label_video), axis=3)

        video = video.transpose(3, 0, 1, 2) # (C, T, H, W)
        
        return video, categ
    # }}}

class MovingMnistDataset(chainer.dataset.DatasetMixin):
    # {{{
    def __init__(self, dataset_path, video_length=16):
        self.video_length = video_length
        
        save_path = Path("data/dataset/moving_mnist/preprocessed")
        if not save_path.exists():
            self.preprocess(dataset_path, save_path)

        self.videos = [path for path in save_path.glob("*") if path.is_dir()]

    def __len__(self):
        return len(self.videos)

    def preprocess(self, dataset_path, save_path):
        print("\npreprocessing....")
        videos = np.load(dataset_path)
        videos = np.tile(videos[:,:,:,:,None], (1, 1, 1, 1, 3))
        videos = videos.transpose(1, 0, 2, 3, 4) # (N, T, H, W, C)
        
        print("writing out {} videos:\n\t{} ---> {}".
                format(videos.shape[0], dataset_path, save_path))
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, video in tqdm(enumerate(videos)):
            path = (save_path / "{:05d}".format(i))
            path.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(video):
                Image.fromarray(img).save(path/"{:02d}.jpg".format(j))
        print("")

    def get_example(self, i):
        video_path = self.videos[i]
        
        frame_paths = sorted(list(video_path.glob("*.jpg")), key=frame_number)
        frame_paths = np.array(frame_paths)

        # videos can be of various length, we randomly sample sub-sequences
        video_len = len(frame_paths)
        if video_len < self.video_length:
            raise ValueError('invalid video length: {} < {} ({})'
                .format(len(frame_paths), self.video_length, video_path))
        else:
            gap = video_len - self.video_length
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.arange(start, start+self.video_length)
            frame_paths = frame_paths[subsequence_idx]

        # read video
        video = read_video(frame_paths)
        if len(video.shape) != 4:
            raise ValueError('invalid video shape: {}'.format(video.shape))
        video = (video - 128.) / 128.
        video = video.astype(np.float32)
        video = video.transpose(3, 0, 1, 2) # (C, T, H, W)
        
        return video, None
    # }}}
