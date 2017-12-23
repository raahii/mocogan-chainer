import sys, os, glob
import re

import chainer
import numpy as np
from PIL import Image

frame_name_regex = re.compile(r'([0-9]+).jpg')

def frame_number(name):
    match = re.search(frame_name_regex, name)
    return match.group(1)

def read_images(paths):
    video = []
    for path in paths:
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.float32)
            video.append(image)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return np.asarray(video, dtype=np.float32)

class MugDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root_path, video_length=16):
        self.root_path = root_path
        self.video_length = video_length

        self.video_categories = glob.glob(os.path.join(self.root_path, "*"))
        self.num_labels = len(self.video_categories)

        self.videos = []
        for category in self.video_categories:
            categ = int(os.path.basename(category))
            for video_path in glob.glob(os.path.join(category, "*")):
                self.videos.append((video_path, categ))

    def __len__(self):
        return len(self.videos)

    def get_example(self, i):
        """return video shape: (ch, frame, width, height)"""
        video_path, categ = self.videos[i]

        frame_paths = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=frame_number)
        frame_paths = np.array(frame_paths)

        # videos can be of various length, we randomly sample sub-sequences
        if len(frame_paths) < self.video_length:
            raise ValueError('invalid video length: {} < {}'
                .format(len(frame_paths), self.video_length))

        # read video
        video = read_images(frame_paths)
        if len(video.shape) != 4:
            raise ValueError('invalid video.shape')

        # label data
        t, y, x, c = video.shape 
        label_video = -1.0 * np.ones((t, y, x, self.num_labels), dtype=np.float32)

        label_video[:,:,:,categ] = 1.0

        # concat video and label
        video = np.concatenate((video, label_video), axis=3)
        video = video.transpose(3, 0, 1, 2) # (ch, video_len, y, x)
        
        return (video - 128.) / 128., categ

class MovingMnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, npz_path, video_length=16):
        self.npz_path = npz_path
        self.video_length = video_length
        # self.data = self.preprocess(np.load(npz_path)[:, :10])
        self.data = self.preprocess(np.load(npz_path))

    def __len__(self):
        return self.data.shape[0]
    
    def preprocess(self, data):
        t, n, h, w = data.shape
        data = data[:self.video_length].transpose(1, 0, 2, 3)
        data = data[:, np.newaxis,:,:,:]
        data = (data - 128.) / 128.

        return data.astype(np.float32)

    def get_example(self, i):
        return self.data[i]
