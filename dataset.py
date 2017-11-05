import sys, os, glob
import chainer
import numpy as np
from PIL import Image

def read_images(paths, dtype):
    video = []
    for path in paths:
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=dtype)
            video.append(image)
        finally:
            if hasattr(f, 'close'):
                f.close()

    return np.asarray(video, dtype=dtype)

class VideoDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root_path, video_length=16, scale=2, dtype=np.float32):
        self.root_path = root_path
        self.video_paths = glob.glob(os.path.join(root_path, "*"))
        self.video_length = video_length
        self.scale = scale
        self.dtype = dtype

    def __len__(self):
        return len(self.video_paths)

    def get_example(self, i):
        """return video shape: (ch, frame, width, height)"""
        frame_paths = np.asarray(glob.glob(os.path.join(self.video_paths[i], '*.jpg')))

        # videos can be of various length, we randomly sample sub-sequences
        video_len = len(frame_paths)
        if video_len > self.video_length * self.scale:
            needed = self.scale * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subseq = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subseq = np.arange(0, self.video_length)
        else:
            raise ValueError('invalid video length: {} < {}'
                .format(video_len, self.video_length))

        # read video
        video = read_images(frame_paths[subseq], self.dtype)
        if len(video.shape) != 4:
            raise ValueError('invalid video.shape')
        
        return (video.transpose(3, 0, 1, 2) - 128.) / 128.
        # return video.transpose(3, 0, 1, 2)
