from __future__ import print_function, division
from torch.utils.data import Dataset
from torch.utils.serialization import load_lua
from skimage import io
from utils.util import read_caption_data
import os

class CMRDataset(Dataset):

    def __init__(self, root_dir, caption_dir, image_dir, split, transform=None):
        assert(os.path.isdir(os.path.join(root_dir, caption_dir)))
        assert(os.path.isfile(os.path.join(root_dir, split)))

        self.caption_list = read_caption_data(os.path.join(root_dir, caption_dir),
                                              os.path.join(root_dir, split))
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        assert(os.path.isfile(self.caption_list[idx]))
        caption = load_lua(self.caption_list[idx])
        char = caption['char']
        img_path = os.path.join(self.root_dir, self.image_dir, caption['img'])
        image = io.imread(img_path)
        txt = caption['txt']
        word = caption['word']
        sample = {'char': char, 'image': image, 'txt': txt, 'word': word}

        if self.transform:
            sample = self.transform(sample)

        return sample
