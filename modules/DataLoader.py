from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torch.utils.serialization import load_lua
from skimage import io, color
from utils.util import read_caption_data, Rescale, RandomCrop, ToTensor, caption_path_to_embeds_path
import os
import torch

from torchvision import transforms, utils
import matplotlib.pyplot as plt

class CMRDataset(Dataset):

    def __init__(self, root_dir, caption_dir, image_dir, embeds_dir, split, transform=None):
        assert(os.path.isdir(os.path.join(root_dir, caption_dir)))
        assert(os.path.isfile(os.path.join(root_dir, split)))
        self.root_dir = root_dir
        self.caption_dir = os.path.join(root_dir, caption_dir)
        self.split_file = os.path.join(root_dir, split)
        self.image_dir = image_dir
        self.embeds_dir = os.path.join(root_dir, embeds_dir)
        self.transform = transform
        self.caption_list = read_caption_data(self.caption_dir, self.split_file)

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        assert(os.path.isfile(self.caption_list[idx]))
        caption = load_lua(self.caption_list[idx])

        embeds_path = caption_path_to_embeds_path(self.caption_list[idx], self.embeds_dir)
        embeds = torch.load(embeds_path)['embeds']

        img_path = os.path.join(self.root_dir, self.image_dir, caption['img'])
        image = io.imread(img_path)

        # if it is a gray image convert to rgb
        if len(image.shape) < 3:
            image = color.gray2rgb(image)

        txt = caption['txt']
        word = caption['word']
        sample = {'embeds': embeds, 'image': image, 'txt': txt, 'word': word}

        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                print(e)
                print(self.caption_list[idx])
        # assert (image.shape[0] == 3, img_path)
        return sample

# cub_dataset = CMRDataset(root_dir='../datasets/CUB_200_2011', caption_dir='cub_icml', image_dir='images',
#                          split = 'train_val.txt',
#                          transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
#
# for i in range(len(cub_dataset)):
#     sample = cub_dataset[i]
#     print(i, sample['char'].shape, sample['image'].shape, sample['txt'].shape, sample['word'].shape)
#
#     if i == 3:
#         break
#
# dataloader = DataLoader(cub_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)
#
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size())
#
#     if i_batch == 3:
#         plt.figure()
#         images_batch = sample_batched['image']
#         batch_size = len(images_batch)
#         grid = utils.make_grid(images_batch)
#         plt.imshow(grid.numpy().transpose((1, 2, 0)))
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break