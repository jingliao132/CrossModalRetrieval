from __future__ import print_function, division
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.serialization import load_lua
#from torchvision import transforms, utils
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

# test
# cub_dataset = CMRDataset(caption_dir='cub_icml', image_dir='images', root_dir='../datasets/CUB_200_2011')
# fig = plt.figure()
# for i in range(len(cub_dataset)):
#     sample = cub_dataset[i]
#     print(i, sample['char'].shape, sample['image'].shape, sample['txt'].shape, sample['word'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['image'])
#
#     if i == 3:
#         plt.show()
#         break

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