import os
import torch
import numpy as np
from skimage import transform
import torch.nn.functional as F
import gensim

# caption data path architecture as:
# -- caption dir
# ---- caption class dir
# ------ caption file
#
def read_caption_data(caption_dir, split_file):
    if not os.path.exists(caption_dir):
        print(caption_dir+' not exists!')
        return []

    cap_cls_list = []
    caption_list = []

    file = open(split_file, 'r')
    line = file.readline().strip('\n')
    while line:
        _, cls = line.split(' ')
        #print(cls)
        cap_cls_list.append(os.path.join(caption_dir, cls))
        line = file.readline().strip('\n')

    file.close()

    for cap_cls_dir in cap_cls_list:
        if not os.path.isdir(cap_cls_dir):
            print(cap_cls_dir + ' not exists')
            continue
        for cap in os.listdir(cap_cls_dir):
            #print(cap)
            caption = os.path.join(cap_cls_dir, cap)
            if os.path.isfile(caption):
                caption_list.append(caption)
            else:
                print('not found caption file ', caption)

    return caption_list


def char_table_to_embeddings(model_path, char, alphabet,
                             sen_size, emb_size, batch_size, device):
    sentence = [char_table_to_sentence(alphabet=alphabet, char_table=char[idx])
                for idx in range(0, batch_size)]
    embeds = torch.zeros([batch_size, sen_size, emb_size],
                         #dtype=torch.float64,
                         device=device)

    # Load word embedding model: using word-to-vec
    #print('Loading the pre-trained word-to-vec model: GoogleNews-vectors-negative300.bin...')
    assert (os.path.exists(model_path))
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=500000)

    for idx in range(0, batch_size):
        embeds[idx] = word2vec(model, sentence[idx],
                               sen_size=sen_size, emb_size=emb_size)
    return embeds


# decode the char_table, and return the sentence in string format
def char_table_to_sentence(alphabet, char_table):
    sentence = None
    has_multiple_sentence = len(char_table.stride()) > 1
    if has_multiple_sentence:
        for sentence_i in torch.t(char_table):
            sentence = [alphabet[int(idx.item()-1)] for idx in sentence_i]    # index start from 1 in char_table
    return ''.join(list(map(str, sentence)))


def word2vec(model, sentence, sen_size, emb_size):
    words = sentence.rstrip().split(' ')
    words_to_remove = []

    for idx, word in enumerate(words):
        cleaned_word = word.strip(",.")
        words[idx] = cleaned_word

        if cleaned_word not in model.vocab.keys():
            # print(word, 'not in vocabulary')
            words_to_remove.append(cleaned_word)

    for word in words_to_remove:
        while word in words:
            words.remove(word)

    wordsInVocab = len(words)
    vocab = {words[idx]: idx for idx in range(0, wordsInVocab)}

    embeddings = np.zeros((sen_size, emb_size))

    for k, v in vocab.items():
        if v > 15:
            continue
        embeddings[v] = model[k]

    return torch.from_numpy(embeddings)


def Cos_similarity(x, y, dim=1):
    assert(x.shape == y.shape)
    if len(x.shape) >= 2:
        return F.cosine_similarity(x, y, dim=dim)
    else:
        return F.cosine_similarity(x.view(1, -1), y.view(1, -1))


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'char': sample['char'], 'image': img, 'txt': sample['txt'], 'word': sample['word']}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        return {'char': sample['char'], 'image': image, 'txt': sample['txt'], 'word': sample['word']}


class CenterCrop(object):
    """Crop the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = int(0.5 * (h - new_h))
        left = int(0.5 * (w - new_w))

        image = image[top: top + new_h, left: left + new_w]

        return {'char': sample['char'], 'image': image, 'txt': sample['txt'], 'word': sample['word']}


class ToTensor(object):

    def __call__(self, sample):
        """Convert ndarrays in sample to Tensors."""
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(image.shape)
        #try:
        image = image.transpose((2, 0, 1))
        # except Exception as e:
        #     print(image.shape)
        #     print(e)

        #image = torch.as_tensor(torch.from_numpy(image), dtype=torch.float64)
        image = torch.from_numpy(image)
        return {'char': sample['char'], 'image': image, 'txt': sample['txt'], 'word': sample['word']}

class Normalize(object):

    def __init__(self, mean, std):
        assert isinstance(mean, (list, tuple))
        assert isinstance(std, (list, tuple))
        for m in mean:
            assert isinstance(m, float)
        assert len(mean) == len(std) == 3
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """Normalize a tensor image with mean and standard deviation."""
        image = sample['image']
        image = [(image[c] - self.mean[c]) / self.std[c] for c in range(0, 3)]
        return {'char': sample['char'], 'image': image, 'txt': sample['txt'], 'word': sample['word']}

#print(read_caption_data('../datasets/CUB_200_2011/cub_icml'))