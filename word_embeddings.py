import gensim
import torch
from utils.util import read_caption_data, char_table_to_sentence, word2vec
from torch.utils.serialization import load_lua
import os


def save_embeddings(filepath, filename, embeddings):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    target_path = os.path.join(filepath, filename)
    torch.save({'embeds': embeddings}, target_path)
    return True


def load_caption_and_save_embeds(caption_list):
    for idx, cap in enumerate(caption_list):
        assert (os.path.isfile(caption_list[idx]))

        cls, fn = cap.split('/')[-2], cap.split('/')[-1]
        # print(cls, fn)
        filepath = '/'.join((dir_path, cls))

        caption = load_lua(cap)
        char = caption['char']

        sentence = char_table_to_sentence(alphabet, char)

        embeds = word2vec(sentence, model, sen_size=16, emb_size=300)

        if save_embeddings(filepath, fn, embeds):
            print(filepath, fn, 'saved')


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

data_dir = '../datasets/CUB_200_2011/'
caption_dir = os.path.join(data_dir, 'cub_icml')
split_file = os.path.join(data_dir, 'train_val.txt')
dir_path = os.path.join(data_dir, 'pretrained_embeddings')

caption_list = read_caption_data(caption_dir, split_file)

model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)

load_caption_and_save_embeds()

# check load
data = torch.load(os.path.join(dir_path, '007.Parakeet_Auklet/Parakeet_Auklet_0064_795954.t7'))
print(data['embeds'])