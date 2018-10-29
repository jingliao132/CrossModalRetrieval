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

class Word_Embeddings():

    def __init__(self, root_dir, caption_dir, split_file, alphabet):
        self.caption_dir = caption_dir
        self.split_file = split_file
        self.alphabet = alphabet
        self.dir_path = os.path.join(root_dir, 'pretrained_embeddings')
        self.model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)

    def load_caption(self, cap):
        assert (os.path.isfile(cap))

        cls, fn = cap.split('/')[-2], cap.split('/')[-1]
        # print(cls, fn)
        filepath = '/'.join((self.dir_path, cls))

        caption = load_lua(cap)
        char = caption['char']

        sentence = char_table_to_sentence(self.alphabet, char)
        # print(sentence)
        embeds = word2vec(self.model, sentence, sen_size=16, emb_size=300)

        return filepath, fn, embeds


# alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
#
# data_dir = '../datasets/CUB_200_2011/'
#
# split_file = os.path.join(data_dir, 'train_val.txt')
#
# caption_dir = os.path.join(data_dir, 'cub_icml')
#
# caption_list = read_caption_data(caption_dir, split_file)
#
# WE = Word_Embeddings(data_dir, caption_dir, split_file, alphabet)
#
# for idx, cap in enumerate(caption_list):
#     filepath, fn, embeds = WE.load_caption(cap)
#     if save_embeddings(filepath, fn, embeds):
#         print(filepath, fn, 'saved')
#
#
# # check load
# data = torch.load(os.path.join(WE.dir_path, '007.Parakeet_Auklet/Parakeet_Auklet_0064_795954.t7'))
# print(data['embeds'])
