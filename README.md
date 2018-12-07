Pytorch implementation of paper 'See, Hear, and Read: Deep Aligned Representations' 
[[arxiv]](http://cn.arxiv.org/abs/1706.00932v1)

In this paper, a Teacher-Student model-like network is designed to learn a deep discriminative representations shared across three major natural modalities: vision, sound and language. The student network accepts input either an image, a sound, or a text, and produces correponding modality-specific representation (gray). The teacher network produces a common shared representation (blue) that is aligned across modality from the modality-specific representations. 

<img src="./image/LearningDeepRepresentations.jpg" width="900px" />

Experiment results show that this representation is useful for several tasks, such as cross-modal retrieval or transferring classifiers between modalities. 

## Prerequisites
- Linux or OSX
- python 3
- [Pytorch installation](https://pytorch.org/get-started/locally/) 
- 8G+ free memory 
- GPU with CUDA is recommended

## Get started
### Clone this repository
run command in terminal
```bash
git clone https://github.com/jingliao132/CrossModalRetrieval.git
cd CrossModalRetrieval
```

### Download the [CUB-200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200.html)
```bash
./datasets/download_dataset.sh CUB_200_2011
```
will download and unzip the CUB-200 data in folder 'CUB_200_2011' under ./datasets

### Download [CUB-200 caption data](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing)(torch format) with browser or wget (refer to [Download Google Drive files with WGET](https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805) and Extract the file under foler 'CUB_200_2011'

Each caption file contains a dict object with keys:

'char', a character-level one-hot mapping of 10 text descriptions on the image; 

'img', the file name of the image; 

'word', word-level coding of 10 text descriptions 

and 'txt', 1024-dimentional text feature by pretrained GoogLeNet (Details in https://github.com/reedscot/icml2016). We use only 'char' and 'img'.

<img src="./image/caption_data_example.jpg" width="900px" />

### Set up training and validation manifest file
Example files train.txt val.txt train_val.txt (for producing word embeddings files) are provided in ./datasets. Move them into folder 'CUB_200_2011'.

Customize your train/val by editing train.txt & val.txt

### New folder 'models' , and Download [pre-trained word-to-vector model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (binary format) to 'models'

### Produce word embeddings files using the pre-trained model
```bash
python3 ./utils/word_embeddings.py
```

### Train the model
```bash
python3 __init__.py
```

## Acknowledgement
@techreport{WelinderEtal2010,
	Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2010-001},
	Title = {{Caltech-UCSD Birds 200}},
	Year = {2010}
}

@inproceedings{reed2016generative,
  title={Generative Adversarial Text-to-Image Synthesis},
  author={Scott Reed and Zeynep Akata and Xinchen Yan and Lajanugen Logeswaran and Bernt Schiele and Honglak Lee},
  booktitle={Proceedings of The 33rd International Conference on Machine Learning},
  year={2016}
}

## Reference
@article{DBLP:journals/corr/AytarVT17,
  author    = {Yusuf Aytar and
               Carl Vondrick and
               Antonio Torralba},
  title     = {See, Hear, and Read: Deep Aligned Representations},
  journal   = {CoRR},
  volume    = {abs/1706.00932},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.00932},
  archivePrefix = {arXiv},
  eprint    = {1706.00932},
  timestamp = {Mon, 13 Aug 2018 16:48:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/AytarVT17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
