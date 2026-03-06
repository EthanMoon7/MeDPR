import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image

def _init_fn(worker_id):
    np.random.seed(2024)

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

class bert_data():
    
    def __init__(self,max_len, batch_size, vocab_file,num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
      
    def load_data(self,path,imagepath,clipimagepath,shuffle):
        self.data = pd.read_csv(path,encoding='utf-8')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./pretrained_model')
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content)
        
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                ordered_image,
                                clip_image,
                                clip_text
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = shuffle,
            worker_init_fn = _init_fn
        )
        return dataloader
