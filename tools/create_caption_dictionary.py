from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VQACAPdataset import Dictionary
from pycocotools.coco import COCO
import pickle

def create_dictionary(dictionary):
    names = ['train','val']
    for name in names:
        coco = COCO('data/annotations/captions_'+name+'2014.json')
        caption_ids = list(coco.anns.keys())
        length_captions = 0
        for i in range(len(caption_ids)):
            idx = caption_ids[i]
            caption = coco.anns[idx]['caption']
            dictionary.tokenize(caption, True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    with open('data/cache/trainval_ans2label.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    with open('data/cache/trainval_label2ans.pkl', 'rb') as f:
        idx2word_ = pickle.load(f)
    
    idx2word = []
    for i in xrange(len(idx2word_)):
      idx2word.append(idx2word_[i])
    caption_dictionary = Dictionary( word2idx=word2idx, idx2word=idx2word)
    caption_dictionary.add_word('<pad>')
    caption_dictionary.add_word('<start>')
    caption_dictionary.add_word('<end>')
    caption_dictionary.add_word('<unk>')

    caption_dictionary = create_dictionary(caption_dictionary)
    caption_dictionary.dump_to_file('data/caption_dictionary.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    #with open('/data/wujial/Attention-on-Attention-for-VQA/data/cache/trainval_label2ans.pkl', 'rb') as f:
    #    x = pickle.load(f)
    weights, word2emb = create_glove_embedding_init(caption_dictionary.idx2word, glove_file)
    np.save('data/glove6b_caption_init_%dd.npy' % emb_dim, weights)
