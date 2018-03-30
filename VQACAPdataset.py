import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    @property
    def caption_padding_idx(self):
        return 3129

    @property
    def caption_start_idx(self):
        return 3130
    '''
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    '''
    @property
    def caption_end_idx(self):
        return 3131
    
    @property
    def caption_unk_idx(self):
        return 3132


    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary,caption_dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label) #because we add captions voc

        self.dictionary = dictionary
        self.caption_dictionary = caption_dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36_ori.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

            #self.spatials = np.ones(5,2)
        self.coco = COCO('data/annotations/captions_'+name+'2014.json')


        #self.entries = _load_dataset(dataroot, name, self.img_id2idx)
        self.entries  = cPickle.load(open('VQA_caption_'+name+'dataset.pkl', 'rb'))
        
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14, max_caption_length = 20):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]

            entry['q_token']= []
            entry['c_token']= []
            for q_id in range(len(entry['question'])):
                tokens = self.dictionary.tokenize(entry['question'][q_id]['question'], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
                entry['q_token'].append(tokens)


            for c_id in range(len(entry['caption'])):
                caption_tokens = []
                caption_tokens.append(self.caption_dictionary.word2idx['<start>'])
                caption_tokens.extend(self.caption_dictionary.tokenize(entry['caption'][c_id], False))
                caption_tokens.append(self.caption_dictionary.word2idx['<end>'])
                caption_tokens = caption_tokens[ : max_caption_length]
                if len(caption_tokens) < max_caption_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.caption_padding_idx] * (max_caption_length - len(caption_tokens))
                    caption_tokens =  caption_tokens + padding 
                utils.assert_eq(len(caption_tokens), max_caption_length)
                entry['c_token'].append(caption_tokens)

    def tensorize(self,max_length = 14, max_caption_length = 20):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            entry['q_token_tensor'] = torch.zeros(len(entry['q_token']), max_length )
            entry['answer_tensor'] = torch.zeros(len(entry['q_token']), self.num_ans_candidates )
            entry['c_token_tensor'] = torch.zeros(len(entry['c_token']), max_caption_length )
            #entry['caption_tensor'] = torch.zeros(len(entry['c_token']),max_caption_length, self.caption_dictionary.ntoken )
            for q_id in range(len(entry['q_token'])):
                question = torch.from_numpy(np.array(entry['q_token'][q_id]))
                entry['q_token_tensor'][q_id,: ] = question

                answer = entry['answer'][q_id]
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer'][q_id]['labels'] = labels
                    entry['answer'][q_id]['scores'] = scores
                    entry['answer_tensor'][q_id,:].scatter_(0, labels, scores)
                else:
                    entry['answer'][q_id]['labels'] = None
                    entry['answer'][q_id]['scores'] = None

            for c_id in range(len(entry['c_token'])):
                caption = torch.from_numpy(np.array(entry['c_token'][c_id]))
                entry['c_token_tensor'][c_id,: ] = caption
                #entry['answer_tensor'] = torch.zeros(len(entry['q_token']), self.num_ans_candidates )
                #entry['caption_tensor'][c_id,:].scatter_(1, caption.view(-1,1), 1)


    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]
        captions = entry['c_token_tensor']
        
        features = features.unsqueeze(0).repeat(5,1,1)
        questions = entry['q_token_tensor'] #[5, 36, 2048]
        target = entry['answer_tensor'] 
        q_idx = np.mod(np.random.permutation(np.maximum(5, questions.shape[0]))[:5],questions.shape[0])
        c_idx = np.mod(np.random.permutation(np.maximum(5,captions.shape[0]))[:5],captions.shape[0])
        #print c_idx,q_idx
        captions =torch.index_select(captions, 0, torch.LongTensor(c_idx.astype('float32')))
        #[5,20,num_hid]
        questions = torch.index_select(questions, 0, torch.LongTensor(q_idx.astype('float32')))
        #[5,14,num_hid]
        target = torch.index_select(target, 0, torch.LongTensor(q_idx.astype('float32')))
        #[5,14,num_ans]
        #caption_target = entry['caption_tensor']
        return features, spatials, questions, target, captions#, caption_target

    def __len__(self):
        return len(self.entries)
