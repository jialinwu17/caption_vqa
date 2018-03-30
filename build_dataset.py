from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
def _create_entry():
    #answer.pop('image_id')
    #answer.pop('question_id')
    entry = {
        'question_id' : [],#question['question_id'],
        'image_id'    : [],#question['image_id'],
        'image'       : [],#img,
        'caption'     : [],
        'question'    : [],#question['question'],
        'answer'      : []}#answer}
    return entry

dataroot = 'data'
names = ['train','val']
for name in names:

    img_id2idx = cPickle.load(open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
    question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))

    entries = {}
    coco = COCO('data/annotations/captions_'+name+'2014.json')

    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        feat_id = img_id2idx[img_id]
        if feat_id not in entries:
            entries[feat_id] = _create_entry()
        entries[feat_id]['question_id'] = question['question_id']
        entries[feat_id]['image_id'] = question['image_id']
        entries[feat_id]['image'] = feat_id
        entries[feat_id]['question'].append(question)
        entries[feat_id]['answer'].append(answer)

        #entries.append(_create_entry(img_id2val[img_id], question, answer))

    caption_ids = list(coco.anns.keys())
    length_captions = 0
    for i in range(len(caption_ids)):
        idx = caption_ids[i]
        caption = coco.anns[idx]['caption']
        feat_id = img_id2idx[coco.anns[idx]['image_id']]
        entries[feat_id]['caption'].append(caption)


    cPickle.dump(entries, open('VQA_caption_'+name+'dataset.pkl', 'wb'))