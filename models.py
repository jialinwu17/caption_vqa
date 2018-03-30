import torch
import torch.nn as nn
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from caption_model import CaptionRNN

import torch.nn.functional as F
import utils
from torch.autograd import Variable
import shutil
import numpy as np
import matplotlib.pyplot as plt
# Dropout p: probability of an element to be zeroed. Default: 0.5

"""
Name: Model

Pre written
"""
class Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]


        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class Model_2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,caption_w_emb, reference_caption_decoder, question_caption_decoder,caption_decoder,v2rc_net,v2qc_net):
        super(Model_2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reference_caption_decoder = reference_caption_decoder
        self.question_caption_decoder = question_caption_decoder
        self.caption_w_emb = caption_w_emb
        self.caption_decoder = caption_decoder
        self.v2rc_net = v2rc_net
        self.v2qc_net = v2qc_net

    def forward(self, v, b, q, labels, c):
        """Forward

        v: [batch,5, num_objs, obj_dim]
        b: [batch, 5,num_objs, b_dim]
        q: [batch, 5, seq_length]
        c: [batch, 5, 20 ]

        return: logits, not probs
        """
        
        batch = c.size(0)
        q = q.view(batch * 5, -1)
        c = c.view(batch * 5, -1)
        v = v.view(batch * 5, 36, -1)
        batch = c.size(0)
        '''
        v: [batch* 5, num_objs, obj_dim]
        q: [batch* 5, seq_length]
        c: [batch* 5, 20 ]
        '''
        #print c.shape, type(c)
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]
        # [batch* 5, num_hid]
        #print c.shape, type(c)
        att_1 = self.v_att_1(v, q_emb) # [batch* 5, 1, v_dim]
        #print c.shape, type(c)
        att_2 = self.v_att_2(v, q_emb)  # [batch* 5, 1, v_dim]
        att = att_1 + att_2
        #print c.shape, type(c)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        #print c.shape, type(c)
        
        q_repr = self.q_net(q_emb) #[batch * 5 ,hid_dim]
        v_repr = self.v_net(v_emb)#[batch *5, hid_dim]
        #print c.shape, type(c)
        # v_repr = v_repr.unsqueeze(1).repeat(1,5,1).view(batch*5,-1)
        joint_repr = q_repr * v_repr #[batch *5,hid_dim ]

        logits = self.classifier(joint_repr)
        #print c.shape, type(c)
        rc_w_emb = self.caption_w_emb(c)
        qc_w_emb = self.caption_w_emb(c) # [batch * 5, 20 , hid_dim]
        #print c.shape, type(c)

        v_rc = self.v2rc_net(v)
        v_qc = self.v2qc_net(joint_repr)

        rc_emb = self.reference_caption_decoder(rc_w_emb, v_rc)
        #[b,5,21,hid_dim]
        qc_emb = self.question_caption_decoder(v_qc ,qc_w_emb)
        #[b,5,21,hid_dim]
        rc_repr = self.caption_decoder(rc_emb)
        qc_repr = self.caption_decoder(qc_emb)
        
        return logits, rc_repr, qc_repr
        # all three returns are logits

class Model_4(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,caption_w_emb, reference_caption_decoder, question_caption_decoder,caption_decoder,v2rc_net, v2qc_net):
        super(Model_4, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reference_caption_rnn = reference_caption_decoder
        self.question_caption_rnn = question_caption_decoder
        self.caption_w_emb = caption_w_emb
        self.caption_decoder = caption_decoder
        self.v2rc_net = v2rc_net
        self.v2qc_net = v2qc_net

    def forward(self, v, b, q, labels, c):
        """Forward

        v: [batch,5, num_objs, obj_dim]
        b: [batch, 5,num_objs, b_dim]
        q: [batch, 5, seq_length]
        c: [batch, 5, 20 ]

        return: logits, not probs
        """
        print 'haha1'
        batch = c.size(0)
        q = q.view(batch * 5, -1)
        c = c.view(batch * 5, -1)
        v = v.view(batch * 5, 36, -1)
        batch = c.size(0)
        '''
        v: [batch* 5, num_objs, obj_dim]
        q: [batch* 5, seq_length]
        c: [batch* 5, 20 ]
        '''
        #print c.shape, type(c)
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]
        # [batch* 5, num_hid]
        #print c.shape, type(c)
        att_1 = self.v_att_1(v, q_emb) # [batch* 5, 1, v_dim]
        #print c.shape, type(c)
        att_2 = self.v_att_2(v, q_emb)  # [batch* 5, 1, v_dim]
        att = att_1 + att_2
        #print c.shape, type(c)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        #print c.shape, type(c)
        #print 'haha1'
        
        q_repr = self.q_net(q_emb) #[batch * 5 ,hid_dim]
        v_repr = self.v_net(v_emb)#[batch *5, hid_dim]
        #print c.shape, type(c)
        # v_repr = v_repr.unsqueeze(1).repeat(1,5,1).view(batch*5,-1)
        joint_repr = q_repr * v_repr #[batch *5,hid_dim ]

        logits = self.classifier(joint_repr)
        #print c.shape, type(c)
        
        rc_w_emb = self.caption_w_emb(c)
        
        #print c.shape, type(c)

        v_rc = self.v2rc_net(v.mean(1))
        v_qc = self.v2qc_net(joint_repr)

        rc_emb = self.reference_caption_rnn( v_rc,rc_w_emb)
        #[b,5,21,hid_dim]
        
        rc_repr = self.caption_decoder(rc_emb)
        #qc_repr = self.caption_decoder(qc_emb)


        pred_ans = F.sigmoid(logits).contiguous()
        pred_rc = F.sigmoid(rc_repr).contiguous()
        #print 'haha2'
        
        batch = batch / 5
        
        caption_from_ans =  pred_rc[:, : , : 3129 ]
        # [b*5, 20, 3129]

        caption_from_ans = caption_from_ans.contiguous().view(batch, 1 ,5, 20, -1).repeat(1,5,1,1,1)
        # [batch ,5, 5, 20, caption set ]
        #print 'haha3'
        
        similarities_ = (caption_from_ans * (pred_ans.view(batch, 5,1,1,-1).repeat(1, 1, 5, 20, 1))).sum(4)
        # [batch, 5, 5, 20] [i,j] i th answer with j th caption
        
        similarities, _ = similarities_.max(3)
        # [batch ,5, 5]
        _, indices = similarities.max(2)
        # [batch, 5 ] 
        indices = indices.view(-1,1 )
            #[batch, 5]
        #print 'haha3.5'
        target_qc_mask = torch.zeros(batch*5, 5)
        #print target_qc_mask.shape, indices.data.shape
        target_qc_mask.scatter_(1, indices.data.type(torch.LongTensor), 1)
        #print 'haha5'
        #target_qc_mask = Variable(target_qc_mask.view(batch, 5, 5, 1).repeat(1,1,1,20), volatile=True).cuda()
        target_qc_mask = Variable(target_qc_mask.view(batch, 5, 5, 1).repeat(1,1,1,20).type(torch.LongTensor)).cuda()
        # [b, 5, 5, 20]
        #print 'haha6'

        target_qc = c.view(batch,1,5,20).repeat(1,5,1,1)
        # [b,5,5, 20]
        #print 'haha7'
        target_qc = target_qc * target_qc_mask
        #print 'haha8'
        target_qc = target_qc.sum(2).view(-1, 20)
        # []
        #print 'haha9'
        qc_w_emb = self.caption_w_emb(target_qc) # [batch * 5, 20 , hid_dim]
        #print 'haha10'
        qc_emb = self.question_caption_rnn(v_qc ,qc_w_emb)
        #print 'haha11'
        qc_repr = self.caption_decoder(qc_emb)
        #print 'haha12'
        pred_qc = F.sigmoid(qc_repr).contiguous()

            
        
        return logits, pred_rc, pred_qc, target_qc
        # all three returns are logits
class Model_3(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier):
        super(Model_3, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.v_att_3 = v_att_3
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]

        att_1 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_3 = self.v_att_3(v, q_emb)  # [batch, 1, v_dim]
        att = att_1 + att_2 + att_3
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

# Attn: 1 layer attention, output layer, softmax
def build_baseline(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_0(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 2 layer attention, output layer, softmax
def build_model_A1(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_1(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_2(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
def build_model_A3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_3(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# Attn: 1 layer seperate, element-wise *, 1 layer, output layer, sigmoid
def build_model_A3S(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_3S(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
# our adopted model
# (self, in_dim, num_hid, v_dim, nlayers, bidirect, dropout, rnn_type='LSTM'):
# (self, embed_size, hidden_size, vocab_size, num_layers):
def build_model_A3x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)
    # num_hid = 1280 , dataset.v_dim = 2048
    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
        
    v2rc_net = FCNet([dataset.v_dim, 300 ], dropout= dropL, norm= norm, act= activation)
    v2qc_net = FCNet([num_hid, 300], dropout= dropL, norm= norm, act= activation)

    caption_w_emb = WordEmbedding(dataset.caption_dictionary.ntoken, emb_dim=300, dropout=dropW)
    reference_caption_decoder = CaptionRNN(300, 512,  num_layers = 1 )
    question_caption_decoder = CaptionRNN(300, 512, num_layers = 1 )
    caption_decoder = SimpleClassifier( in_dim=512, hid_dim=2 * num_hid, out_dim= dataset.caption_dictionary.ntoken, dropout=dropC, norm= norm, act= activation)
    return Model_4(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,caption_w_emb, reference_caption_decoder, question_caption_decoder, caption_decoder,v2rc_net, v2qc_net)

# 2*Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_2(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier)


def build_model_A23P(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_P(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)

# 3*Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
def build_model_A3x3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)

# 3*Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2x3(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_3 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_3(w_emb, q_emb, v_att_1, v_att_2, v_att_3, q_net, v_net, classifier)


# Attn: 1 layer seperate, element-wise *, output layer
def build_model_AP(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, output layer
def build_model_APx2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    v_att_2 = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model_2(w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier)


# Attn: 2 layer seperate, element-wise *, output layer
def build_model_APD(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_PD(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_AP_PC(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1=300, hid_dim_2=2048, out_dim=dataset.num_ans_candidates, dropout=dropC, norm=norm,
        act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_P_exact(dataset, num_hid, dropout, norm, activation):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=0.0)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=0, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = GTH(q_emb.num_hid, num_hid, dropout=0, norm=norm, act=activation)
    v_net = GTH(dataset.v_dim, num_hid, dropout=0, norm=norm, act=activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1= 300, hid_dim_2= 2048, out_dim=dataset.num_ans_candidates, dropout=0, norm=norm, act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_model_P_mod(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_P(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = GTH(q_emb.num_hid, num_hid, dropout=dropL, norm=norm, act=activation)
    v_net = GTH(dataset.v_dim, num_hid, dropout=dropL, norm=norm, act=activation)

    classifier = PaperClassifier(
        in_dim=num_hid, hid_dim_1= 300, hid_dim_2= 2048, out_dim=dataset.num_ans_candidates, dropout=dropC, norm=norm, act=activation)
    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)
