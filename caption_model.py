import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CaptionDecoderRNN(nn.Module):
    def __init__(self, in_dim, num_hid, v_dim, rnn_type='LSTM'):
        """Module for question embedding
        """
        super(CaptionDecoderRNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type == 'LSTM' else nn.GRUCell
        
        self.rnn_v = rnn_cls(
            in_dim + num_hid + v_dim, num_hid)

        self.rnn_c = rnn_cls(
            num_hid + in_dim , num_hid)
        
        self.v_a = nn.Linear(v_dim, num_hid)
        self.h_a = nn.Linear(num_hid, num_hid)
        self.a = nn.Linear(num_hid, 1)
        self.att_softmax = nn.Softmax(dim = 2)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.rnn_type = rnn_type

    def forward(self, xx ,features ):
    	# in_dim = v_dim
        # xx: [batch * 5, sequence 20 , in_dim ] captions
        # features: [batch * 5, 36 , v_dim]
        x = torch.cat((torch.mean(features, 1, keepdim = True) , xx), dim = 1)
        # [batch* 5, 21 , in_dim ]
        batch = x.size(0) 
        hidden_v = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        memory_v = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        hidden_c = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        memory_c = Variable(torch.zeros(batch ,  self.num_hid)).cuda()
        #hidden = self.init_hidden(batch)
        #self.rnn.flatten_parameters()
        repeat_feat = features #(torch.unsqueeze(features, 1)).repeat(1,5,1,1)
        #[b* 5, 36, v_dim]
        outputs = []
        #print repeat_feat.shape
        for i in range(20):
            #print torch.mean(repeat_feat, 1).shape, hidden_c.shape, x[:,i,:].shape
            input = torch.cat( (torch.mean(repeat_feat, 1), hidden_c, x[:,i,:]) , dim = 1)
            # [b * 5, v_dim + num_hid + v_dim] 
            hidden_v, memory_v = self.rnn_v(input, (hidden_v, memory_v))
            # [b * 5 , num_hid]
            va = self.v_a(repeat_feat) #[b *5, 36 , 512]
            ha = self.h_a(hidden_v) #[b*5, 512]
            ha = ha.unsqueeze(1).repeat(1,36,1)
            att = self.a(torch.tanh(va + ha)) # [b*5,36,1]
            att = self.att_softmax(att) # [b*5,36,1]

            features_hat = torch.mean(repeat_feat * att.repeat(1,1,repeat_feat.size(2)),1)
            #[b*5, v_dim]
            input_c = torch.cat((features_hat, hidden_v), 1 )
            hidden_c, memory_c = self.rnn_c(input_c, (hidden_c, memory_c))
            outputs.append(hidden_c)
            #[20, b*5, 512]
            
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0 ,1 )
        return outputs

class QuestionCaptionDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(QuestionCaptionDecoderRNN, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths = 20):
        """Decode image feature vectors and generates captions."""
        #features [b*5, v_dim]
        #captions [b*5, 20, hid_dim]
        batch = features.size(0)
        embeddings = captions
        #[b* 5,20,v_dim]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        #[b * 5,21,v_dim]
        self.lstm.flatten_parameters()
        hiddens, _ = self.lstm(embeddings[:,:-1,:])
        #hiddens = hiddens.view(batch,5,21,-1)
        
        outputs = hiddens #self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()


class CaptionRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN, self).__init__()
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths = 20):
        """Decode image feature vectors and generates captions."""
        #features [b*5, v_dim]
        #captions [b*5, 20, hid_dim]
        batch = features.size(0)
        embeddings = captions
        #[b* 5,20,v_dim]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        #[b * 5,21,v_dim]
        self.lstm.flatten_parameters()
        hiddens, _ = self.lstm(embeddings[:,:-1,:])
        #hiddens = hiddens.view(batch,5,21,-1)
        
        outputs = hiddens #self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()
