import torch
import torch.nn as nn
from transformers import (PreTrainedModel, AutoModel,AutoConfig)
import math
import random
import numpy as np
from transformers import AutoModelWithLMHead


class BiLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTM, self).__init__() 
        #self.setup_seed(seed)
        self.forward_lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=1, bidirectional=False, batch_first=True)
        self.backward_lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers=1, bidirectional=False, batch_first=True)
    
    def forward(self, x):
        batch_size,max_len,feat_dim = x.shape
        out1, (h1,c1) = self.forward_lstm(x)
        reverse_x = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            reverse_x[:,i,:] = x[:,max_len-1-i,:]
                
        out2, (h2,c2) = self.backward_lstm(reverse_x)

        output = torch.cat((out1, out2), 2)
        return output,(1,1)


class HGNER(nn.Module):
    def __init__(self, args, num_labels,hidden_dropout_prob=0.1,windows_list=None):
        super(HGNER, self).__init__()


        config = AutoConfig.from_pretrained(args.bert_model)
        self.bert = AutoModel.from_pretrained(args.bert_model)


        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_labels = num_labels


        self.use_bilstm = args.use_bilstm


        self.use_multiple_window = args.use_multiple_window
        self.windows_list = windows_list
        self.connect_type = args.connect_type
        connect_type = args.connect_type
        self.d_model = args.d_model
        self.num_labels = num_labels


        if self.use_multiple_window and self.windows_list != None:
            if self.use_bilstm:
                self.bilstm_layers = nn.ModuleList([BiLSTM(self.d_model) for _ in self.windows_list])

            else:
                self.bilstm_layers = nn.ModuleList([nn.LSTM(self.d_model, self.d_model, num_layers=1, bidirectional=False, batch_first=True) for _ in self.windows_list])

            if connect_type=='dot-att':
                self.linear = nn.Linear(self.d_model, self.num_labels)
            elif connect_type=='mlp-att':
                self.linear = nn.Linear(self.d_model, self.num_labels)
                self.Q = nn.Linear(self.d_model * (len(windows_list) + 1), self.d_model)
        else:
            self.linear = nn.Linear(self.d_model, self.num_labels)


    def windows_sequence(self,sequence_output, windows, lstm_layer):
        batch_size, max_len, feat_dim = sequence_output.shape
        local_final = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            index_list = []
            for u in range(1, windows // 2 + 1):
                if i - u >= 0:
                    index_list.append(i - u)
                if i + u <= max_len - 1:
                    index_list.append(i + u)
            index_list.append(i)
            index_list.sort()
            temp = sequence_output[:, index_list, :]
            out,(h,b) = lstm_layer(temp)
            local_f = out[:, -1, :]
            local_final[:, i, :] = local_f
        return local_final



    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):

        sequence_output = self.bert(input_ids, token_type_ids= token_type_ids, attention_mask=attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')

        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)


        if self.use_multiple_window:
            mutiple_windows = []

            for i,window in enumerate(self.windows_list):
                if self.use_bilstm:
                    local_final = self.windows_sequence(sequence_output, window, self.bilstm_layers[i])
                mutiple_windows.append(local_final)


            if self.connect_type=='dot-att':
                muti_local_features = torch.stack(mutiple_windows, dim=2)
                sequence_output = sequence_output.unsqueeze(dim=2)
                d_k = sequence_output.size(-1)
                attn = torch.matmul(sequence_output, muti_local_features.permute(0, 1, 3, 2)) / math.sqrt(d_k)
                attn = torch.softmax(attn, dim=-1)
                local_features = torch.matmul(attn, muti_local_features).squeeze()
                sequence_output = sequence_output.squeeze()
                sequence_output = sequence_output + local_features
            elif self.connect_type == 'mlp-att':
                mutiple_windows.append(sequence_output)
                muti_features = torch.cat(mutiple_windows, dim=-1)
                muti_local_features = torch.stack(mutiple_windows, dim=2)
                query = self.Q(muti_features)
                d_k = query.size(-1)
                query = query.unsqueeze(dim=2)
                attn = torch.matmul(query, muti_local_features.permute(0, 1, 3, 2)) / math.sqrt(d_k)
                attn = torch.softmax(attn, dim=-1)
                sequence_output = torch.matmul(attn, muti_local_features).squeeze()


        logits = self.linear(sequence_output)
        
        if labels is not None:
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            
            return logits