# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/model.py ]
#   Synopsis     [ Implementation of downstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import init
#from audtorch.metrics.functional import concordance_cc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
###########################
# FEED FORWARD CLASSIFIER #
###########################
class FeedForwardClassifier(nn.Module):
    def __init__(self, input_dim, class_num, dconfig):
        super(FeedForwardClassifier, self).__init__()
        
        # init attributes
        self.input_dim = input_dim
        self.concat = dconfig['concat']
        self.linear = dconfig['linear']
        self.num_layers = dconfig['layers']
        if self.concat > 1:
            assert(self.concat % 2 == 1) # concat must be an odd number
            self.input_dim *= self.concat

        # process layers
        linears = []
        for i in range(self.num_layers):
            if i == 0 and self.num_layers == 1:
                linears.append(nn.Linear(self.input_dim, class_num)) # single layer
            elif i == 0 and self.num_layers > 1:
                linears.append(nn.Linear(self.input_dim, dconfig['hidden_size'])) # input layer of num_layer >= 2
            elif i == self.num_layers - 1 and self.num_layers > 1:
                linears.append(nn.Linear(dconfig['hidden_size'], class_num)) # output layer of num_layer >= 2
            else: 
                linears.append(nn.Linear(dconfig['hidden_size'], dconfig['hidden_size'])) # middle layer
        self.linears = nn.ModuleList(linears)
        assert self.num_layers == len(self.linears)

        if not self.linear:
            self.drop = nn.Dropout(p=dconfig['drop'])
            self.act_fn = torch.nn.functional.relu
            
        self.out_fn = nn.LogSoftmax(dim=-1) # Use LogSoftmax since self.criterion combines nn.LogSoftmax() and nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)


    def _statistic(self, probabilities, labels, label_mask):
        assert(len(probabilities.shape) > 1)
        assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)
        assert(labels.shape == label_mask.shape)

        valid_count = label_mask.sum()
        correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.cuda.LongTensor) * label_mask).sum()
        return correct_count, valid_count


    def _roll(self, x, n, padding='same'):
        # positive n: roll around to the right on the first axis. For example n = 2: [1, 2, 3, 4, 5] -> [4, 5, 1, 2, 3]
        # negative n: roll around to the left on the first axis. For example n = -2: [1, 2, 3, 4, 5] -> [3, 4, 5, 1, 2]
        assert(n != 0)

        if n > 0: # when n is positive (n=2),
            if padding == 'zero':  # set left to zero: [1, 2, 3, 4, 5] -> [0, 0, 1, 2, 3]
                left = torch.zeros_like(x[-n:])
            elif padding == 'same': # set left to same as last: [1, 2, 3, 4, 5] -> [1, 1, 1, 2, 3]
                left = x[0].repeat(n, 1)
            else: # roll over: [1, 2, 3, 4, 5] -> [4, 5, 1, 2, 3]
                left = x[-n:]
            right = x[:-n]

        elif n < 0: # when n is negative (n=-2), 
            if padding == 'zero': # set right to zero: [1, 2, 3, 4, 5] -> [3, 4, 5, 0, 0]
                right = torch.zeros_like(x[:-n])
            elif padding == 'same': # set right to same as last: [1, 2, 3, 4, 5] -> [3, 4, 5, 5, 5]
                right = x[-1].repeat(-n, 1)
            else: # roll over: [1, 2, 3, 4, 5] -> [3, 4, 5, 1, 2]
                right = x[:-n]
            left = x[-n:]
        else:
            raise ValueError('Argument \'n\' should not be set to 0, acceptable range: [-seq_len, 0) and (0, seq_len].')

        return torch.cat((left, right), dim=0)

    
    def _match_length(self, features, labels, label_mask):
        # since the down-sampling (float length be truncated to int) and then up-sampling process
        # can cause a mismatch between the seq lenth of transformer representation and that of label
        # we truncate the final few timestamp of label to make two seq equal in length
        truncated_length = min(features.size(1), labels.size(-1))
        features = features[:, :truncated_length, :]
        labels = labels[:, :truncated_length]
        label_mask = label_mask[:, :truncated_length]
        return features, labels, label_mask


    def forward(self, features, labels=None, label_mask=None):
        # features: (batch_size, seq_len, feature_dim)
        # labels: (batch_size, seq_len), frame by frame classification
        batch_size = features.size(0)
        seq_len = features.size(1)
        feature_dim = features.size(2)
        features, labels, label_mask = self._match_length(features, labels, label_mask)
        
        if self.concat > 1:
            features = features.repeat(1, 1, self.concat) # (batch_size, seq_len, feature_dim * concat)
            features = features.view(batch_size, seq_len, self.concat, feature_dim).permute(0, 2, 1, 3) # (batch_size, concat, seq_len, feature_dim)
            for b_idx in range(batch_size):
                mid = (self.concat // 2)
                for r_idx in range(1, mid+1):
                    features[b_idx, mid + r_idx, :] = self._roll(features[b_idx][mid + r_idx], n=r_idx)
                    features[b_idx, mid - r_idx, :] = self._roll(features[b_idx][mid - r_idx], n=-r_idx)
            features = features.permute(0, 2, 1, 3).view(batch_size, seq_len, feature_dim * self.concat) # (batch_size, seq_len, feature_dim * concat)

        hidden = features
        for i, linear_layer in enumerate(self.linears):
            hidden = linear_layer(hidden)
            if not self.linear and ((i+1) < self.num_layers): 
                hidden = self.drop(hidden)
                hidden = self.act_fn(hidden)

        logits = hidden
        prob = self.out_fn(logits)
        
        if labels is not None:
            assert(label_mask is not None), 'When frame-wise labels are provided, validity of each timestamp should also be provided'
            labels_with_ignore_index = 100 * (label_mask - 1) + labels * label_mask

            # cause logits are in (batch, seq, class) and labels are in (batch, seq)
            # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
            # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
            class_num = logits.size(-1)
            loss = self.criterion(logits.reshape(-1, class_num), labels_with_ignore_index.reshape(-1))
            
            # statistic for accuracy
            correct, valid = self._statistic(prob, labels, label_mask)

            return loss, prob.detach().cpu(), correct.detach().cpu(), valid.detach().cpu()

        return prob


##################
# RNN CLASSIFIER #
##################
class RnnClassifier(nn.Module):
    def __init__(self, input_dim, class_num, dconfig, seed):
        # The class_num for regression mode should be 1

        super(RnnClassifier, self).__init__()
        torch.manual_seed(seed)
        self.config = dconfig
        self.dropout = nn.Dropout(p=dconfig['drop'])

        linears = []
        last_dim = input_dim
        for linear_dim in self.config['pre_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.pre_linears = nn.ModuleList(linears)

        hidden_size = self.config['hidden_size']
        self.rnn = None
        if hidden_size > 0:
            self.rnn = nn.GRU(input_size=last_dim, hidden_size=hidden_size, num_layers=2, dropout=dconfig['drop'],
                            batch_first=True, bidirectional=False)
            last_dim = hidden_size
            

        linears = []
        for linear_dim in self.config['post_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.post_linears = nn.ModuleList(linears)

        self.act_fn = torch.nn.functional.relu
        self.out = nn.Linear(last_dim, class_num)
        
        self.mode = self.config['mode']
        if self.mode == 'classification':
            self.out_fn = nn.LogSoftmax(dim=-1)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        elif self.mode == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Only classification/regression modes are supported')
        
        if(self.mode == "regression"):
            for layer in self.pre_linears:
                init.uniform(layer.weight, -10, 10)
                
            for layer in self.post_linears:
                init.uniform(layer.weight, -20, 20)
                
            for p in self.rnn.parameters():
                if(p.dim() > 1):
                    init.uniform(p, -1, 1)
        elif(self.mode == "classifier"):
            for layer in self.pre_linears:
                init.uniform(layer.weight, -1, 1)
                
            for layer in self.post_linears:
                init.uniform(layer.weight, -1, 1)
                
            for p in self.rnn.parameters():
                if(p.dim() > 1):
                    init.uniform(p, -1, 1)            
            
        

    def statistic(self, probabilities, labels):
        assert(len(probabilities.shape) > 1)
        assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)

        valid_count = torch.LongTensor([len(labels)])
        correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.LongTensor)).sum()
        return correct_count, valid_count


    def forward(self, features, labels=None, valid_lengths=None):
        assert(valid_lengths is not None), 'Valid_lengths is required.'
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label
        # valid_lengths: (batch_size, )
        seq_len = features.size(1)

        if self.config['sample_rate'] > 1:
            features = features[:, torch.arange(0, seq_len, self.config['sample_rate']), :]
            valid_lengths = valid_lengths // self.config['sample_rate']

        for linear in self.pre_linears:
            features = linear(features)
            features = self.act_fn(features)
            features = self.dropout(features)

        if self.rnn is not None:
            packed = pack_padded_sequence(features, valid_lengths, batch_first=True, enforce_sorted=False)
            _, h_n = self.rnn(packed)
            hidden = h_n[-1, :, :]
            # cause h_n directly contains info for final states
            # it will be easier to use h_n as extracted embedding
        else:
            hidden = features.mean(dim=1)
        
        for linear in self.post_linears:
            hidden = linear(hidden)
            hidden = self.act_fn(hidden)
            hidden = self.dropout(hidden)

        logits = self.out(hidden)

        if self.mode == 'classification':
            result = self.out_fn(logits)
            # result: (batch_size, class_num)
        elif self.mode == 'regression':
            result = logits.reshape(-1)
            # result: (batch_size, )
        else:
            raise NotImplementedError
        
        if labels is not None:
            if self.mode == 'classification':
                loss = self.criterion(hidden, labels)
            elif self.mode == 'regression':
                loss1 = torch.sqrt(self.criterion(result, labels)) 
                loss2 = concordance_cc(result, labels)
                #loss = -torch.abs(loss2) + loss1/10000
                #loss = -torch.abs(loss2)
                loss = loss1

            # statistic for accuracy
            if self.mode == 'classification':
                correct, valid = self.statistic(result, labels)
            elif self.mode == 'regression':
                # correct and valid has no meaning when in regression mode
                # just to make the outside wrapper can correctly function
                correct, valid = torch.LongTensor([1]), torch.LongTensor([1])

            return loss, result.detach().cpu(), correct, valid

        return result
    
class RnnClassifierMosi(nn.Module):
    def __init__(self, input_dim, class_num, dconfig, seed):
        # The class_num for regression mode should be 1

        super(RnnClassifierMosi, self).__init__()
        torch.manual_seed(seed)
        self.config = dconfig
        self.dropout = nn.Dropout(p=dconfig['drop'])

        linears = []
        last_dim = input_dim
        for linear_dim in self.config['pre_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.pre_linears = nn.ModuleList(linears)

        hidden_size = self.config['hidden_size']
        self.rnn = None
        if hidden_size > 0:
            self.rnn = nn.GRU(input_size=last_dim, hidden_size=hidden_size, num_layers=2, dropout=dconfig['drop'],
                            batch_first=True, bidirectional=False)
            last_dim = hidden_size
            

        linears = []
        for linear_dim in self.config['post_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.post_linears = nn.ModuleList(linears)

        self.act_fn = torch.nn.functional.relu
        self.out = nn.Linear(last_dim, class_num)
        
        self.mode = self.config['mode']
        if self.mode == 'classification':
            self.out_fn = nn.LogSoftmax(dim=-1)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        elif self.mode == 'regression':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError('Only classification/regression modes are supported')
        
        if(self.mode == "regression"):
            #for layer in self.pre_linears:
                #init.uniform(layer.weight, -1, 1)
                
            for layer in self.post_linears:
                init.uniform(layer.weight, -10, 10)
                
            #for p in self.rnn.parameters():
                #if(p.dim() > 1):
                    #init.uniform(p, -1, 1)
                    
            
        

    def statistic(self, probabilities, labels):
        assert(len(probabilities.shape) > 1)
        assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)

        valid_count = torch.LongTensor([len(labels)])
        correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.LongTensor)).sum()
        return correct_count, valid_count


    def forward(self, features, labels=None, valid_lengths=None):
        assert(valid_lengths is not None), 'Valid_lengths is required.'
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label
        # valid_lengths: (batch_size, )
        seq_len = features.size(1)

        if self.config['sample_rate'] > 1:
            features = features[:, torch.arange(0, seq_len, self.config['sample_rate']), :]
            valid_lengths = valid_lengths // self.config['sample_rate']

        for linear in self.pre_linears:
            features = linear(features)
            features = self.act_fn(features)
            features = self.dropout(features)

        if self.rnn is not None:
            packed = pack_padded_sequence(features, valid_lengths, batch_first=True, enforce_sorted=False)
            _, h_n = self.rnn(packed)
            hidden = h_n[-1, :, :]
            # cause h_n directly contains info for final states
            # it will be easier to use h_n as extracted embedding
        else:
            hidden = features.mean(dim=1)
        
        for linear in self.post_linears:
            hidden = linear(hidden)
            hidden = self.act_fn(hidden)
            hidden = self.dropout(hidden)

        logits = self.out(hidden)

        if self.mode == 'classification':
            result = self.out_fn(logits)
            # result: (batch_size, class_num)
        elif self.mode == 'regression':
            result = logits.reshape(-1)
            # result: (batch_size, )
        else:
            raise NotImplementedError
        
        if labels is not None:
            if self.mode == 'classification':
                loss = self.criterion(hidden, labels)
            elif self.mode == 'regression':
                loss1 = self.criterion(result, labels) / 10
                #loss2 = concordance_cc(result, labels)
                #loss = -torch.abs(loss2) + loss1/10000
                #loss = -torch.abs(loss2)
                vx = result - torch.mean(result)
                vy = labels - torch.mean(labels)
                
                cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))                
                loss = loss1 - cost

            # statistic for accuracy
            if self.mode == 'classification':
                correct, valid = self.statistic(result, labels)
            elif self.mode == 'regression':
                # correct and valid has no meaning when in regression mode
                # just to make the outside wrapper can correctly function
                correct, valid = torch.LongTensor([1]), torch.LongTensor([1])

            return loss, result.detach().cpu(), correct, valid

        return result

class AvecModel(nn.Module):
    def __init__(self, input_dim, class_num, dconfig, seed):
        super(AvecModel, self).__init__()
        '''contain two GRU, one for local segment of size n seconds,
        the other global GRU takes output of local GRU to make predictions
        '''
        torch.manual_seed(seed)    
        self.config = dconfig
        self.dropout = nn.Dropout(p=dconfig['drop'])
        self.t_local = dconfig['t_local']
        self.t_global = dconfig['t_global']

        linears = []
        last_dim = input_dim
        for linear_dim in self.config['pre_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.pre_linears = nn.ModuleList(linears)

        hidden_size = self.config['hidden_size']
        self.local_rnn = None
        self.global_rnn = None
        if hidden_size > 0:
            self.local_rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=2, dropout=dconfig['drop'],
                            batch_first=True, bidirectional=False)
            last_dim = hidden_size
            self.global_rnn = nn.GRU(input_size=last_dim, hidden_size=hidden_size, num_layers=2, dropout=dconfig['drop'],
                            batch_first=True, bidirectional=False)
            
        
        self.out = nn.Linear(hidden_size, 1)
        linears = []
        for linear_dim in self.config['post_linear_dims']:
            linears.append(nn.Linear(last_dim, linear_dim))
            last_dim = linear_dim
        self.post_linears = nn.ModuleList(linears)

        self.act_fn = torch.nn.functional.relu
        #self.out = nn.Linear(last_dim, class_num)
        
        self.mode = self.config['mode']
        if self.mode == 'classification':
            self.out_fn = nn.LogSoftmax(dim=-1)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        elif self.mode == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Only classification/regression modes are supported')  
        
        if(self.mode == "regression"):
            for layer in self.pre_linears:
                init.uniform(layer.weight, -1, 1)
                
            for layer in self.post_linears:
                init.uniform(layer.weight, -1, 1)
                
            for p in self.local_rnn.parameters():
                if(p.dim() > 1):
                    init.uniform(p, -1, 1)
                    
            for p in self.global_rnn.parameters():
                if(p.dim() > 1):
                    init.uniform(p, -1, 1)            
        elif(self.mode == "classifier"):
            for layer in self.pre_linears:
                init.uniform(layer.weight, -1, 1)
                
            for layer in self.post_linears:
                init.uniform(layer.weight, -1, 1)
                
            for p in self.rnn.parameters():
                if(p.dim() > 1):
                    init.uniform(p, -1, 1)          
        
    def forward(self, features, labels=None, valid_lengths=None):
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label
        # valid_lengths: (batch_size, )
        seq_len = features.size(1)
        local_length = self.config['t_local']
        global_length = self.config['t_global']
        batch_size = features.size(0)
        n_chunks = seq_len // local_length
        subsequences = []
        for i in range(n_chunks):
            tmp = features[:, i*local_length : (i+1)*local_length ,:]
            subsequences.append(tmp)

        pre_linear_outputs = []
        for i in range(n_chunks):
            current_feature = subsequences[i]
            #for linear in self.pre_linears:
                #current_feature = linear(current_feature)
                #current_feature = self.act_fn(current_feature)
                #current_feature = self.dropout(current_feature)
            pre_linear_outputs.append(current_feature)
            
        output_local_rnn = torch.zeros(batch_size, global_length, self.config['hidden_size'])
        for i in range(n_chunks):
            if(i >= global_length):
                continue
            current_feature = pre_linear_outputs[i]
            packed = pack_padded_sequence(current_feature, [local_length]*batch_size, batch_first=True, enforce_sorted=False)
            _, h_n = self.local_rnn(packed)
            hidden_local = h_n[-1, :, :] #batch_size x hidden_size
            for j in range(batch_size):
                output_local_rnn[j, i, :] = hidden_local[j, :]
                
        valid_length_global = [global_length] * batch_size
        packed_global = pack_padded_sequence(output_local_rnn.to(device), valid_length_global, batch_first=True, enforce_sorted=False)
        _, h_n = self.global_rnn(packed_global)
        hidden = h_n[-1, :, :] #batch_size x hidden_size        
            
        
        #for linear in self.post_linears:
            #hidden = linear(hidden)
            #hidden = self.act_fn(hidden)
            #hidden = self.dropout(hidden)

        logits = self.out(hidden)

        if self.mode == 'classification':
            result = self.out_fn(logits)
            # result: (batch_size, class_num)
        elif self.mode == 'regression':
            result = logits.reshape(-1)
            # result: (batch_size, )
        else:
            raise NotImplementedError
        
        if labels is not None:
            if self.mode == 'classification':
                loss = self.criterion(hidden, labels)
            elif self.mode == 'regression':
                loss = self.criterion(result, labels)

            # statistic for accuracy
            if self.mode == 'classification':
                correct, valid = self.statistic(result, labels)
            elif self.mode == 'regression':
                # correct and valid has no meaning when in regression mode
                # just to make the outside wrapper can correctly function
                correct, valid = torch.LongTensor([1]), torch.LongTensor([1])

            return loss, result.detach().cpu(), correct, valid

        return result    
    

##################
# DUMMY UPSTREAM #
##################
class dummy_upstream(nn.Module):
    def __init__(self, input_dim):
        super(dummy_upstream, self).__init__()
        self.out_dim = input_dim

    def forward(self, features):
        return features


######################
# EXAMPLE CLASSIFIER #
######################
class example_classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(example_classifier, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.1,
                          batch_first=True, bidirectional=True)
        
        self.out = nn.Linear(hidden_dim, class_num)
        self.out_fn = nn.LogSoftmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        
    def statistic(self, probabilities, labels):
        assert(len(probabilities.shape) > 1)
        assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)

        valid_count = torch.LongTensor([len(labels)])
        correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.LongTensor)).sum()
        return correct_count, valid_count    

    def forward(self, features, labels, valid_length=None):
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label

        _, h_n = self.rnn(features)
        hidden = h_n[-1, :, :]
        logits = self.out(hidden)
        result = self.out_fn(logits)
        loss = self.criterion(result, labels)
        correct, valid = self.statistic(logits, labels)
        return loss, result, correct, valid
    
    
class example_DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(example_DNN, self).__init__()
        layers = [64,64,64,16]
        self.network = nn.Sequential(
            nn.Linear(input_dim*100, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.ReLU()            
        )
        
        self.out = nn.Linear(layers[-1], class_num)
        self.out_fn = nn.LogSoftmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        
    def statistic(self, probabilities, labels):
        assert(len(probabilities.shape) > 1)
        assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)

        valid_count = torch.LongTensor([len(labels)])
        correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.LongTensor)).sum()
        return correct_count, valid_count    

    def forward(self, features, labels, valid_length=None):
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label
        features = features.view(features.shape[0],features.shape[1]*features.shape[2])
        hidden = self.network(features)
        logits = self.out(hidden)
        result = self.out_fn(logits)
        loss = self.criterion(result, labels)
        correct, valid = self.statistic(logits, labels)
        return loss, result, correct, valid
    
class example_regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(example_regression, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.1,
                          batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_dim, 1)
        self.criterion = nn.MSELoss()

    def forward(self, features, labels, valid_length=None):
        # features: (batch_size, seq_len, feature)
        # labels: (batch_size,), one utterance to one label

        _, h_n = self.rnn(features)
        hidden = h_n[-1, :, :]
        result = self.out(hidden)
        loss = self.criterion(result, labels)
        correct, valid = torch.LongTensor([1]), torch.LongTensor([1])
        return loss, result.detach().cpu(), correct, valid    