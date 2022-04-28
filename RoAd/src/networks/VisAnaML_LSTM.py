import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class Prediction_MLP(BaseNet):
    
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, int(input_size/2))
        self.layer2 = nn.Linear(int(input_size/2), int(input_size/2))
        self.layer3 = nn.Linear(int(input_size/2), 1)
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x

class VisAnaML_LSTM(BaseNet):
    def __init__(self, seq_len, n_features, rep_dim=64):
        super().__init__()
        self.rep_dim = rep_dim
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = rep_dim, 2 * rep_dim
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=rep_dim,
          num_layers=1,
          batch_first=True
        )
    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((-1, self.embedding_dim))
    
class VisAnaML_LSTM_Decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, seq_len, rep_dim=64, n_features=21):
        super().__init__()
        self.seq_len, self.input_dim = seq_len, rep_dim
        self.hidden_dim, self.n_features = 2 * rep_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=rep_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=self.n_features,
          num_layers=1,
          batch_first=True
        )
        # self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x = x.reshape((x.shape[1],-1,self.input_dim))
        x = x.repeat(1, self.seq_len, 1)
        x = x.reshape((-1, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1, self.seq_len, self.n_features))
        return x


class VisAnaML_LSTM_Autoencoder(BaseNet):
    def __init__(self, seq_len, n_features, rep_dim=64):
        super().__init__()
        self.encoder = VisAnaML_LSTM(seq_len, n_features, rep_dim)
        self.decoder = VisAnaML_LSTM_Decoder(seq_len, rep_dim, n_features)
        self.prediction = VisAnaML_LSTM_Decoder(seq_len, rep_dim, n_features)
    def forward(self, x):
        x = self.encoder(x)
        rec = self.decoder(x)
        pre = self.prediction(x)
        return rec, pre
