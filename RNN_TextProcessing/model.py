import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define all layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        

    def forward(self, x, hidden):
        
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initialization of hidden states '''

        # Create two new tenzors of shape [n_layers, batch_size, hidden_dim]
        # for two hidden states and state of LSTM cell
        # Set all created tenzors to zero and (if necessary) send to GPU


        
        return hidden