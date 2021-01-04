import torch

class SentimentRNN(torch.nn.Module):
    

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, device = torch.device('cpu')):
        
        super(SentimentRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        
        # define all layers
        
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first = True)
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        # return last sigmoid output and hidden state
        sig_out = torch.sigmoid(self.fc1(x))
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out.type(torch.FloatTensor).to(device), hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initialization of hidden states '''

        # Create two new tenzors of shape [n_layers, batch_size, hidden_dim]
        # for hidden states and cell states of LSTM cell
        # Set all created tenzors to zero and (if necessary) send to GPU

        hidden_state = torch.zeros(*[self.n_layers, batch_size, self.hidden_dim]).to(self.device)
        cell_state = torch.zeros(*[self.n_layers, batch_size, self.hidden_dim]).to(self.device)

        hidden = (hidden_state, cell_state)

        return hidden