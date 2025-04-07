import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixing the randomness of CUDA.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(42)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, membership_num):
        super(FuzzyLayer, self).__init__()

        # input_dim: feature number of the dataset
        # membership_num: number of membership function, also known as the class number

        self.input_dim = input_dim
        self.membership_num = membership_num

        self.membership_miu = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)
        self.membership_sigma = nn.Parameter(torch.Tensor(self.membership_num, self.input_dim).to(DEVICE), requires_grad=True)

        nn.init.xavier_uniform_(self.membership_miu)
        nn.init.ones_(self.membership_sigma)

    def forward(self, input_seq):
        batch_size = input_seq.size()[0]
        input_seq_exp = input_seq.unsqueeze(1).expand(batch_size, self.membership_num, self.input_dim)
        membership_miu_exp = self.membership_miu.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)
        membership_sigma_exp = self.membership_sigma.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)

        fuzzy_membership = torch.mean(torch.exp((-1 / 2) * ((input_seq_exp - membership_miu_exp) / membership_sigma_exp) ** 2), dim=-1)
        return fuzzy_membership



class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=1)
        self.rnn = nn.LSTM(32, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, metadata):
        #metadata = [batch size, metadata dim]

        embedded = self.dropout(self.embedding(metadata))
        #embedded = [batch size, metadata dim, emb dim]

        embedded = torch.reshape(embedded, (metadata.size(0), 128, 1))

        conved = F.relu(self.conv(embedded))
        #conved = [batch size, n_filters, metadata dim - filter_sizes[n] + 1]

        conved = torch.reshape(conved, (metadata.size(0), 32))

        outputs, (hidden, cell) = self.rnn(conved)
        #outputs = [metadata dim - filter_sizes[n] + 1, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        # hidden = self.dropout(torch.cat((hidden[-1,:], hidden[0,:]), dim = -1))
        #hidden = [batch size, hid dim * num directions]

        return self.fc(outputs)


class LiarModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional):
        super().__init__()

        self.textcnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.justification_cnn = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.textcnn2 = TextCNN(vocab_size, input_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx)
        self.cnn_bilstm = CNNBiLSTM(input_dim_metadata, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.fuzzy = FuzzyLayer(output_dim, output_dim)
        self.fuse = nn.Linear(output_dim * 5, output_dim)
    
    def forward(self, text, metadata_text, metadata_number, justification):
        #text = [batch size, sent len]
        #metadata = [batch size, metadata dim]

        text_output = self.textcnn(text)
        metadata_output_text = self.textcnn2(metadata_text)
        metadata_output_number = self.cnn_bilstm(metadata_number)
        metadata_output_fuzzy = self.fuzzy(metadata_output_number)
        justification_output = self.justification_cnn(justification)

        fused_output = self.fuse(torch.cat((text_output, metadata_output_text, metadata_output_number, metadata_output_fuzzy, justification_output), dim=1))

        return fused_output