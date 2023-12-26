import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn

# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()
        
        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(vocab_size, rnn_size)
                        # shape (vocab_size, rnn_size)의 임베딩 행렬 생성

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, 
                            num_layers=num_layers, dropout=dropout)
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(rnn_size)

        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        self.dropout = nn.Dropout(dropout)

        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(rnn_size, vocab_size)   
                        # in_feature 입력차원이 rnn_size
                        # out_feature 출력차원이 vocab_size
        
    def forward(self,x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # Apply batch normalization to the LSTM output
        lstm_out = lstm_out.permute(0, 2, 1)  # Adjust dimensions for batch normalization
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # Restore original dimensions
        
        lstm_out = self.dropout(lstm_out)
        logits = self.output(lstm_out)

        return logits
