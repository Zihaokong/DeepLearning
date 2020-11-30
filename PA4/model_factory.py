################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torchvision.models as models
import torch.nn as nn
import torch
# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_size = len(vocab) 
        
    
    model = LSTM_RNN(hidden_size, embedding_size, vocab_size)
    
    
    return model



class LSTM_RNN(nn.Module):

    def __init__(self, hidden_size,embedding_size,vocab_size):
        super(LSTM_RNN, self).__init__()
        
        #encoder
        self.conv = models.resnet50(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.conv.fc = nn.Linear(self.conv.fc.in_features,embedding_size)
        
        #decoder
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
    
        
    def forward(self, features, captions):
        captions = captions[:,:-1]
        conv = self.conv(features).unsqueeze(1)
        embedding = self.embedding(captions)
        out = torch.cat((conv, embedding),dim = 1)
        hidden_states, (self.hn, self.cn) = self.lstm(out)
        
        line = self.linear(hidden_states)
        return line
    
    def generate(self, images,vocab,max_length = 20):
        result = []
        states = None
        with torch.no_grad():
            x = self.conv(images).unsqueeze(1)
            print(x.shape)
            for i in range(max_length):
                hidden, states = self.lstm(x,states)
                output = self.linear(hidden)
                batch_predicted = output.argmax(2)
                result.append(batch_predicted)
                x = self.embedding(batch_predicted)
        return result