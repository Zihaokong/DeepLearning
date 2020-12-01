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
        
        #encoder: transfer learning with fully connected layer the embedding size
        self.conv = models.resnet50(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.conv.fc = nn.Linear(self.conv.fc.in_features,embedding_size)
        
        #decoder: word embedding layer, LSTM hidden layer and linear prediction layer
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        #self.lstm = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        
        
        self.lstm = nn.RNN(embedding_size,hidden_size,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
    
        
    def forward(self, features, captions):
        # get rid of the last teaching signal 
        captions = captions[:,:-1]
        
        # add a dimension from (batch size, num_features) to (batch_size, time_step, num_features)
        conv = self.conv(features).unsqueeze(1)
        
        # word embedding input captions (batch_size, max_seq_len, num_features)
        embedding = self.embedding(captions)
        
        # concatenate features from picture and features of words.
        out = torch.cat((conv, embedding),dim = 1)
        
        # generate prediction using teacher forcing
        #hidden_states, (self.hn, self.cn) = self.lstm(out)
        
        # vanilla RNN
        hidden_states, self.hn = self.lstm(out)
        
        # output word prediction at every timestep
        return self.linear(hidden_states)
        
    
    # generate captions without teacher forcing
    def generate(self, images,vocab,temp,max_length = 20):
        # manually add softmax layer
        layer = nn.Softmax(dim = 2)
        
        # a list(list(str)), for every list, it's a prediction at a time step
        # of every batch 
        result = []
        
        # initailize states to be all 0
        states = None
        with torch.no_grad():
            #pass in the image feature to LSTM
            x = self.conv(images).unsqueeze(1)
            for i in range(max_length):
                #generate the <start>
                hidden, states = self.lstm(x,states)
                output = self.linear(hidden)
                # temperature 
                output = layer(output/temp)
                batch_predicted = output.argmax(2)
                result.append(batch_predicted)
                
                # feed the generated word as input to LSTM again
                x = self.embedding(batch_predicted)
        return result
    
    
    
    
    
    