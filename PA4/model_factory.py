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
        
    
    model = LSTM_RNN(hidden_size, embedding_size, vocab_size,model_type)
    
    
    return model



class LSTM_RNN(nn.Module):

    def __init__(self, hidden_size,embedding_size,vocab_size,model_type):
        super(LSTM_RNN, self).__init__()
        self.model_type = model_type
        #encoder: transfer learning with fully connected layer the embedding size
        self.conv = models.resnet50(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.conv.fc = nn.Linear(self.conv.fc.in_features,embedding_size)
        
        #decoder: word embedding layer, LSTM hidden layer and linear prediction layer
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        
        if model_type == "LSTM":
            print("model LSTM baseline")
            self.lstm = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        elif model_type == "Vanilla":
            print("model Vanilla RNN")
            self.lstm = nn.RNN(embedding_size,hidden_size,batch_first=True)
        elif model_type == "LSTM2":
            print("model LSTM architecture 2")
            self.lstm = nn.LSTM(embedding_size*2,hidden_size,batch_first=True)
        
        self.linear = nn.Linear(hidden_size,vocab_size)
    
        
    def forward(self, features, captions):
        # get rid of the last teaching signal 
        if self.model_type == "LSTM2":
            padding = torch.zeros([captions.shape[0],1],dtype=torch.long).to("cuda")
            
            # cutting the last <end>, padding first with <pad>
            captions = torch.cat((padding, captions),dim = 1)[:,:-1]
            conv = self.conv(features).unsqueeze(1)
            
            # duplicate conv features to max_seq_len
            conv_padd = torch.cat((captions.shape[1])*[conv],1)
            embedding = self.embedding(captions)
            out = torch.cat((conv_padd,embedding),2)
            
            hidden_states, self.hn = self.lstm(out)

            # output word prediction at every timestep
            return self.linear(hidden_states)            

        else :
            captions = captions[:,:-1]

            # add a dimension from (batch size, num_features) to (batch_size, time_step, num_features)
            conv = self.conv(features).unsqueeze(1)

            # word embedding input captions (batch_size, max_seq_len, num_features)
            embedding = self.embedding(captions)

            # concatenate features from picture and features of words.
            out = torch.cat((conv, embedding),dim = 1)

            # generate prediction using teacher forcing
            hidden_states, self.hn = self.lstm(out)

            # output word prediction at every timestep
            return self.linear(hidden_states)
        
    
    # generate captions without teacher forcing
    def generate(self, images,vocab,temp,max_length, deterministic):
        # manually add softmax layer
        layer = nn.Softmax(dim = 2)
        
        # a list(list(str)), for every list, it's a prediction at a time step
        # of every batch 
        result = []
        
        # initailize states to be all 0
        states = None
        with torch.no_grad():
            #pass in the image feature to LSTM
            img = self.conv(images).unsqueeze(1)
            if self.model_type == "LSTM2":
                padding = torch.zeros([images.shape[0],1],dtype=torch.long).to("cuda")
                padding = self.embedding(padding)
                x = torch.cat((img,padding),2)
            else:
                x = img
                
            for i in range(max_length):
                #generate the <start>
                hidden, states = self.lstm(x,states)
                output = self.linear(hidden)
                # softmaxout with temp
                
                if deterministic == True:
                    output = layer(output)
                    output_word = output.argmax(2)

                else:                      
                    output = layer(output/temp)
                    # draw from distribution
                    output_word = torch.multinomial(output.flatten(1),1, replacement=True)
                result.append(output_word)
                # feed the generated word as input to LSTM again
                x = self.embedding(output_word)
                if self.model_type == "LSTM2":
                    x = torch.cat((img,x),2)
                    
        return result
    
    
    
    
    
    