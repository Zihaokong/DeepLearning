################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import torch.nn as nn
import copy
from PIL import Image
import nltk
from pycocotools.coco import COCO


# helper function, view caption from a list of index
def generate_caption(vocab,captions):
    string = ""
    for word in captions:
        string = string + vocab.idx2word[word.item()] + " "
    return string

# view caption from output
def view_sent(output,vocab):
    predicted = []
    string = ""
    for i in output:
        wordidx = torch.argmax(i).item()
        string += vocab.idx2word[wordidx] + " "
    print(string)

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        self.config = config_data
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        self.coco_test, self.vocab, self.train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
        
        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        self.model = get_model(config_data, self.vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=config_data["experiment"]["learning_rate"])
      

        
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        self.lowest_val_loss = 99999999
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(epoch,"number epoch")
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                print("best model")
                self.__best_model = copy.deepcopy(self.model.cpu())
                self.model.to("cuda")
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.model.train()
        training_loss = 0
        total_num = 0
        
        for i, (images, captions, _) in enumerate(self.train_loader):
            # send input to GPU
            images = images.to("cuda")
            captions = captions.to("cuda")
            
            # train model
            self.__optimizer.zero_grad()
            output = self.model(images,captions)
            loss = self.__criterion(output.view(-1,len(self.vocab)),captions.view(-1))
            # calculate one pass loss 
            training_loss+=loss         
            loss.backward()
            self.__optimizer.step()
            total_num = i
            
            if i % 200 == 0:
                print("train loss: ", loss.item())
        # average one pass loss
        return training_loss.item()/total_num

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.model.eval()
        self.model.to("cuda")
        val_loss = 0
        total_num = 0
        with torch.no_grad():
            # calculate validation loss 
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to("cuda")
                captions = captions.to("cuda")

                output = self.model(images,captions)
                loss = self.__criterion(output.reshape(-1,len(self.vocab)),captions.reshape(-1))
                val_loss+=loss
                
                total_num = i
        return val_loss.item()/total_num


       
    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    
    
    def test(self):
        if self.__best_model is None:
            self.__best_model = copy.deepcopy(self.model.cpu())
        self.__best_model.eval()
        self.__best_model.to("cuda")
        test_loss = 0
        bleu1_score = 0
        bleu4_score = 0
        total_num = 0
   
        write_file = ""

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to("cuda")
                captions = captions.to("cuda")
                self.__best_model = self.__best_model.to("cuda")
                # calculate test set loss using teacher forcing  
                output = self.__best_model(images,captions)
                loss = self.__criterion(output.reshape(-1,len(self.vocab)),captions.reshape(-1))
                test_loss += loss
                total_num = iter
                
                
                # generate captions for a minibatch

                predicted = self.__best_model.generate(images,self.vocab,self.config["generation"]["temperature"],self.config["generation"]["max_length"],self.config["generation"]["deterministic"])
                # total batch bleu1 and bleu4 score
                batch_bleu1 = 0
                batch_bleu4 = 0
                
                
                
                # generate every picture's reference captions
                for i in range(len(images)):
                    # reference
                    captions = []
                    image = images[i]
                    img_id = img_ids[i]
                    # using unique image id, extract 5 different references captions from coco test set
                    for j in range(5):
                        cap = self.coco_test.imgToAnns[img_id][j]['caption']
                        tokens = nltk.tokenize.word_tokenize(str(cap).lower())
                        captions.append(tokens)
                    
                    # extract the self generated caption from a batch of predictions
                    # and get rid of <start> <end> <unknown> and <pad>
                    captions_self = []
                    for word in predicted:
                        if word[i] >=0 and word[i] <= 3:
                            continue
                        captions_self.append(self.vocab.idx2word[word[i].item()])
                    
                    # calculate for every picture its bleu scores
                    

                    bleu1pic = bleu1(captions, captions_self)
                    bleu4pic = bleu4(captions, captions_self)
                    if i == 1 or i == 30:
                        write_file += "self {} filename {} ref {}, bleu1 [{}], bleu4 [{}]".format(captions_self, self.coco_test.loadImgs(img_id)[0]['file_name'],captions[0],bleu1pic, bleu4pic)+"\n"                   
                    
                    batch_bleu1 += bleu1pic
                    batch_bleu4 += bleu4pic
                    
                # calculate for every batch it's bleu scores        
                batch_bleu1 /= self.config["dataset"]['batch_size']
                batch_bleu4 /= self.config["dataset"]['batch_size']
                print("batch: ",iter,"bleu1 score is ",batch_bleu1,"bleu4 score is ",batch_bleu4, "loss is ",loss)
                bleu1_score += batch_bleu1
                bleu4_score += batch_bleu4
        # calculate total bleu score        
        bleu1_score /= len(self.__test_loader)
        bleu4_score /= len(self.__test_loader)
        test_loss = test_loss.item()/total_num
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,bleu1_score,bleu4_score)
        self.__log(result_str)
        
        
        f = open("captions.txt", "a")
        f.write(write_file)
        f.close()

        return test_loss, bleu1_score, bleu4_score

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
