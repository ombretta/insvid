#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:31:19 2020

@author: ombretta
"""

import os
import sys

if '../' not in sys.path: sys.path.append('../')
if "../../" not in sys.path: sys.path.append("../../")
if "../../kinetics_i3d_pytorch/" not in sys.path: sys.path.append("../../kinetics_i3d_pytorch/")

from simple_classifier import *

import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision

import numpy as np
import pickle as pkl
import math


import CrossTaskdataset 
from args import parse_args

import random
import argparse

import FocalLoss


class NewDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y, num_classes):
        
        data = [{} for i in range(y.shape[0])]
        
        # labels = np.zeros([y.shape[0], y.shape[1], num_classes])
        labels = np.zeros([y.shape[0], y.shape[1], 1])
        
        for data_point in range(y.shape[0]):
            labels[data_point, :, :] = int(y[data_point][0])
            # labels[data_point, :, int(y[data_point][0])] = 1
    
        for i in range(y.shape[0]):
            data[i] = {'X': X[i], 'y': labels[i]} 
        
        self.X = X
        self.y = labels
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    

def fill_data(videos, videos_features, videos_classes, dim_features, classes_labels, max_seq_length=50, downsample_features=False):
    
    print("Loading data...")
    
    videos_with_problems = []    
    
    X_sequences_to_pad = [None]*len(videos)
    y_sequences_to_pad = [None]*len(videos)
    
    seq_length, i = 0, 0
    
    for video in videos:
        
        # print(video)
        
        features = torch.tensor(videos_features[video][:,:dim_features])
        rows = features.shape[0]
        
        features[torch.isnan(features)]=0
        features[torch.abs(features)>10]=10
        # features = features/10 #data between 0 and 1
        
        # print(features.shape)

        if rows == 0: 
            print("Removed video with problems:", video)
            videos_with_problems.append(video)
            X_sequences_to_pad = X_sequences_to_pad[:len(X_sequences_to_pad)-1]
            y_sequences_to_pad = y_sequences_to_pad[:len(y_sequences_to_pad)-1]
        else:
            
            if downsample_features == True:
                features = downsample_i3d_features(features.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
                rows = features.shape[0]
            # print(features.shape)
                
            # if rows > seq_length: seq_length = rows
            
            y = torch.ones(min(rows,max_seq_length), 1)*classes_labels[videos_classes[video]]
            
            j = 0
            while j < rows: 
                if j > 0: 
                    X_sequences_to_pad = X_sequences_to_pad + [None]
                    y_sequences_to_pad = y_sequences_to_pad + [None]
                X_sequences_to_pad[i] = features[j:min(j+rows,j+max_seq_length)]
                y_sequences_to_pad[i] = torch.tensor(y)
                i += 1
                j += max_seq_length
                
    
    # In case videos are shorter than max_seq_length
    # X = pad_sequence(X_sequences_to_pad, batch_first=True) # 0 padding
    X = pad_sequence(X_sequences_to_pad, batch_first=True, padding_value=0.5)
    y = pad_sequence(y_sequences_to_pad, batch_first=True, padding_value=-1) # Pad with -1 to not be confused with class 0
    
    print(X.size())
    print(y.size())

    return X, y, seq_length, videos_with_problems


def downsample_i3d_features(features):
    
    # print(features.shape)

    kernel_size = 10
    
    avg_pooling = torch.nn.AvgPool1d(kernel_size)
    
    downsampled_features = avg_pooling(features) if features.shape[2]>=kernel_size else features
    
    return downsampled_features
        
        

def downsample_training_videos(train_videos, videos_classes, videos_per_class, classes_labels):
    
    print("Downsampling training videos...")
    downsampled_train_videos = []
    mini_videos_per_class = {}
    
    for video in train_videos: 
        if videos_classes[video] not in mini_videos_per_class and videos_classes[video] in classes_labels:
            # print("Added", videos_classes[video])
            mini_videos_per_class[videos_classes[video]] = []
        if videos_classes[video] in mini_videos_per_class: 
            mini_videos_per_class[videos_classes[video]].append(video)
    
    for c in mini_videos_per_class:
        
        videos = mini_videos_per_class[c]
        print(c, len(videos)) 
        random.shuffle(videos)
        downsampled_train_videos += videos[:min(len(videos),50)] # sample max 50 videos per class
    
    return downsampled_train_videos


def upsample_training_videos(train_videos, videos_classes, videos_per_class, classes_labels):
    
    print("Upsampling training videos...")
    upsampled_train_videos = []
    mini_videos_per_class = {}
    
    for video in train_videos: 
        if videos_classes[video] not in mini_videos_per_class and videos_classes[video] in classes_labels:
            # print("Added", videos_classes[video])
            mini_videos_per_class[videos_classes[video]] = []
        if videos_classes[video] in mini_videos_per_class: 
            mini_videos_per_class[videos_classes[video]].append(video)
    
    for c in mini_videos_per_class:
        
        videos = mini_videos_per_class[c]
        print(c, len(videos)) 
        
        upsampled_train_videos += videos
        
        i = 1
        while len(videos)*i < 150: # at least 150 videos per class  
            random.shuffle(videos)
            upsampled_train_videos += videos[:min(len(videos),max(0,150-len(videos)*i))] # sample max 50 videos per class
            i += 1
            
    return upsampled_train_videos


def compose_dataset(num_classes, train_ratio, test_ratio, valid_ratio, frequencies, \
                    videos_per_class, videos_features, videos_classes, classes=[], \
                        train_videos=[], test_videos=[], valid_videos=[], \
                            downsample=False, upsample=False, max_seq_length=50, downsample_features = False):
    
    classes_labels = {}
    
    print("Training videos", len(train_videos))
    
    if len(train_videos) == 0 or len(classes)==0: 
        train_videos, test_videos, valid_videos = [], [], []
    
    if len(classes)!=0: selected_keys = classes
    else:
        if num_classes < len(list(videos_per_class.keys())):
            selected_keys = random.sample([k for k in videos_per_class.keys() if frequencies[k]<=30], num_classes) #previously ==30
        else: selected_keys = list(videos_per_class.keys())
    
    for i, count in zip(list(selected_keys), range(num_classes)):
        classes_labels[i] = count
        
        if len(train_videos) == 0 or len(classes)==0: # Assuming also other sets are empty
            videos = videos_per_class[i]
            # for v in videos: videos_classes[v]=i 
            random.shuffle(videos)
            train_videos += videos[:round(train_ratio*len(videos))]
            test_videos += videos[round(train_ratio*len(videos)):round(train_ratio*len(videos))+round(test_ratio*len(videos))]
            valid_videos += videos[round(train_ratio*len(videos))+round(test_ratio*len(videos)):]
        
    # downsample  
    if downsample == True:
        train_videos = downsample_training_videos(train_videos, videos_classes, videos_per_class, classes_labels)           
        
    # upsample  
    if upsample == True:
        train_videos = upsample_training_videos(train_videos, videos_classes, videos_per_class, classes_labels)           
        
    print("Classes keys and labels", selected_keys, classes_labels)
    print("Number of training videos:", len(train_videos))
#    
    train_rows = sum([videos_features[v].shape[0] for v in train_videos])
    test_rows = sum([videos_features[v].shape[0] for v in test_videos])
    valid_rows = sum([videos_features[v].shape[0] for v in valid_videos])
     
    X_train, y_train, train_seq_length, videos_with_problems_train = fill_data(train_videos, videos_features, videos_classes, 1024, classes_labels, max_seq_length, downsample_features)
    X_test, y_test, test_seq_length, videos_with_problems_test = fill_data(test_videos, videos_features, videos_classes, 1024, classes_labels, max_seq_length, downsample_features)
    X_valid, y_valid, valid_seq_length, videos_with_problems_valid = fill_data(valid_videos, videos_features, videos_classes, 1024, classes_labels, max_seq_length, downsample_features)
    
    [train_videos.remove(v) for v in videos_with_problems_train]
    [test_videos.remove(v) for v in videos_with_problems_test]
    [valid_videos.remove(v) for v in videos_with_problems_valid]
    
    return  X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, classes_labels, \
        train_seq_length, test_seq_length, valid_seq_length, train_videos, test_videos, valid_videos
        

        
def load_videos_sets(output_path):
    if os.path.exists(output_path+"/train_videos.dat"):
        valid_videos = load_data(output_path+"/valid_videos.dat")
        test_videos = load_data(output_path+"/test_videos.dat")
        train_videos = load_data(output_path+"/train_videos.dat") 
        return train_videos, test_videos, valid_videos
    return [], [], []
    

def print_gradients(model):
    
    # LSTM parameters
    model.lstm.weight_ih_l0.register_hook(lambda grad: print("Grad weight_ih_l0", grad[0])) 
    model.lstm.weight_hh_l0.register_hook(lambda grad: print("Grad weight_hh_l0", grad[0])) 
    model.lstm.bias_ih_l0.register_hook(lambda grad: print("Grad bias_ih_l0", grad[0])) 
    model.lstm.bias_hh_l0.register_hook(lambda grad: print("Grad bias_hh_l0", grad[0])) 
    
    # FC parameters
    for np in model.named_parameters():
        print(np[1].register_hook(lambda grad: print(grad[:3])))
    
    
def train_epoch(train_loader, model, optimizer, criterion, num_classes):
    
    running_loss, total_train, correct_train = 0.0, 0, 0
    
    batch_count = 0
    for batch in enumerate(train_loader):
        inputs = batch[1]['X'].cuda() if torch.cuda.is_available() else batch[1]['X']
        labels = batch[1]['y'].long().cuda() if torch.cuda.is_available() else batch[1]['y'].long()
        
        # print("inputs", inputs.shape)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs) #.view(-1, num_classes)
        
        print(batch_count, "Inputs", inputs[0,0,:3])
        print(batch_count, "Outputs", outputs[0,0,:3])
        
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1, 1).squeeze(0)) # Input, Target
        print("Loss", loss.data)
        
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(e, "Exception occurred, stopping training.")
            break
        
        _, predictions = torch.max(torch.exp(outputs).data, 2)
        running_loss += loss.item()
        total_train += labels.shape[0]*labels.shape[1]
        correct_train += predictions.eq(labels.squeeze()).sum().item()  
        
        batch_count += 1
        
        # if batch_count > 10: break # REMOVE IN THE FUTURE
    
    # Training accuracies and losses 
    # print("Avg training loss", running_loss/total_train)
    # print("Avg training accuracy", correct_train/total_train)
    
    return running_loss/total_train, correct_train/total_train, model


def validation(valid_loader, model, num_classes):
                
    running_valid_loss, total_valid, correct_valid = 0, 0, 0
    
    with torch.no_grad():
        for batch in enumerate(valid_loader):
    
            with torch.autograd.detect_anomaly():
                inputs = batch[1]['X'].cuda() if torch.cuda.is_available() else batch[1]['X']
                labels = batch[1]['y'].long().cuda() if torch.cuda.is_available() else batch[1]['y'].long()
                
                outputs = model(inputs) #.view(-1, num_classes)
                loss = criterion(outputs.view(-1, num_classes), labels.view(-1, 1).squeeze()) # Input, Target
                _, predictions = torch.max(torch.exp(outputs).data, 2)
                
                running_valid_loss += loss.item()
                total_valid += labels.shape[0]*labels.shape[1]
                correct_valid+= predictions.eq(labels.squeeze()).sum().item()  
            
    return running_valid_loss/total_valid, correct_valid/total_valid 


def save_training_results(output_path, training_accuracies, training_losses, validation_accuracies, validation_losses):
    save_data(training_accuracies, output_path+"training_accuracies.dat", with_torch=True)
    save_data(training_losses, output_path+"training_losses.dat", with_torch=True)
    save_data(validation_accuracies, output_path+"validation_accuracies.dat", with_torch=True)
    save_data(validation_losses, output_path+"validation_losses.dat", with_torch=True)

    
class LSTMclassifier(torch.nn.Module):

    def __init__(self, D_in, H1, H2, D_out, num_layers):
        
        super(LSTMclassifier, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(D_in, H1, batch_first = True, num_layers=num_layers)
        
        # Parameters initialization
        for name, param in self.lstm.named_parameters():
          if 'bias' in name:
             torch.nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             torch.nn.init.xavier_normal_(param, gain=torch.nn.init.calculate_gain('relu'))
        

        # The linear layer that maps from hidden state space to tag space
        self.linear = torch.nn.Linear(H1, H2)
        self.softmax = torch.nn.Softmax(dim=2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        #lstm_out = tensor containing the next hidden state for each element in the sequence, foe each element in the batch
        # hn: tensor containing the next hidden state for each element in the batch
        
        # print("lstm_out", lstm_out[0,0,:3])
        # print("lstm_out", lstm_out.shape)
        # print("hn", hn.shape)
        
        #Newly added:
        # hidden = torch.tanh(self.linear(lstm_out)) #nm nothing changed
        
        hidden = self.linear(lstm_out) 
        
        # print("hidden", hidden[0,0,:3])
        # print("hidden", hidden.shape)
        
        logits = torch.tanh(hidden) 
        
        # print("logits", logits[0,0,:3])
        # print("logits", logits.shape)
        
        predictions = self.logsoftmax(logits)
        
        # print("predictions", predictions[0,0,:3])
        # print("predictions", predictions.shape)
        
        return predictions
    
    
    
    
if __name__ == "__main__":
    
    # Load features extracted with pytorch i3d
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    args = get_arguments()
    
    # dataset_path = args.dataset_path
    # num_classes = args.num_classes
    # features_type = args.features_type
    # specific_classes = args.specific_classes
    # balance = args.balance
    # max_seq_length = args.max_seq_length
    # downsample = args.downsample
    # downsample_features = args.downsample_features
    
    LSTM_layers = args.LSTM_layers 
    
    dataset_path = args.dataset_path
    num_classes = 83
    features_type = "sw"
    specific_classes = False
    balance = True
    downsample = False
    upsample = False
    
    max_seq_length = 50000
    downsample_features = True
    
    
    #%%
            
    frequencies, _, _, _, all_tasks_info = initialize("full")
    
    #%%
    
    videos_per_class, videos_classes = get_videos_classes(dataset_path)
    videos_features = get_features(dataset_path, features_type)
    
    train_ratio = 0.7 # at least 21 videos per class
    test_ratio = 0.2 # at least 6 videos per class
    valid_ratio = 0.1 # at least 3 videos per class
    
    # root = "./torch/" #cluster
    root = "./LSTM/many_to_many/downsampled_features/"

    
    if balance: output_path = root+"balanced/"+features_type+"_"+str(num_classes)+"/"
    else: output_path = root+features_type+"_"+str(num_classes)+"/"
    
    classes = load_classes(specific_classes_path) if specific_classes else []
    if num_classes == 83: classes =  list(all_tasks_info["title"].keys())
    
    train_videos, test_videos, valid_videos = load_videos_sets(output_path)

    X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, \
            classes_labels, train_seq_length, test_seq_length, valid_seq_length, train_videos, test_videos, valid_videos = \
                compose_dataset(num_classes, train_ratio, test_ratio, valid_ratio, frequencies, \
                    videos_per_class, videos_features, videos_classes, classes, \
                        train_videos, test_videos, valid_videos, downsample, upsample, max_seq_length, downsample_features)
    
#%%
    
    if not os.path.exists(output_path): os.mkdir(output_path)
    output_path = output_path+str(LSTM_layers)+"LSTMlayers/"
    if not os.path.exists(output_path): os.mkdir(output_path)  
    output_path = output_path+"max_len_"+str(max_seq_length)+"/"
    if not os.path.exists(output_path): os.mkdir(output_path)
    
    print("Output path:", output_path)
    print("Number of classes:", num_classes)
    
    save_data(classes_labels, output_path+"classes_labels.dat")
    save_data(train_videos, output_path+"train_videos.dat")
    save_data(valid_videos, output_path+"valid_videos.dat")
    save_data(test_videos, output_path+"test_videos.dat")
    
    train_data = NewDataset(X_train, y_train.numpy(), num_classes)
    valid_data = NewDataset(X_valid, y_valid.numpy(), num_classes)
    test_data = NewDataset(X_test, y_test.numpy(), num_classes)
    
    batch_size = 50 if num_classes != 83 else 25 #reduced batch size to fix error: RuntimeError: CUDA out of memory
    
    D_in, H1, H2, D_out = 1024, 100, num_classes, 1

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = False)
    #%%
    
    weights = get_classes_weights(classes_labels, videos_classes, train_videos, balance)
    print("Classes weights:", weights)
    
    training_accuracies, training_losses = {}, {}
    validation_accuracies, validation_losses = {}, {}
    
    for lr in [1e-3]:#, 1e-3, 1e-2, 1e-1, 1, 10, 100]: #[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        
        # if num_classes < 20 or lr >= 0.1:
        if num_classes <= 83 or lr >= 0.1: # Always true
        
            print("learning rate", lr)
            
            training_accuracies[lr] = []
            training_losses[lr] = []
            validation_accuracies[lr] = []
            validation_losses[lr] = []
        
            model = LSTMclassifier(D_in, H1, H2, D_out, LSTM_layers)
            
            print(torch.cuda.is_available())
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            
            # criterion = torch.nn.NLLLoss(weights) # changed to NLLLoss from CrossEntropyLoss to match LogSoftmax!
            criterion = FocalLoss.FocalLoss(weight=weights, gamma=2., reduction='mean')
            
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train model
            
            epochs = 1 #100
            best_valid_acc = 0
            
            for epoch in range(epochs):  # loop over the dataset multiple times
                
                print("Epoch", epoch)
                
                training_loss, training_acc, model = train_epoch(train_loader, model, optimizer, criterion, num_classes)
                
                print("Avg training loss", training_loss)
                print("Avg training accuracy", training_acc)
                
                # Validation accuracies and losses 
                
                validation_loss, validation_acc = validation(valid_loader, model, num_classes)
                
                print("Validation loss", validation_loss)
                print("Validation accuracy", validation_acc)
                
                training_accuracies[lr].append(training_acc)
                training_losses[lr].append(training_loss)
                validation_accuracies[lr].append(validation_acc)
                validation_losses[lr].append(validation_loss)
                
                if validation_acc >= best_valid_acc: 
                    torch.save(model.state_dict(), output_path+"model_"+str(lr)) # Early stopping
                    best_valid_acc = validation_acc
            
            print('Finished Training')
        
            # torch.cuda.empty_cache() # to avoid RuntimeError: CUDA out of memory. when training for 83 classes
    
    save_training_results(output_path, training_accuracies, training_losses, validation_accuracies, validation_losses)


#%%
    
# # Stats
    
# lengths = []
    
# for video in valid_videos:
#     lengths.append(len(videos_features[video]))
#     print(len(videos_features[video]))
    
    
# plt.figure()
# plt.hist(lengths)#, bins=9)
# plt.title(str(len(valid_videos))+" validation videos")
# plt.ylabel('Number of videos')
# plt.xlabel('Duration (s)')
# plt.savefig("valid_videos_duration.png")
    
    