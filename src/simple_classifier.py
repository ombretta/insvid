#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:34:27 2020

@author: ombretta
"""

import os
import sys

if "../../" not in sys.path: sys.path.append("../../")
if "../../kinetics_i3d_pytorch/" not in sys.path: sys.path.append("../../kinetics_i3d_pytorch/")

import torch

import numpy as np
import pickle as pkl
import CrossTaskdataset 
from args import parse_args

import random
import math

import argparse


# This class generates a torch dataset that can be used to iterate batches 
# during training and testing
class NewDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y, num_classes):
        
        data = [{} for i in range(y.shape[0])]
        
        labels = np.zeros([y.shape[0], num_classes])
        
        for label,i in zip(y.astype(int), range(y.shape[0])):
            labels[i,label] = 1
    
        for i in range(y.shape[0]):
            data[i] = {'X': X[i], 'y': labels[i]} 
            
        self.X = X
        self.y = labels
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


#Command line arguments definition
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_classes',
        type=int,
        default='2',
        help='Number of tasks (classes) to classify.')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="../../pytorch_model/features/sw/",
        help='Path to the features folder')
    parser.add_argument(
        '--features_type',
        type=str,
        default='sw',
        help='sw or is')
    parser.add_argument(
        '--specific_classes',
        type=bool,
        default=False,
        help='Compose dataset with specific classes or random classes (default).')
    parser.add_argument(
        '--specific_classes_path', 
        type=str,
        default='./coffee_case/',
        help='If --specific_classes=True, path where to find the classes labels list file.')
    parser.add_argument(
        '--balance',
        type=bool,
        default=False,
        help='Use weighted CEloss if the dataset is unbalanced (default:False).')
    parser.add_argument(
        '--temporal_aggregation',
        type=bool,
        default=False,
        help='Aggregate features by taking the mean over certain time window (default:False).')
    parser.add_argument(
        '--aggregation_window',
        type=int,
        default=10,
        help='If temporal_aggregation=True, aggregation time window size in seconds (default:10).')
    parser.add_argument(
        '--temporal_window',
        type=int,
        default=50,
        help='Length of the video sequence input to the LSTM (default:50).')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=50000,
        help='Max sequence length of the LSTM input (default:50000, take length of the longest video).') 
    parser.add_argument(
        '--LSTM_layers',
        type=int,
        default=1,
        help='Number of LSTM layers (default:1).') 
    parser.add_argument(
        '--downsample',
        type=bool,
        default=False,
        help='Whether to downsample the training videos to have max 50 videos for all classes (default:False).') 
    parser.add_argument(
        '--upsample',
        type=bool,
        default=False,
        help='Whether to upsample the training videos to have at least 150 videos for all classes (default:False).') 
    parser.add_argument(
        '--downsample_features',
        type=bool,
        default=False,
        help='Whether to downsample the video features to reduce LSTM input sequence length (default:False).') 
    args = parser.parse_args()
    return args


# Extract the CrossTask dataset structure from the original CrossTask repository 
def extract_dataset(dataset, videos_features, videos_classes, frequencies, videos_per_class):
    for batch in dataset:
        for sample in batch:
            if sample['vid'] not in videos_features:
                key = sample['task']
                videos_features[sample['vid']] = sample['X']
                
                videos_classes[sample['vid']] = key
                frequencies[key] = 1 if key not in frequencies else frequencies[key]+1
                if key not in videos_per_class: videos_per_class[key] = [sample['vid']]
                else: videos_per_class[key].append(sample['vid'])  
            else: print("Duplicate detected!")
    return frequencies, videos_per_class, videos_features, videos_classes


# Aggregates i3d features in time by averaging the seconds that fall in the same aggregation_window 
def aggregate_features(videos_features, aggregation_window):
    rows = videos_features.shape[0]
    dim_features = videos_features.shape[1]
    X = np.zeros([rows, dim_features], dtype=np.float32)
    # X[0,:] = videos_features[0,:]
    for i in range(0,rows):
        X[i,:] = np.mean(videos_features[max(0,i-aggregation_window):i+1,:], 0)
    return X


# Construct feature matrices for a specific data set
def fill_data(videos, videos_features, videos_classes, n_rows, dim_features, classes_labels, temporal_aggregation=False, aggregation_window=0):
    
    # print(videos, videos_features, videos_classes, n_rows, dim_features, classes_labels)
    
    print("Loading data...")
    
    X = np.zeros([n_rows, dim_features], dtype=np.float32)
    y = np.zeros([n_rows, 1], dtype=np.float32)
    i = 0
    
    for video in videos:
        
        # print(video)
        
        rows = videos_features[video].shape[0]
        
        if temporal_aggregation:
            X[i:i+rows,:] = aggregate_features(videos_features[video][:,:dim_features], aggregation_window)
        else:
            X[i:i+rows,:] = videos_features[video][:,:dim_features]
        
        
        y[i:i+rows] = classes_labels[videos_classes[video]]
        i += rows
        
    X[np.isnan(X)] = 0
    y[np.isnan(y)] = 0

    return X,y


# Compose CrossTask training, test, validation sets
def compose_dataset(num_classes, train_ratio, test_ratio, valid_ratio, frequencies, videos_per_class, videos_features, videos_classes, temporal_aggregation=False, aggregation_window=0):
    train_videos, test_videos, valid_videos = [], [], []
    
#    sorted_keys = list(frequencies.keys())
#    sorted_keys.Sorted(key=lambda k:frequencies[k])
    classes_labels = {}
    
    if num_classes < len(list(videos_per_class.keys())):
        selected_keys = random.sample([k for k in videos_per_class.keys() if frequencies[k]<=30], num_classes) #previously ==30
    else: selected_keys = list(videos_per_class.keys())
    
    for i, count in zip(list(selected_keys), range(num_classes)):
        classes_labels[i] = count
        videos = videos_per_class[i]
        random.shuffle(videos)
        train_videos += videos[:round(train_ratio*len(videos))]
        test_videos += videos[round(train_ratio*len(videos)):round(train_ratio*len(videos))+round(test_ratio*len(videos))]
        valid_videos += videos[round(train_ratio*len(videos))+round(test_ratio*len(videos)):]
        
    print("Classes keys and labels", selected_keys, classes_labels)

    train_rows = sum([videos_features[v].shape[0] for v in train_videos])
    test_rows = sum([videos_features[v].shape[0] for v in test_videos])
    valid_rows = sum([videos_features[v].shape[0] for v in valid_videos])
     
    X_train, y_train = fill_data(train_videos, videos_features, videos_classes, train_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    X_test, y_test = fill_data(test_videos, videos_features, videos_classes, test_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    X_valid, y_valid = fill_data(valid_videos, videos_features, videos_classes, valid_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    
    return  X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, classes_labels, \
        train_videos, test_videos, valid_videos


# Same ad compose dataset but with predefined classes 
def compose_dataset_with_specific_classes(classes, train_ratio, test_ratio, valid_ratio, videos_per_class, videos_features, videos_classes, temporal_aggregation=False, aggregation_window=10):
    
    num_classes = len(classes)
    selected_keys = classes

    train_videos, test_videos, valid_videos = [], [], []
    classes_labels = {}

    for i, count in zip(list(selected_keys), range(num_classes)):
        classes_labels[i] = count
        videos = videos_per_class[i]
        random.shuffle(videos)
        train_videos += videos[:round(train_ratio*len(videos))]
        test_videos += videos[round(train_ratio*len(videos)):round(train_ratio*len(videos))+round(test_ratio*len(videos))]
        valid_videos += videos[round(train_ratio*len(videos))+round(test_ratio*len(videos)):]
    
    print("Classes keys and labels", selected_keys, classes_labels)

    train_rows = sum([videos_features[v].shape[0] for v in train_videos])
    test_rows = sum([videos_features[v].shape[0] for v in test_videos])
    valid_rows = sum([videos_features[v].shape[0] for v in valid_videos])
     
    X_train, y_train = fill_data(train_videos, videos_features, videos_classes, train_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    X_test, y_test = fill_data(test_videos, videos_features, videos_classes, test_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    X_valid, y_valid = fill_data(valid_videos, videos_features, videos_classes, valid_rows, 1024, classes_labels, temporal_aggregation, aggregation_window)
    
    return  num_classes, X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, classes_labels, \
        train_videos, test_videos, valid_videos


# Loads info and data about CrossTask from the original CrossTask repository
def initialize(dataset_size):

    # Assumes availability of CrossTask repository folder and assumes it's location.
    # Assumes all necessary parameters (e.g. folders path) are saved in args or received in command line.

    args = parse_args()

    [trainloader, testloader, A, M, all_tasks_info] = CrossTaskdataset.load_cross_task_dataset(args)
    
    frequencies = {}
    videos_per_class = {}
    videos_features = {}
    videos_classes = {}
    
    if dataset_size == 'full':
        frequencies, videos_per_class, videos_features, videos_classes = extract_dataset(trainloader, videos_features, videos_classes, frequencies, videos_per_class)
    frequencies, videos_per_class, videos_features, videos_classes = extract_dataset(testloader, videos_features, videos_classes, frequencies, videos_per_class)
    
    return frequencies, videos_per_class, videos_features, videos_classes, all_tasks_info


# Returns dictionary with keys=video_ids and values=true_class_id
def get_videos_classes(features_path):
    videos_per_class = {}
    videos_classes = {}
    for file in [f for f in os.listdir(features_path) if ".npy" in f]:
        key = file.split("__")[0]
        video_url = file.split("__")[1].split(".npy")[0]
        if key not in videos_per_class: videos_per_class[key] = []
        if video_url not in videos_classes:
            videos_per_class[key].append(video_url)
            videos_classes[video_url] = key
        else: print("Duplicate detected!")
    return videos_per_class, videos_classes    


# Returns extracted i3d features for CrossTask videos contained in dataset_path
def get_features(dataset_path, features_type):
    print("loading features...")
    if features_type != "sw" and features_type != "is": 
        print("Invalid feature type")
        return 
    
    video_features = {}
    
    for file in [f for f in os.listdir(dataset_path) if ".npy" in f]:
        video_url = file.split("__")[1].split(".npy")[0]
        features = np.load(dataset_path+file, allow_pickle=True) #, encoding='bytes', allow_pickle=True).item()
        video_features[video_url] = features
        
    return video_features


# Returns weights for each class calculated as #train_videos/#n_videos_per_class
def get_classes_weights(classes_labels, videos_classes, train_videos, balance=False):
    weights = torch.ones([len(classes_labels)])
    if not balance:
        return weights.cuda() if torch.cuda.is_available() else weights
    for c,i in zip(classes_labels, range(len(classes_labels))):
        n_videos = len([v for v in train_videos if videos_classes[v]==c])
        weights[i] = len(train_videos)/n_videos
        # weights[i] = math.pow(len(train_videos)/n_videos,2)
    return weights.cuda() if torch.cuda.is_available() else weights


# Load classes names to compose dataset with specific classes
# Assumes presence of txt file in specific_classes_path
def load_classes(specific_classes_path):
    with open(specific_classes_path+"classes.txt", "r") as file:
        classes = file.read().split(" ")
        return classes


# Function to load data specified in filename (=path+filename) 
def load_data(filename, with_torch = False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else: print("File", filename, "does not exists.")


# Function to save data to filename (=path+filename) 
def save_data(data, filename, with_torch=False):
    with open(filename, "wb") as f:
        if with_torch == True: torch.save(data, f)
        else: pkl.dump(data, f)


# Model with fully connected layer for classification 
class OneLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(OneLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        logits = torch.tanh(self.linear1(x)) # changed to avoid exploding loss 
        predictions = self.logsoftmax(logits) # changed to match NNNLoss
        # predictions = logits #remove softmax to match cross entropy loss        
        return predictions
    
if __name__ == "__main__":
    
    # Load features extracted with pytorch i3d
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    args = get_arguments()
    
    dataset_path = args.dataset_path
    num_classes = args.num_classes
    features_type = args.features_type
    specific_classes = args.specific_classes
    balance = args.balance
    temporal_aggregation = args.temporal_aggregation
    aggregation_window = args.aggregation_window
            
    
    #Load CrossTask dataset data and features
    frequencies, _, _, _, all_tasks_info = initialize("full")
    
    videos_per_class, videos_classes = get_videos_classes(dataset_path)
    
    
    videos_features = get_features(dataset_path, features_type)
    
    train_ratio = 0.7 # at least 21 videos per class
    test_ratio = 0.2 # at least 6 videos per class
    valid_ratio = 0.1 # at least 3 videos per class
    
    # root = "./torch/" #cluster
    root = "../data/"
    
    
    # Prepare dataset and output path
    if not specific_classes:
        X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, \
            classes_labels, train_videos, test_videos, valid_videos = \
            compose_dataset(num_classes, train_ratio, test_ratio, valid_ratio, frequencies, videos_per_class, videos_features, videos_classes, temporal_aggregation, aggregation_window)
        # output_path = "./torch/"+features_type+"_"+str(num_classes)+"/"
        if balance: output_path = root+"balanced/"+features_type+"_"+str(num_classes)+"/"
        else: output_path = root+features_type+"_"+str(num_classes)+"/"

    else:
        specific_classes_path = args.specific_classes_path
        classes = load_classes(specific_classes_path)
        num_classes, X_train, y_train, X_test, y_test, X_valid, y_valid, train_rows, test_rows, valid_rows, \
            classes_labels, train_videos, test_videos, valid_videos = \
            compose_dataset_with_specific_classes(classes, train_ratio, test_ratio, valid_ratio, videos_per_class, videos_features, videos_classes, temporal_aggregation, aggregation_window)
        if balance: output_path = specific_classes_path+"balanced/"+features_type+"/"
        else: output_path = specific_classes_path+features_type+"/"
    
    if not os.path.exists(output_path): os.mkdir(output_path)
    
    print("Output path:", output_path)
    
    print("Number of classes:", num_classes)
    
    
    # Save constructed dataset
    save_data(classes_labels, output_path+"classes_labels.dat")
    save_data(train_videos, output_path+"train_videos.dat")
    save_data(valid_videos, output_path+"valid_videos.dat")
    save_data(test_videos, output_path+"test_videos.dat")
    
    # Prepare for training
    train_data = NewDataset(X_train, y_train, num_classes)
    valid_data = NewDataset(X_valid, y_valid, num_classes)
    test_data = NewDataset(X_test, y_test, num_classes)
    
    batch_size = 50 if num_classes != 83 else 25 #reduced batch size to fix error: RuntimeError: CUDA out of memory
    D_in, H, D_out = 1024, num_classes, 1
    
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    
    weights = get_classes_weights(classes_labels, videos_classes, train_videos, balance)
    print("Classes weights:", weights)
    
    training_accuracies, training_losses = {}, {}
    validation_accuracies, validation_losses = {}, {}
    
    
    # Train one model for each learning rate parameter
    # for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
    for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        
        if num_classes < 20 or lr >= 0.1:
        
            print("learning rate", lr)
            
            training_accuracies[lr] = []
            training_losses[lr] = []
            validation_accuracies[lr] = []
            validation_losses[lr] = []
        
            model = OneLayerNet(D_in, H, D_out)
            
            print(torch.cuda.is_available())
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            
            criterion = torch.nn.NLLLoss(weights) # changed to NLLLoss from CrossEntropyLoss to match LogSoftmax!
            
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)#1e-1)#, lr=1e-4)
            
            # Train model
            
            epochs = 10
            
            for epoch in range(epochs):  # loop over the dataset multiple times
                
                print("Epoch", epoch)
                
                running_loss = 0.0
                total_train = 0
                correct_train = 0
            
                print(epoch)
            
                for batch in enumerate(train_loader):
            
                    with torch.autograd.detect_anomaly():

                        inputs = batch[1]['X'].cuda() if torch.cuda.is_available() else batch[1]['X']
                        labels = batch[1]['y'].cuda() if torch.cuda.is_available() else batch[1]['y']
                    
                        # print(i)
                        # print(inputs, labels)
                        
                        labels = torch.argmax(labels, axis = 1).reshape(labels.shape[0])
                        
                        # zero the parameter gradients
                        optimizer.zero_grad()
                
                        # forward + backward + optimize
                        outputs = model(inputs)
                        
                        loss = criterion(outputs, labels)
                        # print("Loss", loss)
                        
                        try:
                            loss.backward()
                            optimizer.step()
                        except RuntimeError as e:
                            print(e, "Exception occurred, stopping training.")
                            break
                        
                        # print(loss)
                
                        _, predictions = torch.max(outputs.data, 1)
                        # print(predictions.eq(labels).sum().item()/labels.shape[0])
                        
                        running_loss += loss.item()
                        total_train += labels.shape[0]
                        correct_train += predictions.eq(labels).sum().item()  
                
                # Training accuracies and losses 
                    
                print("Avg training loss", running_loss/total_train)
                print("Avg training accuracy", correct_train/total_train)
                
                # Validation accuracies and losses 
                
                inputs = torch.autograd.Variable(torch.from_numpy(valid_data.X)).cuda() \
                    if torch.cuda.is_available() else torch.autograd.Variable(torch.from_numpy(valid_data.X))
                outputs = model(inputs)
                
                labels = torch.autograd.Variable(torch.from_numpy(np.argmax(valid_data.y, axis = 1).reshape(valid_data.y.shape[0]))).long().cuda() \
                    if torch.cuda.is_available() else torch.autograd.Variable(torch.from_numpy(np.argmax(valid_data.y, axis = 1).reshape(valid_data.y.shape[0]))).long()
        
                validation_loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs.data, 1)
                validation_acc = predictions.eq(labels).sum().item()/len(labels)
                
                print("Validation loss", validation_loss)
                print("Validation accuracy", validation_acc)
                
                training_accuracies[lr].append(correct_train/total_train)
                training_losses[lr].append(running_loss/total_train)
                validation_accuracies[lr].append(validation_acc)
                validation_losses[lr].append(validation_loss)
            
            
            print('Finished Training')
        
            torch.save(model.state_dict(), output_path+"model_"+str(lr))
            
            # torch.cuda.empty_cache() # to avoid RuntimeError: CUDA out of memory. when training for 83 classes
    
    save_data(training_accuracies, output_path+"training_accuracies.dat", with_torch=True)
    save_data(training_losses, output_path+"training_losses.dat", with_torch=True)
    save_data(validation_accuracies, output_path+"validation_accuracies.dat", with_torch=True)
    save_data(validation_losses, output_path+"validation_losses.dat", with_torch=True)
