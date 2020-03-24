#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:17:46 2020

@author: ombretta
"""

import os
import sys

sys.path.append('../../CrossTask/')
sys.path.append('../')

import torch

import numpy as np
import pickle as pkl

import CrossTaskdataset 
from args import parse_args

import random

import matplotlib.pyplot as plt

from simple_classifier import *

def save_data(data, filename, with_torch=False):
    with open(filename, "wb") as f:
        if with_torch == True: torch.save(data, f)
        else: pkl.dump(data, f)
        
def load_data(filename, with_torch = False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else: print("File", filename, "does not exists.")

def load_training_results(inter_path):
    classes_labels = load_data(inter_path+"/classes_labels.dat")
    valid_videos = load_data(inter_path+"/valid_videos.dat")
    test_videos = load_data(inter_path+"/test_videos.dat")
    train_videos = load_data(inter_path+"/train_videos.dat")
    training_accuracies = load_data(inter_path+"/training_accuracies.dat", with_torch = True)
    training_losses = load_data(inter_path+"/training_losses.dat", with_torch = True)
    validation_accuracies = load_data(inter_path+"/validation_accuracies.dat", with_torch = True)
    validation_losses = load_data(inter_path+"/validation_losses.dat", with_torch = True)
    return classes_labels, train_videos, valid_videos, test_videos, training_accuracies, training_losses, validation_accuracies, validation_losses

def plot_training_curves(num_classes, features_type, output_path, training_accuracies, training_losses, validation_accuracies, validation_losses):
        
        intro = features_type+", "+str(num_classes)+" classes - "
        legend = ["lr = "+str(lr) for lr in training_losses.keys()]
        
        print(max(validation_accuracies[best_lr[features_type][num_classes]]))
        
        plot_data(intro+"Training accuracy", list(training_accuracies.values()), "Mean accuracy", legend, output_path+"training_accuracies.png", x_ticks=range(1,21))
        plot_data(intro+"Training loss", list(training_losses.values()), "Cross entropy loss", legend, output_path+"training_losses.png", x_ticks=range(1,21))
        plot_data(intro+"Validation accuracy", list(validation_accuracies.values()), "Mean accuracy", legend, output_path+"validation_accuracies.png", x_ticks=range(1,21))
        plot_data(intro+"Validation loss", list(validation_losses.values()), "Cross entropy loss", legend, output_path+"validation_losses.png", x_ticks=range(1,21))
        # plt.close('all')
        return

def plot_data(fig_name, data_series, yaxis, legend, output_path, xaxis="Epochs", x_ticks=range(1,11), figsize = [8,5], save = True, x_font_size=10, colors = []):
        
        figure = plt.figure(figsize = figsize)
        
        if len(colors) == 0: 
            for data in data_series:
                plt.plot(data)
        else:
            for data, c in zip(data_series, colors):
                plt.plot(data,c)
            
        plt.title(fig_name)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        
        plt.xticks(range(len(data)+1), x_ticks, fontsize=x_font_size)
    
        plt.legend(legend)
        plt.savefig(output_path)
        
        return figure

def get_video_information(dataset_path, features_type, test_video, temporal_aggregation, aggregation_window, classes_labels):
    filename = [f for f in os.listdir(dataset_path+"is/") if test_video in f][0]
    print(filename)
    
    features = np.load(dataset_path+features_type+"/"+filename, allow_pickle=True)
    if temporal_aggregation == True: features = aggregate_features(features, aggregation_window)
    
    y_true = classes_labels[filename.split("__")[0]]
    return filename, features, y_true
            
    
def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def predict_unit(data_series, i, percentage_unit, top_k):
    features_interval = data_series[round(percentage_unit*i):round(percentage_unit*i+percentage_unit)]
    if len(features_interval) == 0: 
        features_interval = outputs[round(percentage_unit*i)].unsqueeze(0)
    # predictions_unit = torch.max(features_interval,1)
    predictions_unit = torch.mean(features_interval,0)
    predictions_unit = np.argsort(predictions_unit)[-top_k:]
    return predictions_unit
        
def aggregate_predictions(outputs, t, aggregation_modality="mean"):
    if aggregation_modality=="mean": 
        return torch.mean(outputs.data[:t+1],0)
    if aggregation_modality=="max": 
        return torch.max(outputs.data[:t+1],0)[0]
    if aggregation_modality=="median": 
        return torch.median(outputs.data[:t+1],0)[0]


#%%
frequencies, _, _, _, all_tasks_info = initialize("full")
#%%

temporal_aggregation = False
aggregation_window = 10

# results_path = "./torch/"
# results_path = "./aggregated_features/balanced/"
results_path = "./balanced/"


all_features_types = ["sw"]

all_num_classes = [83]#[10,20,30,40,83] #[2,3,4,5,10,20,30,40,83]

best_lr = {}
correct_predictions_per_class = {}

# Collect results computed on the cluster
    
for features_type in all_features_types:
    best_lr[features_type] = {}
    correct_predictions_per_class[features_type] = {}
    
    for num_classes in all_num_classes:
        
        # Load settings
        
        print(features_type, num_classes)
        
        inter_path = results_path+features_type+"_"+str(num_classes)
        # inter_path = results_path+features_type
        
        print(inter_path)
        classes_labels, train_videos, valid_videos, test_videos, training_accuracies, \
            training_losses, validation_accuracies, validation_losses \
                = load_training_results(inter_path)
                
        for task in classes_labels: print(task, ":", all_tasks_info["title"][task])
        
        # Compare results to get best learning rate 
        
        best_lr[features_type][num_classes] = list(validation_accuracies.keys())[int(np.floor(np.argmax(list(validation_accuracies.values()))/len(list(validation_accuracies.values())[0])))]
        
        output_path = inter_path+"/plots/"
        if not os.path.exists(output_path): os.mkdir(output_path)

        plot_training_curves(num_classes, features_type, output_path, training_accuracies, training_losses, validation_accuracies, validation_losses)
        save_data(best_lr, results_path+"/best_lr.dat")
        
        # best_lr[features_type][num_classes] = 0.001
        
        # Test trained model on test set 
        
        if not os.path.exists(inter_path+"/test_results/"): os.mkdir(inter_path+"/test_results/")
        
        dataset_path = "./features/"
        model_path = inter_path+"/model_"+str(best_lr[features_type][num_classes])
        
        D_in, H, D_out = 1024, num_classes, 1
            
        model = OneLayerNet(D_in, H, D_out)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
            
        # for aggregation_modality in ["mean", "max", "median"]:
        for aggregation_modality in ["mean"]:

            correct_predictions = np.zeros(100)
            smoothed_correct_predictions = np.zeros(100)
            
            correct_predictions_per_class[features_type][num_classes] = {}
            smoothed_correct_predictions_per_class = {}
            videos_count_per_class = np.zeros(num_classes)
            
            top_k = 1
    
    
                
            for test_video in test_videos:
            # for test_video in test_videos[:1]:
                
                # print("Test video:", test_video)
                
                filename, features, y_true = get_video_information(dataset_path, features_type, test_video, temporal_aggregation, aggregation_window, classes_labels)
    
                # print("True label:", y_true, ",", filename.split("__")[0])
                    
                # Confidence all classes
                inputs = torch.autograd.Variable(torch.from_numpy(features)).cuda() \
                    if torch.cuda.is_available() else torch.autograd.Variable(torch.from_numpy(features))
                outputs = torch.exp(model(inputs).data)
                
                fig_path = inter_path+"/test_results/"+features_type+"_"+filename.split(".npy")[0]+".png"
                legend = [c+": "+all_tasks_info["title"][c] for c in classes_labels]
                x_ticks=[str(i) if i%10==0 else "" for i in range(outputs.shape[0])]
                # plot_data("Task predictions "+features_type+", video "+filename.split(".npy")[0], np.swapaxes(outputs,0,1), "Class probability", legend, fig_path, xaxis="Seconds", x_ticks=x_ticks, x_font_size=8, figsize = [30,14])
            
                # Confidence moving average
                confidence_over_time = torch.zeros(outputs.shape)
                for t in range(features.shape[0]): #video duration (s)
                    # confidence_over_time[t] = torch.mean(outputs.data[:t+1],0)
                    confidence_over_time[t] = aggregate_predictions(outputs, t, aggregation_modality)
                    
                fig_path = inter_path+"/test_results/"+features_type+"_confidences_"+filename.split(".npy")[0]+"_"+aggregation_modality+".png"
                # plot_data("Task predictions "+features_type+", video "+filename.split(".npy")[0]+" - Confidence over time ("+aggregation_modality+")", np.swapaxes(confidence_over_time,0,1), "Confidence over time", legend, fig_path, xaxis="Seconds", x_ticks=x_ticks, x_font_size=8, figsize = [30,14])
                
                two_lines_path = inter_path+"/test_results/two_lines_plots/"
                if not os.path.exists(two_lines_path): os.mkdir(two_lines_path)
                
                # Probability predicted class and ground truth 
                fig_path = two_lines_path+features_type+"_2lines_"+filename.split(".npy")[0]+".png"
                legend = ["True class", "Top prediction"]
                true_class = outputs[:,y_true]
                highest_prediction, _ = torch.max(outputs,1)
                # plot_data("Task predictions "+features_type+", video "+filename.split(".npy")[0]+" - Top prediction vs ground truth", torch.stack((true_class,highest_prediction)), "Class probability", legend, fig_path, xaxis="Seconds", x_ticks=x_ticks, x_font_size=8, figsize = [30,14], colors = ["g", "r"])
    
                # Confidence over time predicted class and ground truth 
                fig_path = two_lines_path+features_type+"_2lines_confidence_"+filename.split(".npy")[0]+"_"+aggregation_modality+".png"
                true_class_confidence = confidence_over_time[:,y_true]
                highest_prediction_confidence, _ = torch.max(confidence_over_time,1)
                # plot_data("Task predictions "+features_type+", video "+filename.split(".npy")[0]+" - Top prediction vs ground truth ("+aggregation_modality+")", torch.stack((true_class_confidence,highest_prediction_confidence)), "Class probability", legend, fig_path, xaxis="Seconds", x_ticks=x_ticks, x_font_size=8, figsize = [30,14], colors = ["g", "r"])

        
                if temporal_aggregation == True: features = aggregate_features(features, aggregation_window)
                
                print("True label:", y_true, ",", filename.split("__")[0])
                
                if y_true not in correct_predictions_per_class[features_type][num_classes]:
                    correct_predictions_per_class[features_type][num_classes][y_true] = np.zeros(100)
                    smoothed_correct_predictions_per_class[y_true] = np.zeros(100)
                
                videos_count_per_class[y_true]+=1
                
                # Confidence all classes
                confidence_over_time = torch.zeros(outputs.shape)
                for t in range(features.shape[0]): #video duration (s)
                    confidence_over_time[t] = aggregate_predictions(outputs, t, aggregation_modality)
                
                # Conversion in percentage 
                video_duration = outputs.data.shape[0] #duration in seconds
                percentage_unit = video_duration/100
                        
                for i in range(100):
                    
                    if round(percentage_unit*i) < video_duration:
                        
                        predictions_unit = predict_unit(outputs, i, percentage_unit, top_k)
                        smoothed_predictions_unit = predict_unit(confidence_over_time, i, percentage_unit, top_k)
                        
                        if y_true in predictions_unit.data:
                            correct_predictions[i]+=1 
                            correct_predictions_per_class[features_type][num_classes][y_true][i]+=1
                        if y_true in smoothed_predictions_unit.data: 
                            smoothed_correct_predictions[i]+=1
                            smoothed_correct_predictions_per_class[y_true][i]+=1
                        
            
            correct_predictions_percentage = correct_predictions/len(test_videos)       
            smoothed_correct_predictions_percentage = smoothed_correct_predictions/len(test_videos)       
                    
            fig_path = inter_path+"/test_results/correct_predictions_percentage_over_time_top"+str(top_k)+"_"+aggregation_modality+".png"
            legend = ["Predictions every timestep", "Smoothed predictions (from t=0 to t=t_current) - "+aggregation_modality]
            x_ticks=[str(i) if i%10==0 else "" for i in range(101)]
            
            
            # Only one line:
            # plot_data("Task predictions "+features_type+" - Mean accuracy over time", [correct_predictions_percentage], "Mean accuracy", "Predictions every timestep", inter_path+"/test_results/accuracy_over_time.png", xaxis="Time (%)", x_ticks=x_ticks, x_font_size=8, figsize = [30,14], colors = ["r"])
              
            plot_data("Task predictions "+features_type+" - Mean accuracy over time (top"+str(top_k)+")", [correct_predictions_percentage, smoothed_correct_predictions_percentage], "Mean accuracy", legend, fig_path, xaxis="Time (%)", x_ticks=x_ticks, x_font_size=8, figsize = [30,14], colors = ["r", "C9"])
        

#%%

legend = ["Predictions every timestep", "Predictions every timestep - aggregated features", "Smoothed predictions (from t=0 to t=t_current)", "Smoothed predictions (from t=0 to t=t_current) - aggregated features"]
fig_path = inter_path+"/test_results/prova.png"
plot_data("Task predictions "+features_type+" - Mean accuracy over time", [correct_predictions_percentage, aa, smoothed_correct_predictions_percentage, bb], "Mean accuracy", legend, fig_path, xaxis="Time (%)", x_ticks=x_ticks, x_font_size=8, figsize = [30,14], colors = ["r", "orange", "C9", "y"])
        
#%%

# TODO: calculate accuracy for each class

print(correct_predictions_per_class)
print(smoothed_correct_predictions_per_class)
for task_class in range(num_classes):
    for i in range(100):
        correct_predictions_per_class[task_class][i] = correct_predictions_per_class[task_class][i]/videos_count_per_class[task_class]
        smoothed_correct_predictions_per_class[task_class][i] = smoothed_correct_predictions_per_class[task_class][i]/videos_count_per_class[task_class]
#%%
        
print(correct_predictions_per_class)
print(smoothed_correct_predictions_per_class)

for data in correct_predictions_per_class.values():
    # print(data)
    plt.plot(data)
    
for data in smoothed_correct_predictions_per_class.values():
    # print(data)
    plt.plot(data)