#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:31:01 2019

@author: ombretta
"""


""" Load data """

import tensorflow as tf
import numpy as np

import CrossTaskdataset
from args import parse_args

import pandas as pd

def count_rows(dataset):
    i = 0
    for batch in dataset:
        for sample in batch:
            rows = sample['X'].shape[0]
            i += rows
    return i 

def fill_feature_matrix(dataset, n_rows, dim_features, classes_labels):
    X = np.zeros([n_rows, dim_features], dtype=np.float32)
    y = np.zeros([n_rows, 1], dtype=np.float32)
    i = 0
    for batch in dataset:
        for sample in batch:
            rows = sample['X'].shape[0]
            X[i:i+rows,:] = sample['X'][:,:dim_features]
            y[i:i+rows] = classes_labels[sample['task']]
            i += rows
    return X,y

def get_valid_set(X, y, ratio, dim_features): # 0 < ratio < 1 #Change splitting so that each set incorporates FULL videos!
    
    valid_rows = round(ratio*X.shape[0])
    X_valid = np.zeros([valid_rows, dim_features], dtype=np.float32)
    y_valid = np.zeros([valid_rows, 1], dtype=np.float32)
    X_train = np.zeros([X.shape[0]-valid_rows, dim_features], dtype=np.float32)
    y_train = np.zeros([X.shape[0]-valid_rows, 1], dtype=np.float32)
    for i in range(X.shape[0]):
        if i < valid_rows: X_valid[i], y_valid[i] = X[i], y[i]
        else: X_train[i-valid_rows], y_train[i-valid_rows] = X[i], y[i]
    return X_train, y_train, train_rows, X_valid, y_valid, valid_rows
            
        
        
        
#%%
args = parse_args()

[trainloader, testloader, A, M, all_tasks_info] = CrossTaskdataset.load_cross_task_dataset(args)

classes_labels = {}
print(all_tasks_info['title'])
id = 0
for task in list(all_tasks_info['title'].keys()):
    classes_labels[task] = id
    id += 1              
#%%
#
#for batch in trainloader:
#    for sample in batch:
#        print(".")
#        y[i] = classes_labels[sample['task']]
#        i += rows   
#%%           
print(classes_labels[sample['task']])
    
#%%
print(classes_labels)

#%%
print(len(list(classes_labels.keys())))
n_classes = len(list(classes_labels.keys())) #83

#%%
train_rows = count_rows(trainloader)
test_rows = count_rows(testloader)

X_train_valid, y_train_valid = fill_feature_matrix(trainloader, train_rows, 1024, classes_labels)
X_test, y_test = fill_feature_matrix(testloader, test_rows, 1024, classes_labels)        
X_train, y_train, train_rows, X_valid, y_valid, valid_rows = get_valid_set(X_train_valid, y_train_valid, 0.2, 1024)

#%%
train_rows = X_train.shape[0]
#%%
print(train_rows)
#%%
#print(X_train.shape, X_valid.shape, X_test.shape) #(593900, 1024) (148475, 1024) (513063, 1024)

#%%
#for i in y_valid: print(i)
#%%

tf.reset_default_graph()

learning_rate = 0.01

#Defining placeholders that accept minibatches of different sizes
d = 1024

with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (None, d))
  y = tf.placeholder(tf.float32, (None,))
  
#Defining a hidden layer
with tf.name_scope("hidden-layer"):
  W = tf.Variable(tf.random_normal((d, n_classes)))
  b = tf.Variable(tf.random_normal((n_classes,)))
  x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
  
with tf.name_scope("output"):
  W = tf.Variable(tf.random_normal((n_classes, 1)))
  b = tf.Variable(tf.random_normal((1,)))
  y_logit = tf.matmul(x_hidden, W) + b
  y_one_prob = tf.sigmoid(y_logit)
  y_pred = tf.round(y_one_prob)
  
with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  y_expand = tf.expand_dims(y, 1)
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
  # Sum all contributions
  l = tf.reduce_sum(entropy)
  
with tf.variable_scope('Accuracy'):
    correct_prediction = tf.equal(y_pred, y, name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
  
with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()


#Add a placeholder for dropout probability
keep_prob = tf.placeholder(tf.float32)

#Training on minibatches
sess = tf.Session()

sess.run(tf.global_variables_initializer())

n_epochs = 100
N = train_rows
batch_size = 50
dropout_prob = 0.3
step = 0

#%%

N = int(len(y_train) / batch_size)

for epoch in range(n_epochs):
  pos = 0
  while pos < N:
    
    print(N, pos, N-pos)
    batch_X = X_train[pos:pos+batch_size,:] if N-pos >= batch_size else X_train[pos:N,:]
    batch_y = y_train[pos:pos+batch_size,:].reshape([batch_size,]) if N-pos >= batch_size else y_train[pos:N,:].reshape([N-pos,])
    feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("epoch %d, step %d, loss: %f" % (epoch+1, step, loss))
#    train_writer.add_summary(summary, step)
#    print(summary)
    print(loss)
    
    # Run validation after every epoch
#    x = X_valid[:1000,:]
#    y = y_valid[:1000,:].reshape([1000,])
    feed_dict_valid = {x: X_valid[:1000,:], y: y_valid[:1000,:].reshape([1000,])}
    valid_accuracy, loss_valid = sess.run([accuracy, l], feed_dict=feed_dict_valid)
    print('\n')
    print(epoch + 1, valid_accuracy, loss_valid)
    print('---------------------------------------------------------')
    
    
    
    step += 1
    pos += batch_size
    
#%%
    
train_writer.add_summary(summary, step)


#%%    
#Computing a weighted accuracy
train_weighted_score = accuracy_score(y_train, train_y_pred, sample_weight=train_w)
print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
valid_weighted_score = accuracy_score(y_valid, valid_y_pred, sample_weight=valid_w)
print("Valid Weighted Classification Accuracy: %f" % valid_weighted_score)


