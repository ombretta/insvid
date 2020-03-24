#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:56:25 2019

@author: ombretta
"""

import sys
sys.path
sys.path.append('/Users/ombretta/Documents/Code/CrossTask/')
print(sys.path)
from args import parse_args

from data import get_vids
from data import get_A
from data import read_task_info
from data import random_split
from data import CrossTaskDataset


from torch.utils.data import DataLoader
import torch as th


def adjust_folder_path(old_path):
    return "../../CrossTask/" + old_path

    
def load_cross_task_dataset(args):
    # task_vids = get_vids(adjust_folder_path(args.video_csv_path)) #list of videos in format <Task ID>,<YouTube video ID>,<URL>
    task_vids = get_vids('../../../CrossTask/crosstask_release/videos.csv') #list of videos in format <Task ID>,<YouTube video ID>,<URL>
    # val_vids = get_vids(adjust_folder_path(args.val_csv_path)) #list of videos in format <Task ID>,<YouTube video ID>,<URL> (validation set, other tasks and videos)
    val_vids = get_vids('../../../CrossTask/crosstask_release/videos_val.csv') #list of videos in format <Task ID>,<YouTube video ID>,<URL> (validation set, other tasks and videos)
    task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for task,vids in task_vids.items()}
    
    # primary_info = read_task_info(adjust_folder_path(args.primary_path)) # list of 18 primary tasks (Task ID, Task name, WikiHow url, # steps, steps)
    primary_info = read_task_info('../../../CrossTask/crosstask_release/tasks_primary.txt') # list of 18 primary tasks (Task ID, Task name, WikiHow url, # steps, steps)
    # related_info = read_task_info(adjust_folder_path(args.related_path))
    related_info = read_task_info('../../../CrossTask/crosstask_release/tasks_related.txt')
    
    test_tasks = set(primary_info['steps'].keys()) # all tasks
    
    if args.use_related: # 1 for using related tasks during training, 0 for using primary tasks only (df 1)
        task_steps = {**primary_info['steps'], **related_info['steps']} #keywords argument (**) concatenation 
        n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
    else:
        task_steps = primary_info['steps']
        n_steps = primary_info['n_steps'] 
    all_tasks = set(n_steps.keys()) # same as test_tasks  if use_related==0
    task_vids = {task: vids for task,vids in task_vids.items() if task in all_tasks}
    
    all_tasks_info = primary_info.copy()
    for k in list(all_tasks_info.keys()):
        all_tasks_info[k].update(related_info[k])
    
    # Step-to-component matrices, encode of all possible tasks and steps
    A, M = get_A(task_steps, share=args.share) # depends on granularity of the sharing (words, words within same task, whole step descriptions, no sharing)
    # A: dictionary, keys:tasks, values: Step-to-component matrices (task_steps x number of steps)
    # M: number of total possible task_steps

    if args.use_gpu:
        A = {task: a.cuda() for task, a in A.items()}
    
    train_vids, test_vids = random_split(task_vids, test_tasks, args.n_train)
    # trainset = CrossTaskDataset(train_vids, n_steps, adjust_folder_path(args.features_path), adjust_folder_path(args.constraints_path))
    trainset = CrossTaskDataset(train_vids, n_steps, '../../../CrossTask/crosstask_features', '../../../CrossTask/crosstask_constraints')

    # DataLoader: :class:`~torch.utils.data.DataLoader`
    trainloader = DataLoader(trainset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        shuffle = True, 
        drop_last = True,
        collate_fn = lambda batch: batch,
        )
    testset = CrossTaskDataset(test_vids, n_steps, '../../../CrossTask/crosstask_features', '../../../CrossTask/crosstask_constraints')
    testloader = DataLoader(testset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        shuffle = False, 
        drop_last = False,
        collate_fn = lambda batch: batch,
        )

    return trainloader, testloader, A, M, all_tasks_info



    
#%%

#args = parse_args()
#[trainloader, testloader, A, M, all_tasks_info] = load_cross_task_dataset(args)
#
##%%
#print(all_tasks_info.keys())
#
#print(all_tasks_info["title"])
#print(all_tasks_info["n_steps"])

#print(all_tasks_info.values())

#%%

#for batch in trainloader:
#    for sample in batch:
#        print(sample)


#%%


#print(trainloader.)










#%%
def main():
    
    task_vids = get_vids(adjust_folder_path(args.video_csv_path)) #list of videos in format <Task ID>,<YouTube video ID>,<URL>
    val_vids = get_vids(adjust_folder_path(args.val_csv_path)) #list of videos in format <Task ID>,<YouTube video ID>,<URL> (validation set, other tasks and videos)
    task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for task,vids in task_vids.items()}
    
    primary_info = read_task_info(adjust_folder_path(args.primary_path)) # list of 18 primary tasks (Task ID, Task name, WikiHow url, # steps, steps)
    
    test_tasks = set(primary_info['steps'].keys()) # all tasks
    
    if args.use_related: # 1 for using related tasks during training, 0 for using primary tasks only (df 1)
        related_info = read_task_info(adjust_folder_path(args.related_path))
        task_steps = {**primary_info['steps'], **related_info['steps']} #keywords argument (**) concatenation 
        n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
    else:
        task_steps = primary_info['steps']
        n_steps = primary_info['n_steps'] 
    all_tasks = set(n_steps.keys()) # same as test_tasks  if use_related==0
    task_vids = {task: vids for task,vids in task_vids.items() if task in all_tasks}
    
    # Step-to-component matrices, encode of all possible tasks and steps
    A, M = get_A(task_steps, share=args.share) # depends on granularity of the sharing (words, words within same task, whole step descriptions, no sharing)
    # A: dictionary, keys:tasks, values: Step-to-component matrices (task_steps x number of steps)
    # M: number of total possible task_steps

    if args.use_gpu:
        A = {task: a.cuda() for task, a in A.items()}
    
    
    train_vids, test_vids = random_split(task_vids, test_tasks, args.n_train)
    
    trainset = CrossTaskDataset(train_vids, n_steps, adjust_folder_path('../../../CrossTask/crosstask_features'), adjust_folder_path(args.constraints_path))
    # DataLoader: :class:`~torch.utils.data.DataLoader`
    trainloader = DataLoader(trainset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        shuffle = True, 
        drop_last = True,
        collate_fn = lambda batch: batch,
        )
    testset = CrossTaskDataset(test_vids, n_steps, adjust_folder_path('../../../CrossTask/crosstask_features'), adjust_folder_path(args.constraints_path))
    testloader = DataLoader(testset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        shuffle = False, 
        drop_last = False,
        collate_fn = lambda batch: batch,
        )

    
#    print(task, vid, K, T) # eg: 67160 hLu1-Z05V_Q 10 426
#    print(sample['X'].shape) # eg: torch.Size([426, 3200])

#%%

#if __name__== "__main__":
#  main()


