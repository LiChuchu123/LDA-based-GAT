from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import os
import h5py
import pandas as pd
import random
import pdb
import math
from random import randint, sample

import torch
from sklearn.model_selection import KFold


def load_data(dataset):
    print("Loading lncRNAdrug dataset")

    net_path = 'raw_data/' + dataset + '/lncRNA_drug.csv'
    net = pd.read_csv(net_path,header= None,index_col=None)
    net = np.array(net, dtype='uint8')
    lncSim_path = 'raw_data/' + dataset + '/lncsim_pca.csv'
    lncSim_data = pd.read_csv(lncSim_path, header=None,index_col=None)
    u_features1 = np.array(lncSim_data, dtype=np.float32)


    drugSim_path = 'raw_data/' + dataset + '/drugsim_pca.csv'
    drugSim_data = pd.read_csv(drugSim_path, header=None,index_col=None)
    v_features = np.array(drugSim_data, dtype=np.float32)

    num_list = [len(u_features1)]
    num_list.append(len(v_features))
    temp0 = np.zeros((u_features1.shape[0], v_features.shape[1]),int)
    u_features = np.hstack((u_features1, temp0))

    temp0 = np.zeros((v_features.shape[0], u_features1.shape[1]), int)
    v_features = np.hstack((temp0, v_features))

    a = np.zeros((1, u_features.shape[1]), int)
    b = np.zeros((1, v_features.shape[1]), int)
    u_features = np.vstack((a, u_features))
    v_features = np.vstack((b, v_features))

    total_num_lncRNAs = net.shape[0]
    total_num_drugs = net.shape[1]




    row, col, _ = sp.find(net)

    perm = random.sample(range(len(row)), len(row))
    row, col = row[perm], col[perm]
    sample_pos = (row, col)



    print("the number of all positive sample:", len(sample_pos[0]))

    print("sampling negative links for train and test")
    sample_neg = ([], [])

    X = np.ones((total_num_lncRNAs, total_num_drugs))

    net_neg = X - net
    row_neg, col_neg, _ = sp.find(net_neg)
    perm_neg = random.sample(range(len(row_neg)), len(row))
    row_neg, col_neg = row_neg[perm_neg], col_neg[perm_neg]
    sample_neg = (row_neg, col_neg)
    sample_neg = list(sample_neg)
    print("the number of all negative sample:", len(sample_neg[0]))

    u_idx = np.hstack([sample_pos[0], sample_neg[0]])
    v_idx = np.hstack([sample_pos[1], sample_neg[1]])
    labels = np.hstack([[1] * len(sample_pos[0]), [0] * len(sample_neg[0])])

    l1 = np.zeros((1, net.shape[1]), int)
    print(l1.shape)
    net = np.vstack([l1, net])
    print("old net:", net.shape)
    l2 = np.zeros((net.shape[0], 1), int)
    net = np.hstack([l2, net])
    print("new net:", net.shape)

    u_idx = u_idx + 1
    v_idx = v_idx + 1
    u_features = np.array(u_features,dtype=np.float32)
    v_features = np.array(v_features,dtype=np.float32)

    return u_features, v_features, net, labels, u_idx, v_idx, num_list


def load_predict_data(dataset):
    print("Loading lncRNAdrug dataset")

    net_path = 'raw_data/' + dataset + '/lncRNA_drug.csv'
    net = pd.read_csv(net_path,index_col=0)
    net = np.array(net, dtype=np.int32)

    num_lncRNAs = net.shape[0]
    num_drugs = net.shape[1]




    lncSim_path = 'raw_data/' + dataset + '/lncsim_pca.csv'
    lncSim_data = pd.read_csv(lncSim_path, header=None)
    u_features1 = np.array(lncSim_data, dtype=np.float32)

    drugSim_path = 'raw_data/' + dataset + '/drugsim_pca.csv'
    drugSim_data = pd.read_csv(drugSim_path, header=None)
    v_features = np.array(drugSim_data, dtype=np.float32)

    num_list = [len(u_features1)]
    num_list.append(len(v_features))
    temp0 = np.zeros((u_features1.shape[0], v_features.shape[1]), int)

    u_features = np.hstack((u_features1, temp0))

    temp0 = np.zeros((v_features.shape[0], u_features1.shape[1]), int)
    v_features = np.hstack((temp0, v_features))
    a = np.zeros((1, u_features.shape[1]), int)
    b = np.zeros((1, v_features.shape[1]), int)
    u_features = np.vstack((a, u_features))
    v_features = np.vstack((b, v_features))

    net1 = np.zeros((net.shape[0],net.shape[1]),int)
    net_new = np.zeros((num_lncRNAs + 1, num_drugs + 1), dtype=np.int32)
    for i in range(1, num_lncRNAs + 1):
        for j in range(1, num_drugs + 1):
            net_new[i, j] = net1[i - 1, j - 1]

    # loading drug_name and disease_name
    lncRNA_name = []
    drug_name = []
    drug_name.append([])
    lncRNA_name.append([])
    csv1=pd.read_csv('raw_data/' + dataset + '/lnc_name.csv')
    csv2=pd.read_csv('raw_data/' + dataset + '/drug_name.csv',header=None)
    lnc=[str(i) for i in csv1.iloc[:,0]]
    drug=[str(i) for i in csv2.iloc[:,0]]
    for i in lnc:
        lncRNA_name.append(i)
    for i in drug:
        drug_name.append(i)



    print(len(lncRNA_name))
    print(len(drug_name))
    print("drug_name:", len(drug_name))
    case_drug = 'Panobinostat'
    # case_lncRNA = 'NEAT1'
    if case_drug in drug_name:
        idx = drug_name.index(case_drug)
        # idx = lncRNA_name.index(case_lncRNA)
    print(idx)
    u_idx, v_idx, labels = [], [], []
    list = []
    for i in len(csv1):
        # if net_new[i][idx] == 0:
        # list.append([i, idx, 0])
        list.append([i, idx, 0])
    for i in range(len(list)):
        u_idx.append(list[i][0])
        v_idx.append(list[i][1])
        labels.append(list[i][2])
    class_values = np.array([0, 1], dtype=float)

    return u_features, v_features, net_new, labels, u_idx, v_idx, class_values, lncRNA_name, drug_name


