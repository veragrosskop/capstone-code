#!/usr/bin/env python
'''Summarization:
Given the boundary numbers and a tensor fr_list this code summarizes the frames.
A histogram method, pixel method, and feature comparison method can be used.'''
# coding: utf-8

import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance
from torchvision import models

#VIDEO SUMMARIZATION
#-------------------

#   - Feature method (Alexnet)-
#load imagenet classes for predictions
IMAGENET_CLASSES = {int(idx): entry[1] for (idx, entry) in
                    json.load(open('imagenet_class_index.json')).items()}

def get_alexnet_predictions(tensor_news):
    '''Generates class predictions with alexnet pretrained on imagenet. Code based on:
    http://www.cs.virginia.edu/~vicente/vislang/notebooks/pytorch-lab.html'''
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    predictions = []
    for frame in tensor_news:
        pred = alexnet(frame)
        probs, indices = (-nn.Softmax()(pred).data).sort()
        probs = (-probs).numpy()[0][:10]
        indices = indices.numpy()[0][:10]
        prediction = [IMAGENET_CLASSES[idx] + ': ' + str(prob)
                      for (prob, idx) in zip(probs, indices)]
        predictions.append(prediction)
    return predictions


def get_alexnet_features(tensor_news):
    '''Returns alexnet feature vector (output from fc7 layer)
    with alexnet pretrained on imagenet.'''
    #adapt alexnet so fc7 becomes output layer
    fc7_alexnet = models.alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(fc7_alexnet.classifier.children())[:-1])
    fc7_alexnet.classifier = new_classifier
    fc7_alexnet.eval()
    features = []
    for frame in tensor_news:
        features_fr = fc7_alexnet(frame)
        features.append(features_fr)
    return features


#   - Main greedy summarization functions -

def greedy_summarization(fr_list, method, boundary):
    '''Performs greedy frame comparison for summarization'''
    summary_nrs = []
    diff_list = [0, 0, 0]
    comp_fr = fr_list[0]

    if method == 'pixel':
        for fr_nr, frame in enumerate(fr_list[3::], 3):
            #skips the first three frames, because they contain the dissolve image
            temp_diff = np.absolute((cv2.subtract(comp_fr, frame)))
            diff = np.sum(temp_diff)    #sum all pixel distances in a frame
            diff_list.append(diff)
            if diff >= boundary:        #append if the frame difference is above the boundary
                comp_fr = frame
                summary_nrs.append(fr_nr)

    if method == 'histogram':
        comp_hist = plt.hist(comp_fr.flatten(), 100)
        for fr_nr, frame in enumerate(fr_list[3::], 3):
            #skips the first three frames, because they contain the dissolve image
            fr_hist = plt.hist(frame.flatten(), 100)
            temp_diff = np.absolute(comp_hist[0] - fr_hist[0])
            diff = np.sum(temp_diff) #sum all pixel distances in a frame
            diff_list.append(diff)

            if diff >= boundary:
                comp_hist = fr_hist
                summary_nrs.append(fr_nr)
            plt.close()

    if method == 'network':
        print('choose other method')
        diff = 0

    diff_list = (diff_list - np.min(diff_list))/(np.max(diff_list)
                                                 - np.min(diff_list)) #normalize list
    return diff_list, summary_nrs

def greedy_feature_summarization(fr_torchlist, method, boundary):
    '''summarizes a torch list of frames by their feature difference'''
    features = get_alexnet_features(fr_torchlist) #fr_list = list of frame tensors
    comp_features = features[0]
    summary_nrs = []
    diff_list = [0, 0, 0]
    if method == 'cosine':
        for fr_nr, fr_features in enumerate(features[3::], 3):
            #skips the first two frames, because they contain the dissolve image
            diff = 1 - cosine_similarity(comp_features, fr_features) #calc diff instead of sim
            diff_list.append(diff.item())
            if diff >= boundary:
                comp_features = fr_features
                summary_nrs.append(fr_nr)

    if method == 'distance':
        for fr_nr, fr_features in enumerate(features[3::], 3):
            #skips the first two frames, because they contain the dissolve image
            diff = pairwise_distance(comp_features, fr_features) #calc diff instead of sim
            diff_list.append(diff.item())
            if diff >= boundary:
                comp_features = fr_features
                summary_nrs.append(fr_nr)

    diff_list = (diff_list - np.min(diff_list))/(np.max(diff_list)
                                                 - np.min(diff_list)) #normalize list

    return diff_list, summary_nrs
