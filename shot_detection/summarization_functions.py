#!/usr/bin/env python
'''Summarization:
Given the boundary numbers and a tensor fr_list this code summarizes the frames.
A histogram method, pixel method, and feature comparison method can be used.'''
# coding: utf-8

import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance
from torchvision import models

from shot_detection.shot_boundary_functions import chi2_distance
#VIDEO SUMMARIZATION
#-------------------

#   - Feature method (Alexnet)-
#load imagenet classes for predictions

IMAGENET_CLASSES = {int(idx): entry[1] for (idx, entry) in
                    json.load(open('shot_detection/imagenet_class_index.json')).items()}

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


#   - Main greedy summarization functions -

def greedy_summarization(fr_list, method, boundary):
    '''Performs greedy frame comparison for summarization.
    boundaries: alexnet=75'''
    summary_nrs = []
    diff_list = [0, 0, 0]

    if method == 'pixel':
        comp_fr = fr_list[0]

        for fr_nr, frame in enumerate(fr_list[3::], 3):
            #skips the first three frames, because they contain the dissolve image
            temp_diff = np.absolute((cv2.subtract(comp_fr, frame)))
            diff = np.sum(temp_diff)    #sum all pixel distances in a frame
            diff_list.append(diff)
            if diff >= boundary:        #append if the frame difference is above the boundary
                comp_fr = frame
                summary_nrs.append(fr_nr)

    if method == 'histogram':
        comp_hist = fr_list[0]
        for fr_nr, frame in enumerate(fr_list[3::], 3):
            #skips the first three frames, because they contain the dissolve image
            diff = chi2_distance(frame[0], comp_hist[0])
            # old difference code:
            # temp_diff = np.absolute(comp_hist[0] - frame[0])
            # diff = np.sum(temp_diff) #sum all pixel distances in a frame
            diff_list.append(diff)
            if diff >= boundary:
                comp_hist = frame
                summary_nrs.append(fr_nr)
    if method == 'alexnet':
        comp_features = fr_list[0]
        for fr_nr, fr_features in enumerate(fr_list[3::], 3):
            #skips the first two frames, because they contain the dissolve image
            diff = pairwise_distance(comp_features, fr_features) #calc diff instead of sim
            diff_list.append(diff.item())
            if diff >= boundary:
                comp_features = fr_features
                summary_nrs.append(fr_nr)

    diff_list = (diff_list - np.min(diff_list))/(np.max(diff_list)
                                                 - np.min(diff_list)) #normalize list
    return diff_list, summary_nrs

def greedy_feature_summarization(fr_torchlist, method, boundary, features):
    '''summarizes a torch list of frames by their feature difference'''
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

def report_keyframes_fragments(fragments, method, boundary):
    differences = []
    keyframes = []
    for fragment in fragments:
        diff_list, summary_nrs = greedy_summarization(fragment, method, boundary)
        differences.append(diff_list)
        keyframes.append(summary_nrs)
        # print('summary_nrs', summary_nrs)
    return differences, keyframes

#   - Final Summarization function -
def summarization(fr_list, method, boundary):
    '''Summarize an entire news video per news section.
    Boundary for features: cos_similarity = 0.72, pairwise_distance = 75'''

    diff_list = []
    summary_nrs = []
    for frag in fr_list:
        diff, summary = greedy_summarization(frag, method, boundary)
        # print('\nsummary frs: ', len(summary))
        diff_list.append(diff)
        summary_nrs.append(summary)
    return diff_list, summary_nrs

def keyfr_index_to_image(keyframes_fragments, boundary_nrs, reporter_boundarynr):
    addition_nrs = boundary_nrs
    keyframes_list = []
    for i, keyframes in enumerate(keyframes_fragments):
    #     print(keyframes)
        actual_keyframes = [addition_nrs[i] + 1 + reporter_boundarynr + keyframe for keyframe in keyframes]
    #     print('-', actual_keyframes)
        keyframes_list.append(actual_keyframes)

    return(keyframes_list)
