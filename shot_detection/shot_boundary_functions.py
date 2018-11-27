#!/usr/bin/env python
'''Shot Boundary Detection code:
First the video needs to be seperated into different frames.
This can be done in a seperate build file for all videos at the same time.
For now it is done with a single command with the test-news video.

for all frames: ffmpeg -i test-news.mp4 image-%04d.png
for 10 fps: ffmpeg -i test-news.mp4 -vf fps=10 image-%03d.png'''
# coding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
imageio.plugins.ffmpeg.download() #install ffmpeg plugin
from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance
from torchvision import models

#VIDEO FRAGMENTATION
############################
def t_cut_fr(cum_diff, k):
    '''Calculates the t_cut for frame differences, according to given k parameter.'''
    t_cut = np.mean(cum_diff) + (k * np.std(cum_diff))
    return t_cut

#   - Pixel Method -
def pixcomp_diff(fr_list, comp_fr):
    '''pixel comparison'''
    diff = []
    for frame in fr_list:
        diff_temp = np.absolute((cv2.subtract(frame, comp_fr)))
        diff_sum = np.sum(diff_temp)
        diff.append(diff_sum)
    return diff

#   - Histogram method -
def make_histlist(fr_list):
    '''creates a list of all histogram values for fr_list'''
    hist_list = []
    for frame in fr_list:
        hist = plt.hist(frame.flatten(), 100)
        hist_list.append(hist)
        plt.close()
    return hist_list

def chi2_distance(histA, histB, eps = 1e-10):
    '''Calculates the Chi squared distance between two image histograms.
    Code reference : https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/'''
    # compute the chi-squared distance
    diff_hist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
    for (a, b) in zip(histA, histB)])
 
    # return the chi-squared distance
    return diff_hist

def histcomp_diff(hist_list, comp_hist):
    '''histogram comparison given a histogram list and a comparison frame histogram'''
    diff = []
    for hist in hist_list:
        diff_temp = chi2_distance(hist[0], comp_hist[0])
        diff.append(diff_temp)
        # old  difference code: 
        # diff_temp = np.absolute(hist[0] - comp_hist[0])
        # diff_sum = np.sum(diff_temp)
        # diff.append(diff_sum)
    return diff

#   - Feature method - 
def get_alexnet_features(tensor_news):
    '''Returns alexnet feature vector (output from fc7 layer)
    with alexnet pretrained on imagenet.'''
    #adapt alexnet so fc7 becomes output layer
    fc7_alexnet = models.alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(fc7_alexnet.classifier.children())[:-1])
    fc7_alexnet.classifier = new_classifier
    fc7_alexnet.eval()
    features_list = []
    for frame in tensor_news:
        features_fr = fc7_alexnet(frame)
        features_list.append(features_fr)
    return features_list

def featurecomp_diff(features_list, comp_features):
    diff = []
    for fr_features in features_list:
        diff_temp = pairwise_distance(comp_features, fr_features)
        diff.append(diff_temp.item())
    return diff

#   - Comparison functions -

def pick_similar_frames(fr_diff, fr_list, t_cut):
    '''Given a t_cut parameter this function selects all frames similar to the comparison frame.'''
    sim_frames = []
    for fr_nr, diff in enumerate(fr_diff):
        if diff <= t_cut:
            frame = (fr_nr, fr_list[fr_nr])
            sim_frames.append(frame)
    return sim_frames

def compare_frs(fr_list, comp_method, method, k):
    '''Given a list of frames, frame differnce method, and k the Tcut threshold parameter.'''
    if method == 'histogram':
        fr_diff = histcomp_diff(fr_list, comp_method)
        # fr_diff *= 1.0/max(fr_diff) #scale from 0 to 1 -> otherwise huge numbers are computed (300000)
    elif method == 'pixel':
        fr_diff = pixcomp_diff(fr_list, comp_method)
    elif method == 'alexnet':
        fr_diff = featurecomp_diff(fr_list, comp_method)
    fr_diff = (fr_diff - np.min(fr_diff))/(np.max(fr_diff) - np.min(fr_diff))  #normalize
    t_cut = t_cut_fr(fr_diff, k) #set boundary with Tcut method
    sim_frames = pick_similar_frames(fr_diff, fr_list, t_cut) #save boundary frames
    return sim_frames, fr_diff, t_cut

def improve_boundaries(sim_frames, fr_list):
    '''Select only the valid boundary frames from sim_frames.'''
    boundaries = [(0, fr_list[0])] #append first frame
    last_fr = ((len(fr_list)-1), fr_list[-1])
    nr_a = 0
    for nr_b, fr_b in sim_frames:
        #check if more than 20 frames between the two frames similar to the comp_fr
        if (nr_b - nr_a) <= 20:
            nr_a = nr_b
        elif (nr_b - nr_a) >= 20:
            fr_frnr = (nr_b, fr_b)
            boundaries.append(fr_frnr)
            nr_a = nr_b
    boundaries.append(last_fr)
    boundary_nrs, _ = zip(*boundaries)
    boundary_nrs = list(boundary_nrs)
    return boundaries, boundary_nrs

#   - Fragmentation Functions-

def split_fr_list(fr_list, boundary_nrs):
    '''Seperates the list of frames into seperate lists of frames.
    The lists are seperated according to the comparison frame.'''
    boundary_nrs_2 = np.copy(boundary_nrs[1:])
    boundary_nrs_2[-1] = boundary_nrs_2[-1] + 1
    fragments = []
    for nr_a, nr_b in zip(boundary_nrs, boundary_nrs_2):
        # print('from ', nr_a, ' to ', nr_b, '\n')
        fragments.append(fr_list[nr_a:nr_b])
    return fragments

def split_fr_nrs(fr_amount, boundary_nrs):
    '''returns list of frame numbers in fragmented lists according to boundary_nrs
    uses 0-index'''
    boundary_nrs2 = np.copy(boundary_nrs[1:])
    boundary_nrs2[-1] = boundary_nrs2[-1] + 1
    fragment_nrs = []
    for nr_a, nr_b in zip(boundary_nrs, boundary_nrs2):
        fragment_nrs.append(np.arange(fr_amount)[nr_a:nr_b])
    return fragment_nrs

def boundary_range(boundary_nrs):
    '''returns the start and end boundary of each video fragment'''
    boundary_nrs2 = np.copy(boundary_nrs[1:])
    boundary_nrs2[-1] = boundary_nrs2[-1] + 1

    return list(zip(boundary_nrs, boundary_nrs2))

#   - Main Shot Boundary Function -
def fragment_frs(fr_list, comp_fr, method, k):
    '''Main function for video fragmentation according to comparison frame.
    Splits video into different news sections.
    Does not yet split of the beginning and end of video from news sections,
    but appends them to the first and last frames.'''
    sim_frames, _, _ = compare_frs(fr_list, comp_fr, method, k) #compare frames to comparison fr
    #appends the first and last frames to boundary nrs and removes any subsequently similar frames
    _, boundary_nrs = improve_boundaries(sim_frames, fr_list)
    #splits frame list into different fragments and returns a new fragmented fr_list: fragments
    fragment_nrs = split_fr_nrs(len(fr_list), boundary_nrs)
    return fragment_nrs, boundary_nrs
