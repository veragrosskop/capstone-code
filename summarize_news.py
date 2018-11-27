#!/usr/bin/env python
'''FINAL Summarization:
Find boundaries and keyframes'''
# coding: utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt

#load functions from files
import shot_detection.shot_boundary_functions as sb
import shot_detection.preprocessing_functions as pre
import shot_detection.summarization_functions as summ

BOUNDARY_FILE = 'shot_detection/rgb_frames/comp_frames/boundary-frame.png'
REPORTER_FILE = 'shot_detection/rgb_frames/comp_frames/news-reporter.png'
END_LOGO_FILE = 'shot_detection/rgb_frames/comp_frames/end-logo.png'

NEWS_VIDEOS1 = ['august-07-2018', 'april-27-2018', 'april-12-2018'] #'april-11-2018', 'april-08-2018']
NEWS_VIDEOS2 = ['june-05-2018', 'february-25-2018', 'august-30-2018', 'august-25-2018']
NEWS_VIDEOS3 = ['september-25-2018', 'september-24-2018', 'september-19-2018', 'september-18-2018'] 
NEWS_VIDEOS4 = ['september-17-2018', 'september-16-2018', 'september-13-2018', 'september-12-2018']
NEWS_VIDEOS5 = ['september-03-2018', 'september-01-2018', 'june-25-2018', 'june-18-2018']

NEWS_VIDEOS = [NEWS_VIDEOS1, NEWS_VIDEOS2, NEWS_VIDEOS3, NEWS_VIDEOS4, NEWS_VIDEOS5]
# 'april-21-2018' first news story is empty?
# 'april-26-2018',

ROOT_PATH = '../data/'

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    method = sys.argv[1]
    boundary = int(sys.argv[2])
    TEST_DATA = NEWS_VIDEOS[int(sys.argv[3])]
    image_tmpl = 'image-{:03d}.png'
    
    if method == 'alexnet':
        color = 'tensor'
        k = -2.2
        reporter_torch = pre.process_single_fr(REPORTER_FILE, color) #load reporter boundary frame
        reporter_method = sb.get_alexnet_features([reporter_torch])[0]
        end_logo_torch = pre.process_single_fr(END_LOGO_FILE, color) #load end logo boundary frame
        end_logo_method = sb.get_alexnet_features([end_logo_torch])[0]
        comp_torch = pre.process_single_fr(BOUNDARY_FILE, color) #load comparison frame
        comp_method = sb.get_alexnet_features([comp_torch])[0]
    elif method == 'histogram':
        color = 'RGB'
        k = -2.2
        reporter_rgb = pre.process_single_fr(REPORTER_FILE, color) #load reporter boundary frame
        reporter_method = plt.hist(reporter_rgb.flatten(), 50)
        end_logo_rgb = pre.process_single_fr(END_LOGO_FILE, color) #load end logo boundary frame
        end_logo_method = plt.hist(end_logo_rgb.flatten(), 50)
        comp_rgb = pre.process_single_fr(BOUNDARY_FILE, color) #load comparison frame
        comp_method = plt.hist(comp_rgb.flatten(), 50)
        plt.close()
    elif method == 'pixel':
        color = 'LA'
        k = -2.2
        reporter_method = pre.process_single_fr(REPORTER_FILE, color) #load reporter boundary frame
        end_logo_method = pre.process_single_fr(END_LOGO_FILE, color) #load end logo boundary frame
        comp_method = pre.process_single_fr(BOUNDARY_FILE, color) #load comparison frame
    
    for news_video in TEST_DATA:
        video_name = news_video
        directory = ROOT_PATH + video_name + '/'

        #load and process news video frames
        if method == 'alexnet':
            frs_amount, frs_list = pre.frames_to_tensor(directory, image_tmpl, 'all')
            method_list = sb.get_alexnet_features(frs_list)
        elif method == 'histogram':
            frs_amount, frs_list = pre.frames_to_float(directory, image_tmpl, color)
            method_list = sb.make_histlist(frs_list)
        elif method == 'pixel':
            frs_amount, method_list = pre.frames_to_float(directory, image_tmpl, color)
        # print(frs_amount)

        # shorten video - start
        sim_frames_reporter, _, _ = sb.compare_frs(method_list, reporter_method, method, -2.0)
        reporter_boundarynr = sim_frames_reporter[-1][0] + 1
        # shorten video - end
        sim_frames_endlogo, _, _ = sb.compare_frs(method_list[reporter_boundarynr::], end_logo_method, method, -2.0)
        end_logo_boundarynr = sim_frames_endlogo[0][0]
        # print("reporter boundary: ", reporter_boundarynr, "\nstart of end logo: ", end_logo_boundarynr)
        method_list = method_list[reporter_boundarynr:end_logo_boundarynr]

        #shot boundary detection
        sim_frames, fr_diff, t_cut_value = sb.compare_frs(method_list, comp_method, method, k)
        boundaries, boundary_nrs = sb.improve_boundaries(sim_frames, method_list)
        fragments_frs_list = sb.split_fr_list(method_list, boundary_nrs) #fragment the method list

        # print(boundary_nrs)
        # print(fragments_frs_list)

        #summarize news fragments
        # print('there are ', len(fragments_frs_list), 'news stories')
        differences, keyframes = summ.report_keyframes_fragments(fragments_frs_list, method, int(boundary))
        fragments_keyframes = summ.keyfr_index_to_image(keyframes, boundary_nrs, reporter_boundarynr)
        print(news_video, '; ', fragments_keyframes)
