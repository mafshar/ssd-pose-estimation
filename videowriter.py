#!/usr/bin/env python

import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join


def sort_files(files):
    ret = []
    # sort files numerically
    for file in files:
        num = int(file.split(".png")[0])
        ret.append(num)
    ret.sort()
    ret = [str(f) + ".png" for f in ret]
    return ret


def run():
    directories = ['./data/op_flow1', './data/op_flow2', './data/op_flow3']
    count = 1

    for dir in directories:
        video_name = str(count) + "_mov.avi"
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1080, 720))

        # read in all files
        files = sort_files([f for f in listdir(dir) if isfile(join(dir,f))])

        for file in files:
            img = cv2.imread(join(dir, file))
            video.write(img)

        print "Saving video"
        video.release()
        count += 1

if __name__ == '__main__':
    run()
