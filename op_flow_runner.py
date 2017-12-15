#!/usr/bin/env python

import numpy as np
import cv2

cap = cv2.VideoCapture("./data/bobst_vid.mov")
ret, frame = cap.read()
# params for ShiTomasi corner detection

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
# ret, old_frame = cap.read()
# filename = './data/figure_1_reg.png'
old_frame = frame
# print old_frame.shape
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 =    np.array([
        [np.array([858.,239.])], ## left shoulder
        [np.array([952.,239.])], ## right shoulder
        [np.array([909.,239.])], ## center chest
        [np.array([899.,174.])], ## left eye
        [np.array([917.,174.])], ## right eye
        [np.array([890.,385.])], ## left hip
        [np.array([942.,385.])], ## right hip
        [np.array([850.,321.])], ## left elbow
        [np.array([973.,321.])]], ## right elbow
        np.float32)

# for item in p0:
#     item = np.ndarray(item)
#     for it in item:
#         it = np.ndarray(it)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# filenames = ['./data/figure_2_reg.png', './data/figure_3_reg.png', './data/figure_4_reg.png']

count = 1
while cap.isOpened():
    # ret,frame = cap.read()
    re, frame = cap.read()
    # frame = cv2.imread(file)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imwrite('./data/op_flow3/' + str(count) + '.png', img)
    count += 1
    # cv2.imshow('frame',img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
# cv2.destroyAllWindows()
# cap.release()
