#!/usr/bin/env python

import os
import sys
import tarfile
import signal
import cv2
import freenect
import frame_convert2

# sys.path.append('/Users/mohammadafshar1/projects/models/research/') # point to your tensorflow dir
# sys.path.append('/Users/mohammadafshar1/projects/models/research/slim') # point ot your slim dir
sys.path.append('/Users/mohammadafshar1/Desktop/SCHOOL/College/NYU/Graduate/Fall_2017/vision-meets-ml/SSD-Tensorflow') # point to your tensorflow dir

import tensorflow as tf
import numpy as np
import six.moves.urllib as urllib
import matplotlib.image as mpimg

from time import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

slim = tf.contrib.slim


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# # Restore SSD model.
ckpt_filename = '../SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
#
# # SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

## WORKS UNTIL HERE

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
# path = '../SSD-Tensorflow/demo/'
# image_names = sorted(os.listdir(path))

# img = mpimg.imread(path + image_names[-5])
# rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
# visualization.plt_bboxes(img, rclasses, rscores, rbboxes)


## write code to get take the kinect images and then convert frames to tf records
## tf records then get fed into the tf-models object recognition pipeline

# cv2.namedWindow('Depth')
# cv2.namedWindow('RGB')
KEEP_RUNNING = True

# def get_depth():
#     return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D np array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a np 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


def run():


    cap = get_video()

    while(True):
        # Capture frame-by-frame
        img = get_video()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rclasses, rscores, rbboxes =  process_image(img)
        fig = visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

        frame  = fig2data(fig)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    exit(0)

    while KEEP_RUNNING:
        # cv2.imshow('Depth', get_depth())
        img = get_video()
        rclasses, rscores, rbboxes =  process_image(img)
        visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        # cv2.imshow('Video', get_video())
        # if cv2.waitKey(10) == 27:
        #     break

    # print('Press ESC in window to stop')
    # freenect.runloop(depth=display_depth,
    #                  video=display_rgb,
    #                  body=body)
    return

if __name__ == '__main__':
    run()
