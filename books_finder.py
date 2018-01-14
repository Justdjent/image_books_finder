import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import os
from collections import Counter
import argparse
import logging
import sys


logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt=' %I:%M:%S ', level="INFO")


def descriptor(img):
    # SIFT
    #     descriptor = cv2.xfeatures2d.SIFT_create()
    #     kp2, des2 = descriptor.detectAndCompute(img2,None)

    # Initiate STAR detector
    star = cv2.xfeatures2d.StarDetector_create()

    # Initiate LATCH extractor
    latch = cv2.xfeatures2d.LATCH_create()

    # find the keypoints with STAR
    kp = star.detect(img, None)

    # compute the descriptors with BRIEF
    kp, des = latch.compute(img, kp)
    return kp, des


def find_points(img_folder):
    files = os.listdir(img_folder)
    point_dict = {}
    for image in files:

        if not image.endswith(".jpg"):
            logging.info("{} not an jpg file".format(image))
            continue

        im_path = os.path.join(img_folder, image)
        img1 = cv2.imread(im_path, 0)  # queryImage
        kp1, des1 = descriptor(img1)
        point_dict[image] = (kp1, des1)

    return point_dict


def find_books(image, lib):
    result = []
    g_t = np.ones(18)
    books_on_image = []
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img2 = cv2.imread(image, 0)  # trainImage
    kp2, des2 = descriptor(img2)

    # create clusterer object
    clusterer = DBSCAN(eps=200)

    for key, value in lib.items():

        kp1, des1 = value[0], value[1]

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Initialize lists
        list_kp1 = []
        list_kp2 = []

        # For each match in top 20...
        for mat in matches[:20]:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

        labels = clusterer.fit_predict(list_kp2)

        if Counter(labels)[0] >= 7:
            logging.info('{} book is on the image'.format(key))
            result.append(1)
            books_on_image.append(key)
        else:
            result.append(0)

    logging.info(books_on_image)
    return books_on_image


if __name__ == "__main__":

    # Adding arguments to argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to photo", type=str)
    parser.add_argument("-f", "--covers_folder", help="path to folder that contain all book covers", type=str)

    args = parser.parse_args()

    image = args.image
    img_folder = args.covers_folder

    dict_of_points = find_points(img_folder)
    # print(dict_of_points.keys())
    find_books(image, dict_of_points)