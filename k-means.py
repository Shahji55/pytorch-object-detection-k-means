import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys

cwd = os.getcwd()
print(cwd)

data_folder = cwd + "/inference/images"
print(data_folder)

images = []
original_images = []
for filename in os.listdir(data_folder):
    print("Filename: ", filename)
    img = cv2.imread(os.path.join(data_folder, filename))
#    print(img.shape)
    print("Original dimensions: ", img.size)

    width = 50
    height = 150
    dim = (width, height)

    # resize image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized_img.shape)

    original_images.append(img)

    resized_img = resized_img.flatten()
    print("After flattening: ", resized_img.shape)

    if resized_img is not None:
        images.append(resized_img)

n = 2
kmeans = KMeans(n_clusters=n,init='random')
kmeans.fit(images)
Z = kmeans.predict(images)
print(Z)

for i in range(0, n):
    row = np.where(Z == i)[0]  # row in Z for elements of cluster i
    num = row.shape[0]       # number of elements for each cluster
    r = np.floor(num/10.)    # number of rows in the figure of the cluster

    print("cluster "+str(i))
    print(str(num)+" elements")

first_path = os.getcwd() + '/images_0/'
second_path = os.getcwd() + '/images_1/'

for i in range(0, len(Z)):

    if Z[i] == 0:
        cv2.imwrite(first_path + str(i).zfill(5) + '.png', original_images[i])
    else:
        cv2.imwrite(second_path + str(i).zfill(5) + '.png', original_images[i])





