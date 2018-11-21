import csv
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append([img,filename])
    return images

with open('train_labels.csv', mode='w', newline = '') as csv_file:
    fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'xmax', 'ymin', 'ymax']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    images = load_images_from_folder('testfolders/train')
    for i in images:
        height, width, ch = i[0].shape
        writer.writerow({'class': 'flower', 'filename': i[1], 'height' : height, 'width' : width, 'xmax' : width, 'xmin' : 0, 'ymax' : height, 'ymin' : 0})




# # import scipy.io
# # import numpy as np
# # data = scipy.io.loadmat("imagelabels.mat")

# # for i in data:
# #     if '__' not in i and 'readme' not in i:
# #             np.savetxt(("file.csv"),data[i],delimiter=',')
