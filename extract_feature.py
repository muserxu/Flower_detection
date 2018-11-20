# keras imports
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import glob
import h5py
import os

# the function convert oxford's 17 flower species data to feature descriptors using pretrained model InceptionV3
def extract():
    # craete pretrained models
    IV3 = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
    model = Model(input=IV3.input, output=IV3.get_layer('custom').output)

    train_labels = os.listdir('dataset/train')
    
    features = []
    labels = []

    for i, label in enumerate(train_labels):
        cur_path = 'dataset/train/' + label
        # glob.glob returns a list of files that has this format
        for image_path in glob.glob(cur_path + "/*.jpg"):
            # load as a PIL image with size 299x299x3
            img = image.load_img(image_path, target_size=(299, 299))
            # convert to numpy array
            x = image.img_to_array(img)
            # insert a new row axis because first dimension is used for the number of samples in kera preprocess
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            # get the feature descriptor of this image using pretrained model
            feature = model.predict(x)
            # flatten the data to insert into array to create 2d array in the future
            flat = feature.flatten()
            features.append(flat)
            labels.append(label)

    # encode label from text to number starting from 0
    l = LabelEncoder()
    labels = l.fit_transform(labels)

    # save features and labels
    # use h5py to manipulate huge dimension descriptor well
    h5f_data = h5py.File('features.h5', 'w')
    h5f_data.create_dataset('feature', data=np.array(features))

    h5f_label = h5py.File('labels.h5', 'w')
    h5f_label.create_dataset('label', data=np.array(labels))

    h5f_data.close()
    h5f_label.close()


if __name__ == '__main__':
    extract()