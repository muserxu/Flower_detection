from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# labels of 17 flowers are in alphabetical order
labels = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']

# return top 3 most likely species of this flower input
def predict(im, topResult=5):
    # load trained classifier
    classifier = pickle.load(open('classifier.pickle', 'rb'))

    # get descriptor of the image we want to predict
    des = get_des(im)

    # return probability of in each class
    predictions = classifier.predict_proba(np.atleast_2d(des))[0] 
    # argsort gives index that produces sorted array
    # [::-1] reads the array from the back because argsort is in ascending order
    predictions = np.argsort(predictions)[::-1][:topResult]

    # display result
    f, axarr = plt.subplots(topResult + 1, sharey=True, figsize=(4,7.5))
    im = cv2.resize(im, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
    axarr[0].imshow(im)
    axarr[0].set_title('Your input')
    axarr[0].axis('off')

    for i in range(topResult):
        pred = predictions[i]
        imm = cv2.imread('default/' + labels[pred] + '.jpg')
        imm = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)
        axarr[i + 1].imshow(imm)
        axarr[i + 1].set_title('Rank ' + str(i + 1) + ': ' + labels[pred])
        axarr[i + 1].axis('off')

    plt.show()

    # test accuracy
    return predictions

# convert the input image into a feature descriptor
def get_des(im):
    im = cv2.resize(im, dsize=(299, 299), interpolation=cv2.INTER_LINEAR)
    # insert a new row axis because first dimension is used for the number of samples 
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)

    # craete pretrained models
    IV3 = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
    model = Model(input=IV3.input, output=IV3.get_layer('custom').output)

    # use the pretrained model to get descriptor of the image
    feature = model.predict(im)
    return feature


if __name__ == '__main__':
    im = cv2.imread('unseen/6.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    predict(im, 3)