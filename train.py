from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle

from sklearn.neighbors.nearest_centroid import NearestCentroid

test_size = 0.2
features_path = 'features.h5'
labels_path = 'labels.h5'
train_path = 'dataset/train'

def train():
    # import features and labels
    h5f_data  = h5py.File(features_path, 'r')
    h5f_label = h5py.File(labels_path, 'r')

    features = h5f_data['feature']
    labels = h5f_label['label']

    features = np.array(features)
    labels = np.array(labels)

    h5f_data.close()
    h5f_label.close()

    # split the training and testing data
    trainData, testData, trainLabels, testLabels = train_test_split(features, labels, test_size=test_size)

    # use logistic regression
    model = LogisticRegression()
    model.fit(trainData, trainLabels)

    # compute centroid
    # clf = NearestCentroid()
    # clf.fit(trainData, trainLabels)

    rank1 = 0
    rank4 = 0

    # loop over test data
    for (label, features) in zip(testLabels, testData):
        # predict the probability of each class label and
        # take the top-5 class labels

        # compute test error for logistic regression
        # return probability of in each class
        predictions = model.predict_proba(np.atleast_2d(features))[0] 
        # argsort gives index that produces sorted array
        # [::-1] reads the array from the back because argsort is in ascending order
        predictions = np.argsort(predictions)[::-1]

        # compute test error for clustering
        # predictions = clf.predict(np.atleast_2d(features))

        # rank 1 prediction increment
        if label == predictions[0]:
            rank1 += 1

        # rank 4 prediction increment
        if label in predictions[:5]:
            rank4 += 1

    # convert accuracies to percentages
    rank1 = (rank1 / float(len(testLabels))) * 100
    rank4 = (rank4 / float(len(testLabels))) * 100

    print('The percentage that the most likely item is the correct flower: ' + str(rank1))
    print('The percentage that top 5 most likely result contains the correct flower: ' + str(rank4))

    # serialize and dump classifier to file
    pickle.dump(model, open('classifier.pickle', 'wb'))

if __name__ == '__main__':
    train()