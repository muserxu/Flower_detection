import flower_detection
import predict

import glob
import os
from sklearn.preprocessing import LabelEncoder

train_labels = os.listdir('test_set')
    
features = []
labels = []

for i, label in enumerate(train_labels):
    cur_path = 'test_set/' + label
    # glob.glob returns a list of files that has this format
    for image_path in glob.glob(cur_path + "/*.jpg"):
        features.append(image_path)
        labels.append(label)

# encode label from text to number starting from 0
l = LabelEncoder()
labels = l.fit_transform(labels)

rank1 = 0
rank4 = 0

# loop over test data
for (label, features) in zip(labels, features):

    cropped = flower_detection.crop(features)

    predictions = predict.predict(cropped)

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