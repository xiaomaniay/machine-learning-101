import os
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score


# read email files from a folder and constructs a dictionary for all words
def make_Dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary) # get all unique words

    for item in list_to_remove:
        if (not item.isalpha()) or (len(item) == 1): # remove all numerical values
            del dictionary[item]

    dictionary = dictionary.most_common(3000) # consider only 3000 most common words
    return dictionary


# generate a label and word frequency matrix
def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    train_labels = np.zeros(len(files))
    count = 0
    docID = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                words = line.split()
                for word in words:
                    wordID = 0
                    for i, d in enumerate(dictionary):
                        if d[0] == word:
                            wordID = i
                            features_matrix[docID, wordID] = words.count(word)
                            break
            train_labels[docID] = 0;
            filepathTokens = os.path.split(fil)
            lastToken = filepathTokens[len(filepathTokens) - 1]
            if lastToken.startswith("spmsg"):
                train_labels[docID] = 1;
                count = count + 1
            docID = docID + 1
    return features_matrix, train_labels


TRAIN_DIR = "../train-mails"
TEST_DIR = "../test-mails"

print("make_Dictionary start_time: ", datetime.now())
dictionary = make_Dictionary(TRAIN_DIR)
print("make_Dictionary end_time: ", datetime.now())

print("Reading and processing emails from file.")
print("extract_features start_time: ", datetime.now())
features_matrix, labels = extract_features(TRAIN_DIR)
print("extract_features end_time: ", datetime.now())
print("extract_features start_time: ", datetime.now())
test_feature_matrix, test_labels = extract_features(TEST_DIR)
print("extract_features end_time: ", datetime.now())

model = GaussianNB()

print("Training model..")
print("training start_time: ", datetime.now())
model.fit(features_matrix, labels)
print("training end_time: ", datetime.now())

print("predicting start_time: ", datetime.now())
predicted_labels = model.predict(test_feature_matrix)
print("predicting end_time: ", datetime.now())

print("FINISHED classfying. accuracy score: ", end=" ")
print(accuracy_score(test_labels, predicted_labels))
