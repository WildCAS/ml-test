import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import argparse
import csv
import pickle
import os

_PARSER = argparse.ArgumentParser(description='Processes training set.')
_PARSER.add_argument('-f', '--file', type=str, required=True,
                     help='filename containing training set or pickled file')
_PARSER.add_argument('-s', '--save', action='store_true',
                     help='save the pickled data to particular file')
_PARSER.add_argument('-l', '--load', action='store_true',
                     help='load the pickled data from particular file')


def load(filename):
    """Loads elements from Pickle file

       Args:
       - Filename - str of location

       Return:
        Tuple with:
         - Lists of Activity Names
         - Lists of Activity Types
    """
    assert type(filename) == str, 'Filename must be a str'
    with open(filename, 'rb') as loader:
        elements = pickle.load(loader)

    assert type(elements) == tuple, 'Element is not valid!'
    assert len(elements) == 2, 'Element is not valid!'
    assert len(elements[0]) == len(elements[1]), 'Element is not valid!'
    assert type(elements[0]) == list, 'Element is not valid!'
    assert type(elements[1]) == list, 'Element is not valid!'

    return elements


def save(elements, filename):
    """Pickles elements to a particular filename

       Args:
       - elements: tuple(list, list)
       - filename: str of location to save serialization

       Return:
        None
    """
    assert type(filename) == str, 'Filename must be a str'
    assert filename.endswith('.pkl'), 'File must be pkl!'
    assert type(elements) == tuple, 'Element must be a tuple!'
    assert len(elements) == 2, 'Tuple must be of length 2!'
    assert len(elements[0]) == len(elements[1]), 'Elements not equal length!'
    assert type(elements[0]) == list, 'First element must be a list!'
    assert type(elements[1]) == list, 'Second element must be a list!'
    with open(filename, 'wb') as dump:
        pickle.dump(elements, dump)
    return


def extractor(filename):
    """Return a tuple containing two lists of the Activity title and
       Activity type(s)

       Args:
       - filename: string of the filename

       Returns:
        Tuple with:
         - Lists of Activity Names
         - Lists of Activity Types
    """
    assert filename.endswith('.csv'), 'File must be a csv file!'

    activity_names = []
    activity_types = []
    with open(filename, 'r') as cas_file:
        cas_reader = csv.reader(cas_file, delimiter=',')
        for row in cas_reader:
            activity_names.append(row[0])
            activity_types.append([int(tp) for tp in row[1:]])
    print(activity_types)
    vectors = np.empty(shape=(len(activity_names),3))
    for i in range(len(activity_names)):
        vectors[i] = activity_types[i]
    return (activity_names, vectors)

def learn(activity_names, activity_types):
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])
    classifier.fit(activity_names, activity_types)
    return classifier

def test_learn(classifier, test_data):
    categories = ['Creativity', 'Action', 'Service']
    predicted = classifier.predict(test_data)
    for item, labels in zip(test_data, predicted):
        print('{0} => {1}'.format(item, labels))

def main():
    """Main function of utils.py"""
    parse_args = _PARSER.parse_args()
    if parse_args.save:
        csv_file = parse_args.file
        data = extractor(csv_file)
        save(data, os.path.realpath(os.path.join(os.path.dirname(csv_file),
                                    'data.pkl')))
    elif parse_args.load:
        pkl_file = parse_args.file
        data = extractor(pkl_file)
        classifier = learn(data[0], data[1])
        test_learn(classifier, extractor("test-data.csv")[0])
    else:
        print('Nothing happened')


if __name__ == '__main__':
    main()