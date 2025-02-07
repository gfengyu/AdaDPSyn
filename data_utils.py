# Imported from https://github.com/tonyzhaozh/few-shot-learning/blob/main/data_utils.py
import pandas as pd
import json
import pickle
import numpy as np
import random

def load_agnews():
    train_data = pd.read_csv(f'./data/agnews/train.csv')
    test_data = pd.read_csv(f'./data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index'])
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]

    return train_sentences, train_labels, test_sentences, test_labels

def load_dbpedia():
    train_data = pd.read_csv(f'./data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'./data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])

    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'./data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'./data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def load_movie_director():
    train_data = pd.read_csv(f'./data/movie/Director/train.csv')
    test_data = pd.read_csv(f'./data/movie/Director/test.csv')

    train_sentences = train_data['content']
    train_labels = list(train_data['label'])

    test_sentences = test_data['content']
    test_labels = list(test_data['label'])

    return train_sentences, train_labels, test_sentences, test_labels

def load_movie_genre():
    train_data = pd.read_csv(f'./data/movie/Genre/train.csv')
    test_data = pd.read_csv(f'./data/movie/Genre/test.csv')

    train_sentences = train_data['content']
    train_labels = list(train_data['label'])

    test_sentences = test_data['content']
    test_labels = list(test_data['label'])

    return train_sentences, train_labels, test_sentences, test_labels



