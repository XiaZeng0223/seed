import os
from sentence_transformers import SentenceTransformer, models
import jsonlines
import random
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetBuilder
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import dot, mean, absolute
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm

import sys
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert-base-uncased')   #nli-distilroberta-base-v2
parser.add_argument('--load_model_from_disk', type=bool, default=False)
parser.add_argument('--corpus', type=str, default="data/scifact/corpus.jsonl")
parser.add_argument('--train_set', type=str, default="data/scifact/claims_train.jsonl")
parser.add_argument('--dev_set', type=str, default="data/scifact/claims_dev.jsonl")
parser.add_argument('--norm', type=bool, default=False)
parser.add_argument('--abs', type=bool, default=True)
parser.add_argument('--dis', type=str, default='euclidean')
parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--m', type=int, default=20)     # number of fit samples per class
parser.add_argument('--output', type=str, default="output/scifact")


args = parser.parse_args()

def distance(a, b):
    '''Euclidean distance'''
    if args.dis == 'euclidean':
        dist = 1 - norm(a - b)
    elif args.dis == 'cosine':
        dist = dot(a, b)/(norm(a)*norm(b))
    return dist


def predict_dis(mean_vecs, X_dev_sampled):
    # make predictions based on Euclidean distance.
    # This works because the Euclidean distance is the l2 norm, and the default value of the ord parameter in numpy.linalg.norm is 2.
    y_list = []
    for diff in tqdm(X_dev_sampled):
        similarity_0 = distance(diff, mean_vecs[0])
        similarity_1 = distance(diff, mean_vecs[1])
        similarity_2 = distance(diff, mean_vecs[2])
        y_hat = np.array([similarity_0, similarity_1, similarity_2]).argmax()
        y_list.append(y_hat)
    return y_list


def evaluate(mean_vecs, X_dev, y_truth):

    # print(Counter(y_truth))

    # print('Euclidean distance results:')
    y_pred = predict_dis(mean_vecs, X_dev)
    y_pred = ['s' if i == 0 else 'n' if i ==1 else 'c' for i in y_pred]   # 0:s, 1:n, 2:c
    y_truth = ['s' if i == 0 else 'n' if i ==1 else 'c' for i in y_truth]

    print('Accuracy:', round(accuracy_score(y_truth, y_pred), 4))
    print('F1-macro:', f1_score(y_truth, y_pred, average=None, labels=["c", "n", "s"]).round(4))
    print('F1-macro:', f1_score(y_truth, y_pred, average=None).round(4))

    print("Confusion Matrix:")
    print("s", "n", "c")
    print(confusion_matrix(y_truth, y_pred, labels=["s", "n", "c"]))
    wrong_index =[]
    for i in range(len(y_truth)):
        if y_truth[i] != y_pred[i]:
            wrong_index.append(i)
    # print(len(wrong_index), wrong_index)
    wrong_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in wrong_index]
    table = pd.DataFrame(wrong_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
    # print(table)
    os.makedirs(args.output, exist_ok=True )
    table.to_csv("{}/wrong.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')

    correct_index =[]
    for i in range(len(y_truth)):
        if y_truth[i] == y_pred[i]:
            correct_index.append(i)
    correct_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in correct_index]
    table = pd.DataFrame(correct_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
    # print(table)
    os.makedirs(args.output, exist_ok=True )
    table.to_csv("{}/correct.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')


def diff(claims, evidences, model, abs):
    claim_embeddings = model.encode(claims)
    evidence_embeddings = model.encode(evidences)
    if abs == True:
        # print('calculating diff with abs')
        diffs = absolute(evidence_embeddings - claim_embeddings)
    else:
        # print('calculating diff without abs')
        diffs = evidence_embeddings - claim_embeddings
    return diffs

def read_scifact(corpus, dataset):
    label_encodings = {'SUPPORT': 0, 'NOT ENOUGH INFO': 1, 'CONTRADICT': 2}
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
    dataset = jsonlines.open(dataset)
    claims = []; evidences = []; labels = []
    random.seed(0)
    for data in dataset:
        claim = data['claim']
        if data['evidence']=={}:
            for doc_id in data['cited_doc_ids']:
                doc = corpus[int(doc_id)]
                indices = random.sample(range(len(doc['abstract'])), k=1)       # There is no gold rationales, so we randomly select some sentences.
                evidence = " ".join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                evidences.append(evidence)
                labels.append(1)    # neutral
                claims.append(claim)
        else:
            for doc_id in data['evidence'].keys():
                indices = data['evidence'][doc_id][0]["sentences"]
                evidence = ' '.join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                evidences.append(evidence)
                claims.append(claim)
                labels.append(label_encodings[data['evidence'][doc_id][0]["label"]])
    dataset = Dataset.from_dict({'claim':claims, 'evidence':evidences, 'label':labels})
    return dataset

if __name__ == '__main__':

    abs = args.abs
    seed = args.seed
    # print(args.model, seed, abs, args.dis)


    if args.load_model_from_disk:
        word_embedding_model = models.Transformer(args.model)
        pooling_model = models.Pooling(word_embedding_dimension = word_embedding_model.get_word_embedding_dimension(), pooling_mode = args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(args.model)

    # read scifact train datset as fit set here
    trainset = read_scifact(args.corpus, args.train_set)
    m = args.m
    trainset_s = trainset.filter(lambda example: example['label'] == 0)
    trainset_n = trainset.filter(lambda example: example['label'] == 1)
    trainset_c = trainset.filter(lambda example: example['label'] == 2)
    trainset_s_sampled = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:m])
    trainset_n_sampled = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:m])
    trainset_c_sampled = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:m])
    print('full trainset length:', len(trainset_s_sampled), len(trainset_n_sampled), len(trainset_c_sampled))
    trainset_sampled = concatenate_datasets([trainset_s_sampled, trainset_n_sampled, trainset_c_sampled])


    # read scifact dev datset
    n = 70
    devset = read_scifact(args.corpus, args.dev_set)
    devset_s = devset.filter(lambda example: example['label'] == 0)
    devset_n = devset.filter(lambda example: example['label'] == 1)
    devset_c = devset.filter(lambda example: example['label'] == 2)
    devset_s_sampled = Dataset.from_dict(devset_s.shuffle(seed=seed)[:n])
    devset_n_sampled = Dataset.from_dict(devset_n.shuffle(seed=seed)[:n])
    devset_c_sampled = Dataset.from_dict(devset_c.shuffle(seed=seed)[:n])
    devset_sampled = concatenate_datasets([devset_s_sampled, devset_n_sampled, devset_c_sampled])
    print('full devset length:', len(devset_s_sampled), len(devset_n_sampled), len(devset_c_sampled))

    vec_scifact = []
    for set in [trainset_s_sampled, trainset_n_sampled, trainset_c_sampled]:
        diff_ = diff(set['claim'], set['evidence'], model, abs)
        vec_scifact.append(mean(diff_, axis=0))

    X_dev_scifact = diff(devset_sampled['claim'], devset_sampled['evidence'], model, abs)
    y_truth_scifact = devset_sampled['label']
    #
    #
    print("scifact --> scifact:")
    # evaluating scifact dev on scifact vectors
    evaluate(vec_scifact, X_dev_scifact, y_truth_scifact)
