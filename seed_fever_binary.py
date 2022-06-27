from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetBuilder
import argparse
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import dot, mean, absolute
from numpy.linalg import norm
import jsonlines
from tqdm import tqdm
import pandas as pd
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
parser.add_argument('--model', type=str, default='bert-base-nli-mean-tokens')   #nli-distilroberta-base-v2
parser.add_argument('--load_model_from_disk', type=bool, default=False)
parser.add_argument('--norm', type=bool, default=False)
parser.add_argument('--abs', type=bool, default=True)
parser.add_argument('--dis', type=str, default='euclidean')
parser.add_argument('--both', type=bool, default=False)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_set', type=str, default="data/fever/fever_test.jsonl")
parser.add_argument('--train_set_ppl', type=str, default="data/fever2/fever_test.jsonl")
parser.add_argument('--m', type=int, default=20)     # number of fit samples per class
parser.add_argument('--output', type=str, default="output/binary/test")
parser.add_argument('--ppl_data', type=bool, default=True)





args = parser.parse_args()



def distance(a, b):
    '''Euclidean distance'''
    if args.dis == 'euclidean':
        dist = 1 - norm(a - b)
    elif args.dis == 'cosine':
        dist = dot(a, b)/(norm(a)*norm(b))
    return dist


def predict_dis(mean_vecs, X_dev_sampled, both):
    # make predictions based on Euclidean distance.
    # This works because the Euclidean distance is the l2 norm, and the default value of the ord parameter in numpy.linalg.norm is 2.
    y_list = []
    if both == True:
        for diff in tqdm(X_dev_sampled):
            similarity_0 = distance(diff, mean_vecs[0])
            similarity_1 = distance(diff, mean_vecs[1])
            similarity_2 = distance(diff, mean_vecs[2])
            similarity_00 = distance(diff, mean_vecs[3])
            similarity_11 = distance(diff, mean_vecs[4])
            similarity_22 = distance(diff, mean_vecs[5])
            y_hat = np.array(
                [similarity_0, similarity_1, similarity_2, similarity_00, similarity_11, similarity_22]).argmax()
            y_list.append(y_hat)
        y_list = [i - 3 if i > 2 else i for i in y_list]
    else:
        for diff in tqdm(X_dev_sampled):
            similarity_0 = distance(diff, mean_vecs[0])
            similarity_1 = distance(diff, mean_vecs[1])
            y_hat = np.array([similarity_0, similarity_1]).argmax()
            y_list.append(y_hat)
    return y_list


def evaluate(mean_vecs, X_dev, y_truth, both):
    #
    # print(Counter(y_truth))
    # print('Euclidean distance results:')
    y_pred = predict_dis(mean_vecs, X_dev, both)
    y_pred = ['s' if i == 0 else 'not' for i in y_pred]   # 0:s, 1:n, 2:c
    print('Accuracy:', round(accuracy_score(y_truth, y_pred), 4))
    print('F1-macro:', f1_score(y_truth, y_pred, average='macro').round(4))
    print('F1-labelwise:', f1_score(y_truth, y_pred, average=None, labels=['s', 'not']).round(4))

    print("Confusion Matrix: 's', 'not'")
    print(confusion_matrix(y_truth, y_pred, labels=['s', 'not']))
    wrong_index =[]
    for i in range(len(y_truth)):
        if y_truth[i] != y_pred[i]:
            wrong_index.append(i)
    # print(len(wrong_index), wrong_index)
    wrong_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in wrong_index]
    table = pd.DataFrame(wrong_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
    # print(table)
    os.makedirs(args.output, exist_ok=True )
    table.to_csv("{}/fever_binary_wrong.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')

    correct_index =[]
    for i in range(len(y_truth)):
        if y_truth[i] == y_pred[i]:
            correct_index.append(i)
    correct_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in correct_index]
    table = pd.DataFrame(correct_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
    # print(table)
    table.to_csv("{}/fever_binary_correct.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')

def diff(claims, evidences, model, abs):
    claim_embeddings = model.encode(claims)
    evidence_embeddings = model.encode(evidences)
    if abs == True:
        print('calculating diff with abs')
        diffs = absolute(evidence_embeddings - claim_embeddings)
    else:
        print('calculating diff without abs')
        diffs = evidence_embeddings - claim_embeddings
    return diffs

def fever_data_cleaning(sent):
    sent = sent.replace('-LRB-', '(')
    sent = sent.replace('-RRB-', ')')
    sent = sent.replace('-LSB-', '[')
    sent = sent.replace('-RSB-', ']')
    return sent

def read_ppl_fever(data_set):
    label_encodings = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 1, 'REFUTES': 2}
    claims = []
    evidences = []
    labels = []     # s:0, n:1, c: 2
    for obj in jsonlines.open(data_set):
        evidence = fever_data_cleaning(obj['evidences'][0][0]).lower().strip()
        evidences.append(evidence)
        labels.append(label_encodings[obj['label']])
        claims.append(fever_data_cleaning(obj['claim'].lower().strip()))
    dataset = Dataset.from_dict({'claim':claims, 'evidence':evidences, 'label':labels})
    return dataset

def read_2fever(data_set):

    label_encodings = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 1, 'REFUTES': 2}
    claims = []
    evidences = []
    labels = []     # s:0, n:1, c: 2
    for data in jsonlines.open(data_set):
        for evidence_set in data['evidence_sets']:
            if len(evidence_set)==1:
                evidence =' '.join([data['sentences'][i] for i in evidence_set])
                evidences.append(evidence)
                labels.append(label_encodings[data['label']])
                claims.append(data['claim'])
    dataset = Dataset.from_dict({'claim':claims, 'evidence':evidences, 'label':labels})
    return dataset

if __name__ == '__main__':

    abs = args.abs
    both = args.both
    seed = args.seed
    # print(args.model, abs, both, args.dis)

    if args.load_model_from_disk:
        word_embedding_model = models.Transformer(args.model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(args.model)


    # # read fever train dataset
    if args.ppl_data ==True:
        trainset = read_ppl_fever(args.train_set_ppl)
        trainset_s = trainset.filter(lambda example: example['label'] == 0)
        trainset_n = trainset.filter(lambda example: example['label'] == 1)
        trainset_c = trainset.filter(lambda example: example['label'] == 2)
        print(len(trainset_s), len(trainset_n), len(trainset_c))
        # trainset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:3333])
        # trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:1666])
        # trainset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:1667])
        trainset_not = concatenate_datasets(
            [trainset_n, trainset_c])
    else:
        trainset = read_2fever(args.train_set)
        trainset_s = trainset.filter(lambda example: example['label'] == 0)
        trainset_n = trainset.filter(lambda example: example['label'] == 1)
        trainset_c = trainset.filter(lambda example: example['label'] == 2)
        trainset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:3333])
        trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:1666])
        trainset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:1667])
        trainset_not = concatenate_datasets(
            [trainset_n, trainset_c])

    print('balanced dataset sample counts:', len(trainset_s), len(trainset_n), len(trainset_c), len(trainset_not))
    m = args.m
    trainset_s_sampled = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:m])
    trainset_not_sampled = Dataset.from_dict(trainset_not.shuffle(seed=seed)[:m])

    # use the rest as dev set
    devset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[m:])
    devset_not = Dataset.from_dict(trainset_not.shuffle(seed=seed)[m:])
    print('full devset length:', len(devset_s), len(devset_not))
    devset_sampled = concatenate_datasets([devset_s, devset_not])

    vec_fever = []
    for set in [trainset_s_sampled, trainset_not_sampled]:
        diff_ = diff(set['claim'], set['evidence'], model, abs)
        vec_fever.append(mean(diff_, axis=0))

    X_dev_fever = diff(devset_sampled['claim'], devset_sampled['evidence'], model, abs)
    y_truth_fever = ['s' if i == 0 else 'not' for i in devset_sampled['label']]


    print("fever --> fever:")
    evaluate(vec_fever, X_dev_fever, y_truth_fever, both=False)
