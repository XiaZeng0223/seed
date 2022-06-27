from sentence_transformers import SentenceTransformer, models
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
import jsonlines
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy import dot, mean, absolute
from numpy.linalg import norm
import jsonlines
from tqdm import tqdm
import math
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
parser.add_argument('--m', type=int, default=20)     # number of fit samples per class
parser.add_argument('--output', type=str, default="output")



args = parser.parse_args()


def merge_sentence(example):
    example['sentences'] = ' '.join(example['sentences'])
    return example
#
# def distance(a, b, dis):
#     '''Euclidean distance'''
#     if dis == 'euclidean':
#         dist = 1 - norm(a - b)
#     elif dis == 'cosine':
#         dist = 1-dot(a, b)/(norm(a)*norm(b))
#     return dist


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
            similarity_2 = distance(diff, mean_vecs[2])
            y_hat = np.array([similarity_0, similarity_1, similarity_2]).argmax()
            y_list.append(y_hat)
    return y_list


def evaluate(mean_vecs, X_dev, y_truth, both):

    # print(Counter(y_truth))

    # print('Euclidean distance results:')
    y_pred = predict_dis(mean_vecs, X_dev, both)
    y_pred = ['s' if i == 0 else 'n' if i ==1 else 'c' for i in y_pred]   # 0:s, 1:n, 2:c
    print('Accuracy:', round(accuracy_score(y_truth, y_pred), 4))
    print('F1-macro:', round(f1_score(y_truth, y_pred, average='macro'), 4))
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
    table.to_csv("{}/fever_wrong.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')

    correct_index =[]
    for i in range(len(y_truth)):
        if y_truth[i] == y_pred[i]:
            correct_index.append(i)
    correct_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in correct_index]
    table = pd.DataFrame(correct_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
    # print(table)
    table.to_csv("{}/fever_correct.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')


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

def read_fever(data_set):
    label_encodings = {'SUPPORTS': 0, 'NOT ENOUGH INFO': 1, 'REFUTES': 2}
    claims = []
    evidences = []
    labels = []     # s:0, n:1, c: 2
    for data in jsonlines.open(data_set):
        for evidence_set in data['evidence_sets']:
            evidence =' '.join([data['sentences'][i] for i in evidence_set])
            evidences.append(evidence)
            labels.append(label_encodings[data['label']])
            claims.append(data['claim'])
    dataset = Dataset.from_dict({'claim':claims, 'evidence':evidences, 'label':labels})
    return dataset

def Cosine(vec1, vec2) :
    result = InnerProduct(vec1,vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return result

def VectorSize(vec) :
    return math.sqrt(sum(math.pow(v,2) for v in vec))

def InnerProduct(vec1, vec2) :
    return sum(v1*v2 for v1,v2 in zip(vec1,vec2))

def Euclidean(vec1, vec2) :
    return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))

def Theta(vec1, vec2) :
    return math.acos(Cosine(vec1,vec2)) + math.radians(10)

def Triangle(vec1, vec2) :
    theta = math.radians(Theta(vec1,vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2

def Magnitude_Difference(vec1, vec2) :
    return absolute(VectorSize(vec1) - VectorSize(vec2))

def Sector(vec1, vec2) :
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED+MD),2) * theta/360

def TS_SS(vec1, vec2) :
    return Triangle(vec1, vec2) * Sector(vec1, vec2)



def distance(a, b, dis):
    '''Euclidean distance'''
    if dis == 'euclidean':
        # dist = 1 - norm(a - b)
        dist = Euclidean(a, b)
    elif dis == 'cosine':
        # dist = dot(a, b)/(norm(a)*norm(b))
        dist = Cosine(a, b)
    elif dis =='triangle':
        dist = TS_SS(a, b)
    return dist

if __name__ == '__main__':

    abs = args.abs
    both = args.both
    seed = args.seed
    # print(args.model, seed, abs, args.dis)

    if args.load_model_from_disk:
        word_embedding_model = models.Transformer(args.model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(args.model)


    # # read fever train dataset
    trainset = read_fever(args.train_set)
    trainset_s = trainset.filter(lambda example: example['label'] == 0)
    trainset_n = trainset.filter(lambda example: example['label'] == 1)
    trainset_c = trainset.filter(lambda example: example['label'] == 2)
    # trainset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:333])
    # trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:333])
    # trainset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:333])
    trainset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:3333])
    trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:3333])
    trainset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:3333])

    # print('balanced dataset sample counts:', len(trainset_s), len(trainset_n), len(trainset_c))

    # shots = [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    # shots = range(5, 105, 5)
    shots = range(1, 101)
    print([x for x in shots])
    cosine_dis = {}
    euclidean_dis = {}
    change_effect = {}
    tsss_dis = {}
    m = shots[0]
    trainset_s_sampled = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:m])
    trainset_n_sampled = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:m])
    trainset_c_sampled = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:m])

    vec_fever_m = []
    for set in [trainset_s_sampled, trainset_n_sampled, trainset_c_sampled]:
        diff_ = diff(set['claim'], set['evidence'], model, abs)
        vec_fever_m.append(mean(diff_, axis=0))
    for i in range(1, len(shots)):

        n = shots[i]
        trainset_s_sampled_ = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:n])
        trainset_n_sampled_ = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:n])
        trainset_c_sampled_ = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:n])

        vec_fever_n = []
        for set in [trainset_s_sampled_, trainset_n_sampled_, trainset_c_sampled_]:
            diff_ = diff(set['claim'], set['evidence'], model, abs)
            vec_fever_n.append(mean(diff_, axis=0))

        euclidean_dis['{}_{}'.format(m, n)] = [m, n] + [distance(vec_fever_m[j], vec_fever_n[j], dis='euclidean') for j in
                                               range(3)]

        vec_fever_m=vec_fever_n; m=n



    # print('euclidean dis')
    table= pd.DataFrame.from_dict(euclidean_dis, orient='index', columns=['start', 'end', 'Support', 'Neutral', 'Contradict'])
    table.to_csv("euclidean_dis.csv".format(), mode='w', sep='\t', encoding='utf-8', float_format='%.3f')


