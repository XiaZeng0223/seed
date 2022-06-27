import argparse
import torch
import jsonlines
import random
import os
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetBuilder
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="fever")
parser.add_argument('--fever_train_set', type=str, default="data/fever/fever_test.jsonl")
parser.add_argument('--train_set_ppl', type=str, default="data/fever2/fever_test.jsonl")
parser.add_argument('--corpus', type=str, default="data/scifact/corpus.jsonl")
parser.add_argument('--scifact_train_set', type=str, default="data/scifact/claims_train.jsonl")
parser.add_argument('--scifact_dev_set', type=str, default="data/scifact/claims_dev.jsonl")
parser.add_argument('--dest', type=str, default='output/123/fever/baseline')
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--model_base', type=str, default='bert')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size-gpu', type=int, default=2, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=32, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=5e-6)
parser.add_argument('--lr-linear', type=float, default=5e-6)
parser.add_argument('--m', type=int, default=20)     # number of train samples per class
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--output', type=str, default="output/123/fever/baseline")


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

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

def label2binary(example):
    mapping = {0: 0, 1: 1, 2: 1}    #merge 'n' and 'c' into 'not'
    example['label']=mapping[example['label']]
    return example

def parse_2fever(fever_path, ppl=True):
    if ppl:
        trainset = read_ppl_fever(args.train_set_ppl)
        trainset_s = trainset.filter(lambda example: example['label'] == 0)
        trainset_n = trainset.filter(lambda example: example['label'] == 1)
        trainset_c = trainset.filter(lambda example: example['label'] == 2)
        trainset_not = concatenate_datasets(
            [trainset_n, trainset_c])
    else:
        trainset = read_2fever(fever_path)
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
    trainset_sampled = concatenate_datasets([trainset_s_sampled, trainset_not_sampled]).map(label2binary)

    # use the rest as dev set
    devset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[m:])
    devset_not = Dataset.from_dict(trainset_not.shuffle(seed=seed)[m:])
    print('full devset length:', len(devset_s), len(devset_not))
    devset_sampled = concatenate_datasets([devset_s, devset_not]).map(label2binary)


    return trainset_sampled, devset_sampled

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

def parse_fever(fever_path):
    # # read fever train dataset
    trainset = read_fever(fever_path)
    trainset_s = trainset.filter(lambda example: example['label'] == 0)
    trainset_n = trainset.filter(lambda example: example['label'] == 1)
    trainset_c = trainset.filter(lambda example: example['label'] == 2)
    trainset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:3333])
    trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:3333])
    trainset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:3333])

    m = args.m
    trainset_s_sampled = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:m])
    trainset_n_sampled = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:m])
    trainset_c_sampled = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:m])
    trainset_sampled = concatenate_datasets(
        [trainset_s_sampled, trainset_n_sampled, trainset_c_sampled])
    print('full trainset length:', len(trainset_s_sampled), len(trainset_n_sampled), len(trainset_c_sampled))

    # use the rest as dev set
    devset_s = Dataset.from_dict(trainset_s.shuffle(seed=seed)[m:])
    devset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[m:])
    devset_c = Dataset.from_dict(trainset_c.shuffle(seed=seed)[m:])
    devset_sampled = concatenate_datasets([devset_s, devset_n, devset_c])
    print('full devset length:', len(devset_s), len(devset_n), len(devset_c))
    return trainset_sampled, devset_sampled

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

def parse_scifact(train_path, dev_path):
    # read scifact train datset as fit set here
    trainset = read_scifact(args.corpus, train_path)
    m= args.m
    trainset_s = trainset.filter(lambda example: example['label'] == 0)
    trainset_n = trainset.filter(lambda example: example['label'] == 1)
    trainset_c = trainset.filter(lambda example: example['label'] == 2)
    trainset_s_sampled = Dataset.from_dict(trainset_s.shuffle(seed=seed)[:m])
    trainset_n_sampled = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:m])
    trainset_c_sampled = Dataset.from_dict(trainset_c.shuffle(seed=seed)[:m])
    trainset_sampled = concatenate_datasets([trainset_s_sampled, trainset_n_sampled, trainset_c_sampled])
    print('full trainset length:', len(trainset_s_sampled), len(trainset_n_sampled), len(trainset_c_sampled))

    # read scifact dev datset
    n=70
    devset = read_scifact(args.corpus, dev_path)
    devset_s = devset.filter(lambda example: example['label'] == 0)
    devset_n = devset.filter(lambda example: example['label'] == 1)
    devset_c = devset.filter(lambda example: example['label'] == 2)
    devset_s_sampled = Dataset.from_dict(devset_s.shuffle(seed=seed)[:n])
    devset_n_sampled = Dataset.from_dict(devset_n.shuffle(seed=seed)[:n])
    devset_c_sampled = Dataset.from_dict(devset_c.shuffle(seed=seed)[:n])
    devset_sampled = concatenate_datasets([devset_s_sampled, devset_n_sampled, devset_c_sampled])
    print('full devset length:', len(devset_s_sampled), len(devset_n_sampled), len(devset_c_sampled))

    return  trainset_sampled, devset_sampled

def encode(claim: List[str], rationale: List[str]):
    encodings = tokenizer(claim, rationale, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return encodings

def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['evidence'])
            logits = model(**encoded_dict.to(device)).logits
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'Accuracy': round(accuracy_score(targets, outputs), 4),
        'F1-macro': round(f1_score(targets, outputs, average='macro'), 4),
        "Confusion Matrix":confusion_matrix(targets, outputs)
    }

def final_evaluate(model, dataset, save=False):
    model.eval()
    y_truth = []
    y_pred = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['evidence'])
            logits = model(**encoded_dict.to(device)).logits
            y_truth.extend(batch['label'].float().tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())

    if args.dataset == '2fever':
        y_pred = ['s' if i == 0 else 'not' for i in y_pred]  # 0:s, 1:not
        y_truth = ['s' if i == 0 else 'not' for i in y_truth]  # 0:s, 1:not
        print('F1-labelwise:', f1_score(y_truth, y_pred, average=None, labels=['s', 'not']).round(4))
        print('Accuracy:', round(accuracy_score(y_truth, y_pred), 4))
        print('F1-macro:', f1_score(y_truth, y_pred, average='macro').round(4))
        print("Confusion Matrix:")
        print("s", "not")
        print(confusion_matrix(y_truth, y_pred, labels=["s", "not"]))
    else:
        y_pred = ['s' if i == 0 else 'n' if i == 1 else 'c' for i in y_pred]  # 0:s, 1:n, 2:c
        y_truth = ['s' if i == 0 else 'n' if i == 1 else 'c' for i in y_truth]  # 0:s, 1:n, 2:c
        print('F1-labelwise:', f1_score(y_truth, y_pred, average=None, labels=['c', 'n', 's']).round(4))
        print('Accuracy:', round(accuracy_score(y_truth, y_pred), 4))
        print('F1-macro:', f1_score(y_truth, y_pred, average='macro').round(4))
        print("Confusion Matrix:")
        print("s", "n", "c")
        print(confusion_matrix(y_truth, y_pred, labels=["s", "n", "c"]))
    if save:
        wrong_index = []
        for i in range(len(y_truth)):
            if y_truth[i] != y_pred[i]:
                wrong_index.append(i)
        # print(len(wrong_index), wrong_index)
        wrong_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in
                       wrong_index]
        table = pd.DataFrame(wrong_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
        # print(table)
        table.to_csv("{}/fever_wrong.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')

        correct_index = []
        for i in range(len(y_truth)):
            if y_truth[i] == y_pred[i]:
                correct_index.append(i)
        correct_preds = [[i, devset_sampled[i]['claim'], devset_sampled[i]['evidence'], y_truth[i], y_pred[i]] for i in
                         correct_index]
        table = pd.DataFrame(correct_preds, columns=['dataset_index', 'claim', 'evidence', 'y_truth', 'y_pred'])
        # print(table)
        os.makedirs(args.output, exist_ok=True)
        table.to_csv("{}/fever_correct.csv".format(args.output), sep='\t', encoding='utf-8', float_format='%.3f')


if __name__ == '__main__':
    seed = args.seed
    if args.dataset == 'fever':
        trainset_sampled, devset_sampled = parse_fever(args.fever_train_set)
        config = AutoConfig.from_pretrained(args.model, num_labels=3)
    elif args.dataset =='scifact':
        trainset_sampled, devset_sampled = parse_scifact(args.scifact_train_set, args.scifact_dev_set)
        config = AutoConfig.from_pretrained(args.model, num_labels=3)
    elif args.dataset =='bfever':
        trainset_sampled, devset_sampled = parse_2fever(args.fever_train_set)
        config = AutoConfig.from_pretrained(args.model, num_labels=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
    if args.model_base == 'bert':
        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}])
    elif args.model_base =='roberta':
            optimizer = torch.optim.Adam([
            {'params': model.roberta.parameters(), 'lr': args.lr_base},
            {'params': model.classifier.parameters(), 'lr': args.lr_linear}])

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs*args.batch_size_accumulated)


    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(trainset_sampled, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['evidence'])
            outputs = model(**encoded_dict.to(device), labels=batch['label'].long().to(device))
            loss=outputs.loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
                scheduler.step()
        # Eval
        train_score = evaluate(model, trainset_sampled)
        print(f'Epoch {e} train score:')
        print("Acc:", train_score["Accuracy"], "F1-macro:", train_score["F1-macro"])
    final_evaluate(model, devset_sampled)


    # # Save
    # save_path = os.path.join(args.dest, f'checkpoint')
    # os.makedirs(save_path)
    # tokenizer.save_pretrained(save_path)
    # model.save_pretrained(save_path)
