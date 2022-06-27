import numpy as np
from numpy import dot, mean, absolute
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--svector_results_fever', type=str, default='final_results/multiple_seeds.txt')
parser.add_argument('--svector_results_scifact', type=str, default='final_results/multiple_seeds_scifact.txt')
parser.add_argument('--baseline_results_dir', type=str, default='final_results/')
parser.add_argument('--output_dir', type=str, default='final_graphs')

args = parser.parse_args()
def parse_and_plot(data0, data1, y, title):
    plt.figure()
    plt.title(title)
    ax = plt.gca()
    label_encodings = {'bert_base':'$FT_{BERT_{B}}$',
                       'bert_large':'$FT_{BERT_{L}}$',
                       'roberta_base':'$FT_{RoBERTa_{B}}$',
                       'roberta_large':'$FT_{RoBERTa_{L}}$',
                       'bert-base-nli-mean-tokens':'$VEC_{BERT_{B-NLI}}$',
                       'bert-base-uncased':'$VEC_{BERT_{B}}$',
                       'bert-large-uncased':'$VEC_{BERT_{L}}$'}

    #
    # # baseline models
    # model_list = ['bert_base', 'bert_large', 'roberta_base', 'roberta_large']
    # for model in model_list:
    #     df0 = pd.DataFrame(data0[model], columns=['Random Seed', '# of shots', 'Accuracy', 'F1-macro'])
    #     df0_mean = df0.groupby('# of shots').mean()
    #     del df0_mean['Random Seed']
    #     # print(model)
    #     # print(df0_mean)
    #     if model == 'bert_base':
    #         print('finetuning', model, df0_mean['F1-macro'][10])
    #
    #     df0_mean.plot(label = label_encodings[model], y = y, ax=ax, linestyle='dotted', markevery=1, marker='+')


    # s-vector models
    model_list = ['bert-base-nli-mean-tokens', 'bert-base-uncased']
    df = pd.DataFrame(data1, columns=['# of shots', 'Model Base', 'Random Seed', 'Accuracy', 'F1-macro'])
    # del df['Accuracy']
    grouped_df = df.groupby('Model Base')
    for model in model_list:
        df1= grouped_df.get_group(model)
        df1_mean = df1.groupby('# of shots').mean()
        del df1_mean['Random Seed']
        # print(df1_mean)
        if model == 'bert-base-uncased':
            print('s-vector', model, df1_mean['F1-macro'][10])
        df1_mean.plot(label = label_encodings[model], y = y, ax=ax, markevery=1, marker='o')
    # plt.show()
    plt.savefig('{}/{}.png'.format(args.output_dir, title))


def read_baseline_results_fever(dataset_name):
    baseline_results = {}
    for baseline_model in baseline_models:
        # print(baseline_model)
        with open("{}/{}_baseline/results_{}_o.txt".format(args.baseline_results_dir, dataset_name, baseline_model)) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::5], content[4::5])]
        content = [
            x.replace("Acc: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").split() for x in content]
        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4])] for x in content]
        # print(content)

        average_over_runs = []
        for i in range(0, len(content), 5):
            sum_acc = 0
            sum_f1 = 0
            for j in range(0, 5):
                sum_acc+= content[i+j][3]
                sum_f1+= content[i+j][4]
            average_over_runs.append([content[i][0], content[i][1], sum_acc/5, sum_f1/5])
        # print(average_over_runs)
        #
        baseline_results[baseline_model] = average_over_runs
    return baseline_results

def read_baseline_results_scifact(dataset_name):
    baseline_results = {}
    for baseline_model in baseline_models:
        # print(baseline_model)
        with open("{}/{}_baseline/results_{}_o.txt".format(args.baseline_results_dir, dataset_name, baseline_model)) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::3], content[2::3])]
        content = [
            x.replace("Acc: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").split() for x in content]
        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4])] for x in content]
        # print(content)

        average_over_runs = []
        for i in range(0, len(content), 5):
            sum_acc = 0
            sum_f1 = 0
            for j in range(0, 5):
                sum_acc+= content[i+j][3]
                sum_f1+= content[i+j][4]
            average_over_runs.append([content[i][0], content[i][1], sum_acc/5, sum_f1/5])
        # print(average_over_runs)
        #
        baseline_results[baseline_model] = average_over_runs
    return baseline_results
if __name__ == '__main__':

    # # fever
    baseline_models = ["bert_base", "bert_large", "roberta_base", "roberta_large"]
    baseline_results_fever = read_baseline_results_fever('fever')


    with open(args.svector_results_fever) as f:
        content = f.readlines()
    svector_results = [x.strip() for x in content]

    svector_results_fever = svector_results[svector_results.index('fever_fever')+1:]
    svector_results_fever = [ ' '.join(x) for x in zip(svector_results_fever[0::3], svector_results_fever[1::3], svector_results_fever[2::3] )]
    svector_results_fever = [x.replace("Accuracy: ", "").replace("F1-macro: ", "").split() for x in svector_results_fever]
    svector_results_fever = [[int(x[0]), x[1], int(x[2]), float(x[3]), float(x[4])] for x in svector_results_fever]

    parse_and_plot(baseline_results_fever, svector_results_fever, title='Comparison of Few-Shot Performance (Accuracy) on FEVER Dataset', y='Accuracy')
    parse_and_plot(baseline_results_fever, svector_results_fever, title='Comparison of Few-Shot Performance (F1-macro) on FEVER Dataset', y='F1-macro')

    #
    #
    # baseline_models = ["bert_base", "bert_large", "roberta_base", "roberta_large"]
    #
    # noise = ['2 2 2', '4 4 4', '6 6 6', '8 8 8', '10 10 10', '20 20 20', '30 30 30', '40 40 40', '50 50 50', '100 100 100', 'scifact --> scifact:']
    # #scifact
    # baseline_results_scifact = read_baseline_results_scifact('scifact')
    # with open(args.svector_results_scifact) as f:
    #     content = f.readlines()
    # svector_results = [x.strip() for x in content]
    # svector_results = [x for x in svector_results if x not in noise]
    #
    # print(svector_results)
    # svector_results_scifact = svector_results[svector_results.index('scifact_scifact')+1:]
    # svector_results_scifact = [ ' '.join(x) for x in zip(svector_results_scifact[0::3], svector_results_scifact[1::3], svector_results_scifact[2::3] )]
    # svector_results_scifact = [x.replace("Accuracy: ", "").replace("F1-macro: ", "").split() for x in svector_results_scifact]
    # svector_results_scifact = [[int(x[0]), x[1], int(x[2]), float(x[3]), float(x[4])] for x in svector_results_scifact]
    #
    # parse_and_plot(baseline_results_scifact, svector_results_scifact, title='Comparison of Few-Shot Performance (Accuracy) on SCIFACT Dataset', y='Accuracy')
    # parse_and_plot(baseline_results_scifact, svector_results_scifact, title='Comparison of Few-Shot Performance (F1-macro) on SCIFACT Dataset', y='F1-macro')
    #


