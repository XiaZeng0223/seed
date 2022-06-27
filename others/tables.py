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

def read_svec_results_fever(mapping, baseline_models, list_of_m):
    baseline_results = {}
    for baseline_model in baseline_models:
        print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::9], content[1::9], content[2::9], content[3::9], content[4::9])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]
        print(content)

        content = [[x[0], int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6])] for x in content]
        content = [instance for instance in content if instance[0] == baseline_model]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_n', 'mean_c', 'std_acc', 'std_s', 'std_n', 'std_c'])
        print(table)
        table.to_csv("svec_{}_macro_f1.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')
        #
    #
    #     average_over_runs = []
    #     for i in range(0, len(content), 5):
    #         sum_acc = 0
    #         sum_f1 = 0
    #         for j in range(0, 5):
    #             sum_acc+= content[i+j][3]
    #             sum_f1+= content[i+j][4]
    #         average_over_runs.append([content[i][0], content[i][1], sum_acc/5, sum_f1/5])
    #     # print(average_over_runs)
    #     #
    #     baseline_results[baseline_model] = average_over_runs
    # return baseline_results

def read_svec_results_bfever(mapping, baseline_models, list_of_m):
    for baseline_model in baseline_models:
        print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::16], content[1::16], content[2::16], content[10::16], content[12::16])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-macro: ", "").replace("F1-labelwise: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]
        print(content)

        content = [[x[0], int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in content]
        content = [instance for instance in content if instance[0] == baseline_model]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_not', 'std_acc', 'std_s', 'std_not'])
        print(table)
        table.to_csv("labelwise/bfever/svec_{}_macro_f1.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')


def read_svec_results_scifact(mapping, baseline_models, list_of_m):
    baseline_results = {}
    for baseline_model in baseline_models:
        print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::13], content[1::13], content[2::13], content[6::13], content[7::13])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]
        print(content)

        content = [[x[0], int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6])] for x in content]
        content = [instance for instance in content if instance[0] == baseline_model]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_n', 'mean_c', 'std_acc', 'std_s', 'std_n', 'std_c'])
        print(table)
        table.to_csv("labelwise/scifact/svec_{}_macro_f1.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')


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


    # baseline models
    model_list = ['bert_base', 'bert_large', 'roberta_base', 'roberta_large']
    for model in model_list:
        df0 = pd.DataFrame(data0[model], columns=['Random Seed', '# of shots', 'Accuracy', 'F1-macro'])
        df0_mean = df0.groupby('# of shots').mean()
        del df0_mean['Random Seed']
        # print(model)
        # print(df0_mean)
        # if model == 'bert_base':
        #     print('finetuning', model, df0_mean['F1-macro'][10])
        df0_mean = df0.groupby('# of shots').mean()
        df0_sd = df0.groupby('# of shots').std()

        del df0_mean['Random Seed']
        del df0_sd['Random Seed']

        df_processed = pd.concat([df0_mean.rename(columns={'Accuracy': 'Accuracy_mean', 'F1-macro': 'F1-macro_mean'}),
                                  df0_sd.rename(columns={'Accuracy': 'Accuracy_sd', 'F1-macro': 'F1-macro_sd'})], axis=1)
        print(df_processed)
        df_processed.to_csv('test/baseline_'+model, sep='\t', encoding='utf-8')
        # df0_mean.plot(label = label_encodings[model], y = y, ax=ax, linestyle='dotted', markevery=1, marker='+')

    #
    # # s-vector models
    # model_list = ['bert-base-nli-mean-tokens', 'bert-large-uncased', 'bert-base-uncased']
    # # model_list = ['bert-large-uncased']
    #
    # df = pd.DataFrame(data1, columns=['# of shots', 'Model Base', 'Random Seed', 'Accuracy', 'F1-macro'])
    # # del df['Accuracy']
    # grouped_df = df.groupby('Model Base')
    # for model in model_list:
    #     df1= grouped_df.get_group(model)
    #     print(df1)
    #     df1_mean = df1.groupby('# of shots').mean()
    #     df1_sd = df1.groupby('# of shots').std()
    #
    #     del df1_mean['Random Seed']
    #     del df1_sd['Random Seed']
    #
    #     print(df1_mean)
    #     print(df1_sd)
    #     df_processed = pd.concat([df1_mean.rename(columns={'Accuracy': 'Accuracy_mean', 'F1-macro': 'F1-macro_mean'}),
    #                               df1_sd.rename(columns={'Accuracy': 'Accuracy_sd', 'F1-macro': 'F1-macro_sd'})], axis=1)
    #     print(df_processed)
    #     df_processed.to_csv('test/'+model, sep='\t', encoding='utf-8')
    #
    #     # df1_mean.plot(label = label_encodings[model], y = y, ax=ax, markevery=1, marker='o')
    # # plt.show()
    # # plt.savefig('{}/{}.png'.format(args.output_dir, title))

def read_baseline_results_bfever(mapping, baseline_models, list_of_m):
    for baseline_model in baseline_models:
        # print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::33], content[1::33], content[2::33], content[26::33], content[27::33])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-labelwise: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]

        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[5]), float(x[3]), float(x[4])] for x in content]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_not', 'std_acc', 'std_s', 'std_not'])
        print(table)
        table.to_csv("labelwise/bfever/bfever_baseline_{}.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')


def read_baseline_results_fever(mapping, baseline_models, list_of_m):
    baseline_results = {}
    for baseline_model in baseline_models:
        # print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::34], content[1::34], content[2::34], content[27::34], content[28::34])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]

        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6])] for x in content]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_n', 'mean_c', 'std_acc', 'std_s', 'std_n', 'std_c'])
        print(table)
        table.to_csv("labelwise/fever_baseline_{}1.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')

def read_baseline_results_scifact(mapping, baseline_models, list_of_m):
    for baseline_model in baseline_models:
        # print(baseline_model)
        with open(mapping[baseline_model]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::33], content[1::33], content[2::33], content[26::33], content[27::33])]
        content = [
            x.replace("Accuracy: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace("[", "").replace("]", "").split() for x in content]

        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6])] for x in content]
        print(content)

        average_per_m = {}

        for m in list_of_m:
            results = []
            for instance in content:
                if instance[1] == m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m] = np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table = pd.DataFrame.from_dict(average_per_m, orient='index',
                                       columns=['mean_acc', 'mean_s', 'mean_n', 'mean_c', 'std_acc', 'std_s', 'std_n', 'std_c'])
        print(table)
        table.to_csv("labelwise/scifact/scifact_baseline_{}.csv".format(baseline_model), sep='\t', encoding='utf-8', float_format='%.3f')


def read_ft_results_scifact(mapping, model_bases, list_of_m):
    ft_results = {}
    for model_base in model_bases:
        # print(baseline_model)
        with open(mapping[model_base]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::13], # seed, # of shots, index of run
                                            content[2::13], content[3::13], content[5::13], content[6::13], # before tuning: acc on train, f1 on train, acc on dev, f1 on dev
                                            content[8::13], content[9::13], content[11::13], content[12::13])] # after tuning: acc on train, f1 on train, acc on dev, f1 on dev
        print(content)
        content = [
            x.replace("Acc: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace('Accuracy: ', "").split() for x in content]
        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]), float(x[7]), float(x[8]), float(x[9]), float(x[10])] for x in content]
        print('content', content)

        # average_per_m = []
        # list_of_m = [10, 20, 30, 40, 50, 100]
        # for m in list_of_m:
        #     sums=np.zeros(8)
        #     for instance in content:
        #         if instance[1]==m:
        #             for k in range(8):
        #                 sums[k]+= instance[k+3]
        #     average_per_m.append([m, sums/(len(content)/len(list_of_m))])
        # print('average over m', average_per_m)

        average_per_m = {}

        for m in list_of_m:
            results =[]
            for instance in content:
                if instance[1]==m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m]= np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table=pd.DataFrame.from_dict(average_per_m, orient='index',
                                     columns=['before_acc_train', 'before_f1_train', 'before_acc_dev', 'before_f1_dev',
                                              'after_acc_train', 'after_f1_train', 'after_acc_dev', 'after_f1_dev',
                                              'before_acc_train', 'before_f1_train', 'before_acc_dev', 'before_f1_dev',
                                              'after_acc_train', 'after_f1_train', 'after_acc_dev', 'after_f1_dev'
                                              ])
        print(table)
        table.to_csv("{}.csv".format(model_base), sep='\t', encoding='utf-8', float_format='%.3f')
        #
        ft_results[model_base] = average_per_m

    return ft_results

def read_hyper_results_scifact(mapping, model_bases):
    ft_results = {}
    for model_base in model_bases:
        # print(baseline_model)
        with open(mapping[model_base]) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [' '.join(x) for x in zip(content[0::13], # seed, # of shots, index of run
                                            content[2::13], content[3::13], content[5::13], content[6::13], # before tuning: acc on train, f1 on train, acc on dev, f1 on dev
                                            content[8::13], content[9::13], content[11::13], content[12::13])] # after tuning: acc on train, f1 on train, acc on dev, f1 on dev
        content = [
            x.replace("Acc: ", "").replace("F1-macro: ", "").replace("seed: ", "").replace("# of shots: ", "").replace(
                "index of run:", "").replace(";", "").replace('Accuracy: ', "").split() for x in content]
        content = [[int(x[0]), int(x[1]), int(x[2]), float(x[5]), float(x[6]), float(x[7]), float(x[8]), float(x[9]), float(x[10]), float(x[11]), float(x[12])] for x in content]
        print('content', content)

        # average_per_m = []
        # list_of_m = [10, 20, 30, 40, 50, 100]
        # for m in list_of_m:
        #     sums=np.zeros(8)
        #     for instance in content:
        #         if instance[1]==m:
        #             for k in range(8):
        #                 sums[k]+= instance[k+3]
        #     average_per_m.append([m, sums/(len(content)/len(list_of_m))])
        # print('average over m', average_per_m)

        average_per_m = {}

        list_of_m = [10, 20, 30, 40, 50, 100]
        for m in list_of_m:
            results =[]
            for instance in content:
                if instance[1]==m:
                    results.append(instance[3:])
            results = np.array(results)
            # print(m, results)
            # print(m, np.mean(results, axis=0))
            # print(m, np.std(results, axis=0))

            average_per_m[m]= np.concatenate((np.mean(results, axis=0), np.std(results, axis=0)), axis=None)
        print('average over m')
        print(average_per_m)
        table=pd.DataFrame.from_dict(average_per_m, orient='index',
                                     columns=['before_acc_train', 'before_f1_train', 'before_acc_dev', 'before_f1_dev',
                                              'after_acc_train', 'after_f1_train', 'after_acc_dev', 'after_f1_dev',
                                              'before_acc_train', 'before_f1_train', 'before_acc_dev', 'before_f1_dev',
                                              'after_acc_train', 'after_f1_train', 'after_acc_dev', 'after_f1_dev'
                                              ])
        print(table)
        table.to_csv("{}.csv".format(model_base), sep='\t', encoding='utf-8', float_format='%.3f')
        #
        ft_results[model_base] = average_per_m

    return ft_results

if __name__ == '__main__':

    # # # bfever
    baseline_models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
    mapping = {'bert-base-uncased':'labelwise/bfever/bfever_baseline_bert_base_o.txt',
               'bert-large-uncased': 'labelwise/bfever/bfever_baseline_bert_large_o.txt',
               'roberta-base':'labelwise/bfever/bfever_baseline_roberta_base_o.txt',
               'roberta-large':'labelwise/bfever/bfever_baseline_roberta_large_o.txt'}
    list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]

    baseline_results_fever = read_baseline_results_bfever(mapping, baseline_models, list_of_m)

    # # # # bfever
    # models = ["bert-base-uncased", "bert-large-uncased", "bert-base-nli-mean-tokens"]
    # mapping = {'bert-base-uncased':'labelwise/bfever/bfever_svec_all.txt',
    #            'bert-large-uncased': 'labelwise/bfever/bfever_svec_all.txt',
    #            'bert-base-nli-mean-tokens':'labelwise/bfever/bfever_svec_all.txt'}
    # list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    #
    # svec_results_bfever = read_svec_results_bfever(mapping, models, list_of_m)
    #
    # # # # # scifact
    # models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
    # mapping = {'bert-base-uncased':'labelwise/scifact/scifact_baseline_bert_base.txt',
    #            'bert-large-uncased': 'labelwise/scifact/scifact_baseline_bert_large.txt',
    #            'roberta-base':'labelwise/scifact/scifact_baseline_roberta_base.txt',
    #            'roberta-large':'labelwise/scifact/scifact_baseline_roberta_large.txt'}
    # list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    #
    # baseline_results_scifact = read_baseline_results_scifact(mapping, models, list_of_m)

    # # # # # scifact
    # models = ["bert-base-uncased", "bert-large-uncased", "bert-base-nli-mean-tokens"]
    # mapping = {'bert-base-uncased':'labelwise/scifact/scifact_svec_all.txt',
    #            'bert-large-uncased': 'labelwise/scifact/scifact_svec_all.txt',
    #            'bert-base-nli-mean-tokens':'labelwise/scifact/scifact_svec_all.txt'}
    # list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    #
    # svec_results_scifact = read_svec_results_scifact(mapping, models, list_of_m)

    # # # # fever
    # baseline_models = ["bert-large-uncased", "roberta-large"]
    # mapping = {'bert-base-uncased':'labelwise/fever_baseline_bert_base_o.txt',
    #            'bert-large-uncased': 'labelwise/3fever/1/fever_baseline_bert_large_o.txt',
    #            'roberta-base':'labelwise/fever_baseline_roberta_base_o.txt',
    #            'roberta-large':'labelwise/3fever/1/fever_baseline_roberta_large_o.txt'}
    # list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    #
    # baseline_results_fever = read_baseline_results_fever(mapping, baseline_models, list_of_m)

    # baseline_models = ["bert-base-uncased", 'bert-large-uncased', 'bert-base-nli-mean-tokens']
    # mapping = {'bert-base-uncased':'fever_svec_o1.txt', 'bert-large-uncased':'fever_svec_o1.txt', 'bert-base-nli-mean-tokens':'fever_svec_o1.txt'}
    # list_of_m = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
    # svec_results_fever = read_svec_results_fever(mapping, baseline_models, list_of_m)
    # noise = ["Counter({'s': 3331, 'not': 3331})", "Counter({'s': 3323, 'not': 3323})", "Counter({'s': 3283, 'not': 3283})"]
    #
    #
    # with open(args.svector_results_fever) as f:
    #     content = f.readlines()
    # svector_results = [x.strip() for x in content]
    # svector_results = [x for x in svector_results if x not in noise]
    #
    # #binary
    # svector_results_fever = svector_results[svector_results.index('fever_fever')+1:]
    # svector_results_fever = [ ' '.join(x) for x in zip(svector_results_fever[0::2], svector_results_fever[1::2])]
    # svector_results_fever = [x.replace("Accuracy: ", "").replace("F1-macro: ", "").split() for x in svector_results_fever]
    # svector_results_fever = [[int(x[0]), x[1], int(x[2]), float(x[3]), float(x[4])] for x in svector_results_fever]
    #
    # # svector_results_fever = svector_results[svector_results.index('fever_fever')+1:]
    # # svector_results_fever = [ ' '.join(x) for x in zip(svector_results_fever[0::3], svector_results_fever[1::3], svector_results_fever[2::3] )]
    # # svector_results_fever = [x.replace("Accuracy: ", "").replace("F1-macro: ", "").split() for x in svector_results_fever]
    # # svector_results_fever = [[int(x[0]), x[1], int(x[2]), float(x[3]), float(x[4])] for x in svector_results_fever]
    #
    # parse_and_plot(baseline_results_fever, svector_results_fever, title='Comparison of Few-Shot Performance (Accuracy) on FEVER Dataset', y='Accuracy')
    # parse_and_plot(baseline_results_fever, svector_results_fever, title='Comparison of Few-Shot Performance (F1-macro) on FEVER Dataset', y='F1-macro')


    #
    # ft_models = ["biobert"]
    # mapping = {'bert-base-uncased':'reserve/scifact_bert_base_uncased.txt',
    #            'bert-nli':'reserve/scifact_bert_nli.txt',
    #            'biobert':'reserve/scifact_biobert.txt',
    #            'roberta-base':'scifact_rb_o.txt',
    #            "fever_bert-base-uncased": 'fever_o.txt',
    #            "scifact_hyper_bert-base-uncased": 'scifact_hyper_o.txt'}
    #
    # list_of_m = [30, 40, 50, 100]
    #
    # noise = ['2 2 2', '4 4 4', '6 6 6', '8 8 8', '10 10 10', '20 20 20', '30 30 30', '40 40 40', '50 50 50', '100 100 100', 'scifact --> scifact:']
    # #scifact
    # ft_results_scifact = read_ft_results_scifact(mapping, ft_models, list_of_m)
    # # ft_results_scifact = read_hyper_results_scifact(mapping, ft_models)

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
    #
    #
