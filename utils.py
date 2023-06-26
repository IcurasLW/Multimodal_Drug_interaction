import numpy as np
import torch
import pandas as pd
import os
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import csv
import math



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Drug_Features:
    '''
    A class used to stored the feature embedding data
    This drug_features is used as a look-up table.
    Given two drugs ID, it returns the concated embedding vectors
    '''
    def __init__(self, smiles_features, enzyme_features, target_features, dataset_path):
        self.smiles_features = smiles_features
        self.enzyme_features = enzyme_features
        self.target_features = target_features
        self.drugs_list = pd.read_csv(dataset_path)['drugs_id'].to_list()


    def concat_drugs(self, feature_type:str, drug_A:str, drug_B:str):
        '''
        Input: two drugs ID
        return: the concat embedding
        '''
        
        if feature_type == 'smiles':
            feature = self.smiles_features
        elif feature_type == 'enzymes':
            feature = self.enzyme_features
        elif feature_type == 'targets':
            feature = self.target_features

        if drug_A in self.drugs_list and drug_B in self.drugs_list:
            drug_A_index = self.drugs_list.index(drug_A)
            drug_B_index = self.drugs_list.index(drug_B)

        # concat two drugs features
        return torch.concat(feature[drug_A_index], feature[drug_B_index], dim=1)



class EarlyStopping:
    def __init__(self, delta=0.005, patience=5, path='checkpoint.pt'):
        self.patience = patience
        self.path = path
        self.epoch = None
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    
    
    def __call__(self, val_loss, epoch, model):
        self.epoch = epoch
        if self.best_score is None:
            self.best_score = val_loss
            
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                save_path = f'{self.epoch}_{self.path}'
                torch.save(model.state_dict(), save_path)
                
        else:
            self.best_score = val_loss
            self.counter = 0




def data_split(df, val_rate = 0.1):
    ddis_type = df['interaction'].unique()
    df_val = pd.DataFrame(columns=df.columns)
    for ddi in ddis_type:
        mid_df = df.loc[df['interaction']==ddi].sample(frac=val_rate,axis=0)
        df_val = pd.concat([df_val, mid_df],axis=0)
    
    df_val.reset_index(inplace=True)
    df_train = df.drop(index=df_val['index'])

    '''
    test
    '''
    sorted_ddis_train = df_train['interaction'].value_counts().to_dict()

    # Shuffle dataframe
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    df_val = df_val.loc[:, ['id1', 'name1', 'id2', 'name2', 'interaction']]
    df_train = df_train.loc[:, ['id1', 'name1', 'id2', 'name2', 'interaction']]
    return df_train, df_val



def roc_aupr_score(y_true, y_score, average="macro"):
    
    '''
    y_score: the probality of prediction in shape: [num_sample, num_class]
    y_true: the model predcition in shape: [num_sample, num_class], it's one-hot reprensentation of class
    '''
    y_one_hot = label_binarize(y=y_true, classes=[i for i in range(213)])

    def _binary_roc_aupr_score(y_one_hot, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_one_hot, y_score)
        return auc(recall, precision) # Liangwei: I delete a re-order

    def _average_binary_score(binary_metric, y_one_hot, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_one_hot, y_score)
        if average == "micro":
            y_one_hot = y_one_hot.ravel()
            y_score = y_score.ravel()
        if y_one_hot.ndim == 1:
            y_one_hot = y_one_hot.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_one_hot.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_one_hot, y_score, average)



def evaluate(y_pred, y_true, mode):
    '''
    y_pred: Prediction probability
    y_true: index of classes from 0 to 67, Pandas Series 
    '''
    y_pred_input = y_pred.clone().to('cpu').detach().numpy()
    y_true = y_true.to_numpy()
    pred_max_indices = np.argmax(y_pred_input, axis=1)
    acc = accuracy_score(y_pred=pred_max_indices, y_true=y_true)
    f1_macro = f1_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    f1_micro = f1_score(y_pred=pred_max_indices, y_true=y_true, average='micro', zero_division=1)
    f1_weighted = f1_score(y_pred=pred_max_indices, y_true=y_true, average='weighted', zero_division=1)
    precision_macro = precision_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    precision_micro = precision_score(y_pred=pred_max_indices, y_true=y_true, average='micro', zero_division=1)
    precision_weighted = precision_score(y_pred=pred_max_indices, y_true=y_true, average='weighted', zero_division=1)
    recall_macro = recall_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    recall_micro = recall_score(y_pred=pred_max_indices, y_true=y_true, average='micro', zero_division=1)
    recall_weighted = recall_score(y_pred=pred_max_indices, y_true=y_true, average='weighted', zero_division=1)

    if mode == 'train':
        output = {
                'acc':acc,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'f1_weighted': f1_weighted,
                'PR_macro': precision_macro,
                'PR_micro': precision_micro,
                'PR_weighted': precision_weighted,
                'RE_macro': recall_macro,
                'RE_micro': recall_micro,
                'RE_weighted': recall_weighted
                }

    elif mode == 'val':
        # 2023/06/01: prediction 是不是预测的类的概率看一下
        
        auc_score_micro = roc_auc_score(y_score=y_pred_input, y_true=y_true, multi_class='ovr', average='micro')
        auc_score_macro = roc_auc_score(y_score=y_pred_input, y_true=y_true, multi_class='ovr', average='macro')
        auc_score_weighted = roc_auc_score(y_score=y_pred_input, y_true=y_true, multi_class='ovr', average='weighted')

        aupr_score_micro = roc_aupr_score(y_true, y_pred_input, average='micro')
        aupr_score_macro = roc_aupr_score(y_true, y_pred_input, average='macro')

        output = {
                'acc':acc,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'f1_weighted': f1_weighted,
                'PR_macro': precision_macro,
                'PR_micro': precision_micro,
                'PR_weighted': precision_weighted,
                'RE_macro': recall_macro,
                'RE_micro': recall_micro,
                'RE_weighted': recall_weighted,
                'AUC_micro': auc_score_micro,
                'AUC_macro': auc_score_macro,
                'AUC_weighted': auc_score_weighted,
                'AUPR_micro': aupr_score_micro,
                'AUPR_macro': aupr_score_macro
                }
    return output


def get_newest_file(directory):
    files = os.listdir(directory)
    if not files:
        return None

    # Sort the files based on modification time in descending order
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    newest_file = files[0]
    return newest_file



# used to preprocess data
def sample_data(data_path, output_filename, sample_rate):
    df = pd.read_csv(data_path)
    ddis_type = df['interaction'].unique()
    df_ddis_sample = pd.DataFrame(columns=df.columns)

    for ddi in ddis_type:
        mid_df = df.loc[df['interaction']==ddi].sample(frac=sample_rate,axis=0)
        df_ddis_sample = pd.concat([df_ddis_sample, mid_df],axis=0)

    df_ddis_sample.reset_index(drop=False, inplace=True)
    df_filtered = df.drop(index=df_ddis_sample['Unnamed: 0'])



def save_results(fold_num, metrics:dict, mode:str):
    '''
    Each row is one epoch
    '''
    output_filename = f'fold_{fold_num}_{mode}_results.csv'

    # save train results
    try:
        # if the results exist
        with open (output_filename, 'r') as log:
            pass
        
        print('file exists')
        with open(output_filename, 'a+', newline='') as log:
            writer = csv.DictWriter(log, fieldnames=metrics.keys())
            writer.writerow(metrics)

    except:
        print('file not exists')
        with open(output_filename, 'w', newline='') as log:
            writer = csv.DictWriter(log, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
            
            
            

def make_features_matrix(feature_data_path):
    '''
    Input: Features type, features data
    Convert the 
    Output: Enzymes, target embeddings 
    '''
    df = pd.read_csv(feature_data_path)
    df = df.dropna(axis=0) 
    fn_1 = lambda row: row.split('|')
    df['Targets'] = df['Targets'].apply(fn_1)
    df['Enzymes'] = df['Enzymes'].apply(fn_1)
    
    # Find the unique elements of targets and enzymes
    targets_set = set()
    enzymes_set = set()
    for each in df['Targets'].to_list():
        targets_set = targets_set | set(each)
        
    for each in df['Enzymes'].to_list():
        enzymes_set = enzymes_set | set(each)

    targets_set = list(targets_set)
    enzymes_set = list(enzymes_set)

    # targets features matrix 
    targets_drugs_matrix = []
    for each in df['Targets'].to_list():
        cur_drug = [0 for _ in range(len(targets_set))]
        for i in each:
            index = targets_set.index(i)
            cur_drug[index] = 1
        targets_drugs_matrix.append(cur_drug)

    # enzymes features matrix 
    enzymes_drugs_matrix = []
    for each in df['Enzymes'].to_list():
        cur_drug = [0 for _ in range(len(enzymes_set))]
        for i in each:
            index = enzymes_set.index(i)
            cur_drug[index] = 1
        enzymes_drugs_matrix.append(cur_drug)

    return targets_drugs_matrix, enzymes_drugs_matrix



def similarity_matrix(enzymes_matrix, targets_matrix):
    # This Jaccard Function came from DDIMDL
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator
    
    enzymes_sim_matrix = torch.tensor(Jaccard(enzymes_matrix))
    targets_sim_matrix = torch.tensor(Jaccard(targets_matrix))

    return enzymes_sim_matrix, targets_sim_matrix


def make_label(df):
    sorted_ddis = df['interaction'].value_counts().to_dict()
    dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
    df['labels_num'] = df['interaction'].map(dict_ddis)
    labels = df['labels_num']
    return labels


def split_train_test(df, test_size):
    interactions = df['interaction'].value_counts()
    interactions = interactions.loc[interactions >= 2]

    train_idx = []
    for each in interactions.keys():
        train_num = math.floor(interactions[each] * (1-test_size))
        train_frac = train_num / interactions[each]
        df_each = df.loc[df['interaction'] == each]
        df_each_train = df_each.sample(frac=train_frac)
        train_idx.extend(list(df_each_train.index))
    
    train_df = df.loc[df.index.isin(train_idx)].reset_index(drop=True)
    test_df = df.loc[~df.index.isin(train_idx)].reset_index(drop=True)
    
    return train_df, test_df