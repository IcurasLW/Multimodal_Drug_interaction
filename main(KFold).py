import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import argparse
from SMILES_Embeddings.SMILES_Embedding import running_SMILES_embeddings, smiles_embedding
import SMILES_Embeddings.molbart.util as util
import os
from sklearn.model_selection import KFold
import math
from torch import optim
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import trange



from utils import evaluate
from utils import data_split
from utils import get_newest_file
import csv

DEFAULT_BATCH_SIZE = 20
DEFAULT_NUM_BEAMS = 10
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


class FocalLoss(nn.Module):
    def __init__(self, alphas:list=None, gamma=2, reduction:str='mean'):
        super().__init__()
        self.alphas = alphas # The inverse of frequency of class
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, input, target):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my input = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        loss = torch.empty((0,)).to(DEVICE)
        target = list(target)
        if self.alphas==None:
            for each in range(len(target)):
                index_label = target[each]
                Pt = input[each, index_label]
                focal_loss= -((1-Pt)**self.gamma)*torch.log2(Pt)
                loss = torch.cat((loss, focal_loss.unsqueeze(0)), dim=0)
        else:
            self.alphas = torch.tensor(self.alphas) 
            for each in range(len(target)-1):
                index_label = target[each]
                Pt = input[each, index_label]
                alphas_i = self.alphas[index_label]
                focal_loss= -alphas_i*((1-Pt)**self.gamma)*torch.log2(Pt)
                loss = torch.cat((loss, focal_loss.unsqueeze(0)), dim=0)
                
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class Classifier_NN(nn.Module):
    def __init__(
                self,
                drug_features:object,
                num_classes,
                dropout=0.5,
                ):
        super().__init__()
        self.drug_features = drug_features
        input_size_1 = drug_features.smiles_features.shape[1]*2
        input_size_2 = drug_features.enzyme_features.shape[1]*2
        input_size_3 = drug_features.target_features.shape[1]*2
        hidden_size_1 = 512
        hidden_size_2 = 256
        

        self.classifier_block_1 = nn.Sequential(
            nn.Linear(in_features=input_size_1, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            # nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(hidden_size_1),
            # nn.Dropout(p=dropout, inplace=False),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout,inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=num_classes),
            nn.Softmax(dim=1)
        )

        self.classifier_block_2 = nn.Sequential(
            nn.Linear(in_features=input_size_2, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            # nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(hidden_size_1),
            # nn.Dropout(p=dropout, inplace=False),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout,inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=num_classes),
            nn.Softmax(dim=1)
        )

        self.classifier_block_3 = nn.Sequential(
            nn.Linear(in_features=input_size_3, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            # nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(hidden_size_1),
            # nn.Dropout(p=dropout, inplace=False),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout,inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, train_X):
        '''
        Input drug batch to generate the smiles, enzyme, target vector pairs
        The drug pairs used to generate smiles and enzymes and target of the 
        '''
        
        drug_pairs = [(each['id1'], each['id2']) for _, each in train_X.iterrows()]

        smiles_input, enzyme_input, target_input  = self.concat_vector(drug_pairs)
        smiles_output = self.classifier_block_1(smiles_input)
        enzyme_output = self.classifier_block_2(enzyme_input)
        target_output = self.classifier_block_3(target_input)
        output = nn.functional.normalize(enzyme_output + target_output + smiles_output, dim=1)
        output = nn.functional.softmax(output,dim=1)
        return output


    def concat_vector(self, drug_pairs):
        '''
        input: features 2-d array
        output: [[concat smiles features for drug pair1], 
                [concat smiles features for drug pair2],.......]
        
        drug_pairs = (id1, id2)
        smiles_features is a matrix that 
        '''
        output_smiles = torch.empty((0,)).to(device=DEVICE)
        output_enzyme = torch.empty((0,)).to(device=DEVICE)
        output_target = torch.empty((0,)).to(device=DEVICE)
        for each in drug_pairs:
            drug_1 = each[0]
            drug_2 = each[1]
            drug_1_index = self.drug_features.drugs_list.index(drug_1)
            drug_2_index = self.drug_features.drugs_list.index(drug_2)

            drug_1_vector_smiles = torch.unsqueeze(self.drug_features.smiles_features[drug_1_index], dim=0)
            drug_2_vector_smiles = torch.unsqueeze(self.drug_features.smiles_features[drug_2_index], dim=0)
            cat_smiles = torch.concat((drug_1_vector_smiles, drug_2_vector_smiles), dim=1)
            
            drug_1_vector_enzymes = torch.unsqueeze(self.drug_features.enzyme_features[drug_1_index], dim=0)
            drug_2_vector_enzymes = torch.unsqueeze(self.drug_features.enzyme_features[drug_2_index], dim=0)
            cat_enzyme = torch.concat((drug_1_vector_enzymes, drug_2_vector_enzymes), dim=1)

            drug_1_vector_target = torch.unsqueeze(self.drug_features.target_features[drug_1_index], dim=0)
            drug_2_vector_target = torch.unsqueeze(self.drug_features.target_features[drug_2_index], dim=0)
            cat_target = torch.concat((drug_1_vector_target, drug_2_vector_target), dim=1)

            # You need to concat the tensors
            output_smiles = torch.cat((output_smiles, cat_smiles), dim=0)
            output_enzyme = torch.cat((output_enzyme, cat_enzyme), dim=0)
            output_target = torch.cat((output_target, cat_target), dim=0)

        output_smiles = nn.functional.normalize(output_smiles)
        output_enzyme = nn.functional.normalize(output_enzyme)
        output_target = nn.functional.normalize(output_target)
        return output_smiles, output_enzyme, output_target



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



def get_labels_frequency(df):
    '''
    Generate the frequency of labels for the alpha factor in focal loss
    '''
    total_count = len(df)
    unique, counts = np.unique(df['interaction'], return_counts=True)
    dict_ddis_count = {}

    for ddi, count in zip(unique, counts):
        dict_ddis_count[ddi] = count
    sorted_dict_ddi_count = dict(sorted(dict_ddis_count.items(), key=lambda x: x[1], reverse=True))

    alphas = [1/np.float32(each/total_count) for each in sorted_dict_ddi_count.values()]
    return alphas


def train(model, train_X, train_y, optimizer, loss_fn, batch_size=512):
    # 2023/05/24 note: find some evaluation metrics to evaluate the model.
    model.train()
    batch_nums = math.ceil(len(train_X)/batch_size)
    metrics = None
    
    # optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.95)

    for i in trange(1, batch_nums+1):
        # print(f'------------------Training Batch {i}/{batch_nums} started------------------')
        if i * batch_size < len(train_X):
            start_index = i * batch_size
            end_index = (i+1) * batch_size
        else:
            start_index = (i-1) * batch_size
            end_index = len(train_X) - 1

        optimizer.zero_grad()
        y_pred = model(train_X[start_index:end_index])
        y_true_batch = torch.tensor(list(train_y[start_index:end_index])).to(DEVICE)
        loss = loss_fn(y_pred, y_true_batch)
        loss.backward()
        optimizer.step()

        # take average of each epochs
        if metrics==None:
            metrics = evaluate(y_pred, train_y[start_index:end_index], mode='train')
            metrics['loss'] = loss.item()
            
        else:
            mid_metrics = evaluate(y_pred, train_y[start_index:end_index], mode='train')
            mid_metrics['loss'] = loss.item()
            
            for key in mid_metrics:
                metrics[key] += mid_metrics[key]
    
    for key in metrics:
        metrics[key] = metrics[key]/batch_nums
        
    return metrics


def test(model, test_X, test_y, loss_fn):
    model.eval()
    with torch.inference_mode():
        y_pred = model(test_X)
        y_true = torch.tensor(list(test_y)).to(DEVICE)
        loss = loss_fn(y_pred, y_true)
        metrics = evaluate(y_pred, test_y, mode='val')
        metrics['loss'] = loss.item()
        

    return metrics


def validate(model, val_X, val_y, loss_fn):
    model.eval()
    with torch.inference_mode():
        y_pred = model(val_X)
        y_true = torch.tensor(list(val_y)).to(DEVICE)
        loss = loss_fn(y_pred, y_true)
        metrics = evaluate(y_pred, val_y, mode='val')
        metrics['loss'] = loss.item()
        

    return metrics

def cross_validate(drug_features, loss_function, df_ddis, K=5, epochs=100):

    def make_label(df):
        sorted_ddis = df['interaction'].value_counts().to_dict()
        dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
        df['labels_num'] = df_ddis['interaction'].map(dict_ddis)
        labels = df['labels_num']
        return labels
    
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    # skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    # df_train,df_val = data_split(df=df_ddis, val_rate=0.1)
    # train_y = make_label(df_train)
    # val_y = make_label(df_val)
    ddis_labels = make_label(df_ddis)
    for k, (train_index, test_index) in enumerate(kf.split(df_ddis, ddis_labels)):
        print(f'----------------------------Fold {k}----------------------------')
        model = Classifier_NN(
                            drug_features=drug_features,
                            num_classes=65,
                            dropout=0.65
                            )
        
        # Resume training.
        directory_path = './temp_models/'
        if os.path.exists(directory_path):
            newest_model = get_newest_file(directory_path)
            checkpoint = torch.load(directory_path+newest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_epoch = checkpoint['epoch']
            lr = 0.001 * 0.95 **(resume_epoch // 20)
        else:
            resume_epoch = 1
            lr= 0.001
        
        model.to(DEVICE)
        train_X_k, test_X_k = df_ddis.loc[train_index], df_ddis.loc[test_index]
        train_y_k, test_y_k = ddis_labels[train_index], ddis_labels[test_index]
        print(lr)
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, total_iters=20, factor=0.95)
        train_metrics_fold_k = None
        test_metrics_fold_k = None
        for e in range(resume_epoch, epochs+1):
            print(f'----------------------------Epoch {e}----------------------------')
            
            
            train_metrics = train(
                                model=model,
                                train_X=train_X_k,
                                train_y=train_y_k,
                                loss_fn=loss_function,
                                optimizer=optimizer
                                )
            save_results(fold_num=k, metrics=train_metrics, mode='train')
            
            test_metrics = test(
                                model,
                                test_X=test_X_k,
                                test_y=test_y_k,
                                loss_fn=loss_function
                                )
            
            
            scheduler.step(test_metrics['loss'])
            save_results(fold_num=k, metrics=test_metrics, mode='test')

            if e % 5 == 0:
                path_to_save = f'./temp_models/{k}_fold_model_at_epoch_{e}.pth'
                torch.save({
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path_to_save)


            # Add up the results of each epoch
            if train_metrics_fold_k==None:
                train_metrics_fold_k = train_metrics
            else:
                for key in train_metrics:
                    train_metrics_fold_k[key] = train_metrics_fold_k[key] + train_metrics[key]

            if test_metrics_fold_k==None:
                test_metrics_fold_k = test_metrics
            else:
                for key in test_metrics:
                    test_metrics_fold_k[key] = test_metrics_fold_k[key] + test_metrics[key]

        break

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


def main():
    torch.manual_seed(0)
    ddis_data_path = "ddis.csv"
    features_data_path = "features.csv"
    # get all the featrues embeddings
    # smiles_features = running_SMILES_embeddings(features_data_path) # Liangwei: If you need to run the embedding from Encoder
    smiles_features = torch.load('SMILES_features.pt') # If you have the embeding already
    torch.save(smiles_features, 'SMILES_features.pt')
    targets_matrix, enzymes_matrix = make_features_matrix(features_data_path)
    targets_sim_matrix, enzymes_sim_matrix = similarity_matrix(targets_matrix, enzymes_matrix)
    targets_sim_matrix = targets_sim_matrix.to(DEVICE).to(torch.float32)
    enzymes_sim_matrix = enzymes_sim_matrix.to(DEVICE).to(torch.float32)
    smiles_features = smiles_features.to(DEVICE).to(torch.float32)
    drug_features = Drug_Features(
                                    smiles_features=smiles_features,
                                    target_features=targets_sim_matrix,
                                    enzyme_features=enzymes_sim_matrix,
                                    dataset_path=features_data_path
                                    )
    # Get ddis counts the inverse frequency
    df_ddis = pd.read_csv(ddis_data_path)
    # alphas = get_labels_frequency(df_ddis)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss()
    cross_validate(
                    drug_features=drug_features,
                    loss_function=loss_fn,
                    df_ddis=df_ddis,
                    K=5,
                    epochs=400
                    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # working_dir = os.getcwd()
    # smile_path = working_dir + '/SMILES_Embeddings/SMILES_1.txt'
    # model_path = working_dir + "/SMILES_Embeddings/models/pre-trained/combined-large/step=1000000.ckpt"
    # vocab_path = working_dir + "/SMILES_Embeddings/bart_vocab.txt"

    # # Program level args
    # parser.add_argument("--reactants_path", type=str, default=smile_path)  # Each line is a input SMILES
    # parser.add_argument("--model_path", type=str, default=model_path)
    # parser.add_argument("--products_path", type=str, default="embedding.pickle")
    # parser.add_argument("--vocab_path", type=str, default=vocab_path)
    # parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # # Model args
    # parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    # parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)

    # args = parser.parse_args()
    # embed = running_SMILES_embeddings("./dataset/complete_drug_features.csv")
    # print(embed)



    '''
    Hypyerparameters: 
    Learning rate
    batch_size
    Dropout_rate
    focal_loss_gamma
    Number of network layer
    Number of Neurons
    Features_path

    I gonna ignore the affect the Features path
    '''


    main()