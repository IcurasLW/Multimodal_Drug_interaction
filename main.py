import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
from SMILES_Embeddings.SMILES_Embedding import running_SMILES_embeddings, smiles_embedding
import SMILES_Embeddings.molbart.util as util
import os
import math
from torch import optim
from sklearn.model_selection import StratifiedKFold
from tqdm import trange
from losses import *
from utils import *
from classifier import *
import csv




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'





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
        metrics = evaluate(y_pred, test_y, mode='test')
        metrics['loss'] = loss.item()
    return metrics



def cross_validate(args, drug_features, loss_function, df_ddis, num_classes ,K=5, epochs=100):
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    # kf = KFold(n_splits=K, shuffle=True, random_state=0)
    # df_train,df_val = data_split(df=df_ddis, val_rate=0.1)
    Y = make_label(df_ddis)
    for k, (train_index, test_index) in enumerate(skf.split(df_ddis, Y)):
        print(f'----------------------------Fold {k}----------------------------')
        model = Classifier_NN(
                            drug_features=drug_features,
                            num_classes=num_classes,
                            dropout=args.dropout
                            )
        
        # Resume training.
        directory_path = './temp_models/'
        try:
            newest_model = get_newest_file(directory_path)
            checkpoint = torch.load(directory_path+newest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_epoch = checkpoint['epoch']
            
        except:
            resume_epoch = 0


        lr = args.lr
        model.to(DEVICE)
        train_X_k, test_X_k = df_ddis.loc[train_index], df_ddis.loc[test_index]
        train_y_k, test_y_k = Y[train_index], Y[test_index]
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ConstantLR(optimizer, total_iters=5, factor=0.90) 
        # train_metrics_fold_k = None
        # test_metrics_fold_k = None
        # early_stop = EarlyStopping(patience=5)
        
        
        
        for e in range(resume_epoch, epochs+1):
            print(f'----------------------------Epoch {e}----------------------------')
            
            train_metrics = train(
                                model=model,
                                train_X=train_X_k,
                                train_y=train_y_k,
                                loss_fn=loss_function,
                                optimizer=optimizer,
                                batch_size=args.batch_size
                                )
            save_results(fold_num=k, metrics=train_metrics, mode='train')



            test_metrics = test(
                                model,
                                test_X=test_X_k,
                                test_y=test_y_k,
                                loss_fn=loss_function
                                )
            # scheduler.step(test_metrics['loss'])
            save_results(fold_num=k, metrics=test_metrics, mode='test')
            # early_stop(test_metrics['loss'], epoch=e, model=model) # pass in a loss to earlystopping
            
            
            
            if e % 5 == 0:
                path_to_save = f'./temp_models/{k}_fold_model_at_epoch_{e}.pth'
                torch.save({
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path_to_save)

            # if early_stop.early_stop:
            #     print('Save model and stop')
            #     break # Break this KFoldtraining
        break
            # Add up the results of each epoch
            # if train_metrics_fold_k==None:
            #     train_metrics_fold_k = train_metrics
            # else:
            #     for key in train_metrics:
            #         train_metrics_fold_k[key] = train_metrics_fold_k[key] + train_metrics[key]

            # if test_metrics_fold_k==None:
            #     test_metrics_fold_k = test_metrics
            # else:
            #     for key in test_metrics:
            #         test_metrics_fold_k[key] = test_metrics_fold_k[key] + test_metrics[key]


def train_no_CV(args, drug_features, loss_function, df_ddis, num_classes ,epochs=100):
    sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
    dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
    df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
    
    train_df, test_df = split_train_test(df_ddis, test_size=0.2)
    model = Classifier_NN(
                            drug_features=drug_features,
                            num_classes=num_classes,
                            dropout=args.dropout
                            )
    
    # Resume training.
    directory_path = './temp_models/'
    try:
        newest_model = get_newest_file(directory_path)
        checkpoint = torch.load(directory_path+newest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        resume_epoch = checkpoint['epoch']
        
    except:
        resume_epoch = 0


    lr = args.lr
    model.to(DEVICE)
    train_X_k, test_X_k = train_df, test_df
    train_y_k, test_y_k = train_df['labels_num'], test_df['labels_num']
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ConstantLR(optimizer, total_iters=5, factor=0.90) 
    # train_metrics_fold_k = None
    # test_metrics_fold_k = None
    # early_stop = EarlyStopping(patience=5)
    
    
    
    for e in range(resume_epoch, epochs+1):
        print(f'----------------------------Epoch {e}----------------------------')
        
        train_metrics = train(
                            model=model,
                            train_X=train_X_k,
                            train_y=train_y_k,
                            loss_fn=loss_function,
                            optimizer=optimizer,
                            batch_size=args.batch_size
                            )
        save_results(fold_num=0, metrics=train_metrics, mode='train')


        test_metrics = test(
                            model,
                            test_X=test_X_k,
                            test_y=test_y_k,
                            loss_fn=loss_function
                            )
        # scheduler.step(test_metrics['loss'])
        save_results(fold_num=0, metrics=test_metrics, mode='test')
        # early_stop(test_metrics['loss'], epoch=e, model=model) # pass in a loss to earlystopping
        
        
        if e % 5 == 0:
            path_to_save = f'./temp_models/{0}_fold_{args.loss_function}_at_epoch_{e}.pth'
            torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, path_to_save)



def main(args):
    torch.manual_seed(0)
    ddis_data_path = args.data_path + "ddi.csv"
    features_data_path = args.data_path + "features.csv"
    df_ddis = pd.read_csv(ddis_data_path)
    
    # get all the featrues embeddings
    # smiles_features = running_SMILES_embeddings(features_data_path) # If you need to run the embedding from Encoder
    smiles_features = torch.load(args.data_path + 'SMILES_features.pt') # If you have the embeding already
    
    
    # Decide number of class
    if '65' in args.data_path:
        print('Num_classes: 65')
        num_cls = 65
    elif '130' in args.data_path:
        print('Num_classes: 130')
        num_cls = 130
    elif '213' in args.data_path:
        print('Num_classes: 213')
        num_cls = 213
    
    
    
    # Decide feature path
    if 'smiles' in args.feature_path:
        print('Using SMILES')
        smiles_features = torch.load(args.data_path + 'SMILES_features.pt')
    else:
        smiles_features = None
        
    if 'target' in args.feature_path:
        print('Using Target')
        targets_matrix, enzymes_matrix = make_features_matrix(features_data_path)
        targets_sim_matrix, enzymes_sim_matrix = similarity_matrix(targets_matrix, enzymes_matrix)
    
    
    # Decide loss function
    if args.loss_function == 'focalloss':
        loss_fn = FocalLoss()
    elif args.loss_function == 'crossentropy':
        loss_fn = CrossEntropy()
    elif args.loss_function == 'ratioloss':
        loss_fn = RatioLoss()
    else:
        raise Exception('Allowable loss: focalloss, crossentropy, ratioloss')
    
    
    
    # Emebedding the 
    targets_sim_matrix = targets_sim_matrix.to(DEVICE).to(torch.float32)
    enzymes_sim_matrix = enzymes_sim_matrix.to(DEVICE).to(torch.float32)
    smiles_features = smiles_features.to(DEVICE).to(torch.float32)
    drug_features = Drug_Features(
                                    smiles_features=smiles_features,
                                    target_features=targets_sim_matrix,
                                    enzyme_features=enzymes_sim_matrix,
                                    dataset_path=features_data_path
                                    )


    if num_cls != 213:
        cross_validate(
                        args=args,
                        drug_features=drug_features,
                        loss_function=loss_fn,
                        df_ddis=df_ddis,
                        K=5,
                        epochs=args.epochs,
                        num_classes=num_cls
                        )
    else:
        train_no_CV(
                    args=args,
                    drug_features=drug_features,
                    loss_function=loss_fn,
                    df_ddis=df_ddis,
                    epochs=args.epochs,
                    num_classes=num_cls
                    )



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./dataset/ddi_65_30%/')
    parser.add_argument("--feature_path", type=list, default=['smiles', 'target', 'enzyme'])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--loss_function", type=str, default='focal_loss')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cross-validate", type=str, default='stratified')
    
    
    args = parser.parse_args()
    main(args)