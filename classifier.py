import torch
import torch.nn as nn
from tqdm import trange
from losses import *
from utils import *


class Classifier_NN(nn.Module):
    def __init__(
                self,
                drug_features:object,
                num_classes,
                dropout=0.5,
                ):
        super().__init__()
        hidden_size_1 = 1024
        hidden_size_2 = 512
        hidden_size_3 = 256
        self.drug_features = drug_features
        

        if drug_features.smiles_features != None:
            input_size_1 = drug_features.smiles_features.shape[1]*2
            
            self.classifier_block_1 = nn.Sequential(
            nn.Linear(in_features=input_size_1, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_3)
            )


        if drug_features.enzyme_features != None:
            input_size_2 = drug_features.enzyme_features.shape[1]*2
            
            self.classifier_block_2 = nn.Sequential(
            nn.Linear(in_features=input_size_2, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_3)
            )
        
        
        if drug_features.target_features != None:
            input_size_3 = drug_features.target_features.shape[1]*2
            
            self.classifier_block_3 = nn.Sequential(
            nn.Linear(in_features=input_size_3, out_features=hidden_size_1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_1),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_2),
            nn.Dropout(p=dropout, inplace=True),

            nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size_3),
            )
        
        
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size_3, out_features=num_classes),
            nn.Softmax(dim=1)
            )
        
    def forward(self, train_X):
        '''
        Input drug batch to generate the smiles, enzyme, target vector pairs
        The drug pairs used to generate smiles and enzymes and target of the 
        '''
        
        drug_pairs = [(each['id1'], each['id2']) for _, each in train_X.iterrows()]
        smiles_input, enzyme_input, target_input  = self.concat_vector(drug_pairs)
        if smiles_input != None:
            smiles_output = self.classifier_block_1(smiles_input)
        else:
            smiles_output = 0
        
        
        if enzyme_input != None:
            enzyme_output = self.classifier_block_2(enzyme_input)
        else:
            enzyme_output = 0
        
        
        if target_input != None:
            target_output = self.classifier_block_3(target_input)
        else:
            target_output = 0
            
        output = nn.functional.normalize(enzyme_output + smiles_output + target_output, dim=1)
        output = self.final_layer(output)
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


            if self.drug_features.smiles_features != None:
                drug_1_vector_smiles = torch.unsqueeze(self.drug_features.smiles_features[drug_1_index], dim=0)
                drug_2_vector_smiles = torch.unsqueeze(self.drug_features.smiles_features[drug_2_index], dim=0)
                cat_smiles = torch.concat((drug_1_vector_smiles, drug_2_vector_smiles), dim=1)
                output_smiles = torch.cat((output_smiles, cat_smiles), dim=0)
                output_smiles = nn.functional.normalize(output_smiles)
            else:
                output_smiles = None
            
            
            if self.drug_features.target_features != None:
                drug_1_vector_enzymes = torch.unsqueeze(self.drug_features.enzyme_features[drug_1_index], dim=0)
                drug_2_vector_enzymes = torch.unsqueeze(self.drug_features.enzyme_features[drug_2_index], dim=0)
                cat_enzyme = torch.concat((drug_1_vector_enzymes, drug_2_vector_enzymes), dim=1)
                output_target = torch.cat((output_target, cat_target), dim=0)
                output_target = nn.functional.normalize(output_target)
            else:
                output_target = None
            
            
            if self.drug_features.enzyme_features != None:
                drug_1_vector_target = torch.unsqueeze(self.drug_features.target_features[drug_1_index], dim=0)
                drug_2_vector_target = torch.unsqueeze(self.drug_features.target_features[drug_2_index], dim=0)
                cat_target = torch.concat((drug_1_vector_target, drug_2_vector_target), dim=1)
                output_enzyme = torch.cat((output_enzyme, cat_enzyme), dim=0)
                output_enzyme = nn.functional.normalize(output_enzyme)
            else:
                output_enzyme = None
        
        
        return output_smiles, output_enzyme, output_target

