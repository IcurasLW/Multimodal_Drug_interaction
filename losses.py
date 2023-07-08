import numpy as np
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RatioLoss(nn.Module):
    def __init__(self, threshold, alphas:list=None, gamma=2, reduction:str='mean', ):
        super().__init__()
        self.alphas = alphas # The inverse of frequency of class
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold
        
    def forward(self, probs, target):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        
        Pt_index = target.squeeze()
        pred_p = torch.max(probs, dim=1)[0]
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        FL_loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        # ratio_loss = -torch.log2((ratio) * Pt)
        # ratio = (Pt/pred_p)
        # ratio_inve = 1 / ratio
        
        gate = torch.where(Pt_index > self.threshold, 1, 0).to(DEVICE)
        # Pt_dash = torch.where(gate == 0, 1, Pt)
        Pt_dash = Pt**gate
        loss = FL_loss + (-torch.log2(Pt_dash))
        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        
        
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction:str='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    
    def forward(self, probs, target):
        
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
    
        Pt_index = target.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)




class CrossEntropy(nn.Module):
    def __init__(self, alphas:list=None, gamma=2, reduction:str='mean'):
        super().__init__()
        self.alphas = alphas # The inverse of frequency of class
        self.gamma = gamma
        self.reduction = reduction
    
    
    def forward(self, probs, target):
        Pt_index = target.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = -torch.log2(Pt)

        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)