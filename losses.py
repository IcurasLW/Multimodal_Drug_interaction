import numpy as np
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RatioLoss(nn.Module):
    def __init__(self, alphas:list=None, gamma=2, reduction:str='mean'):
        super().__init__()
        self.alphas = alphas # The inverse of frequency of class
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, probs, target):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        
        Pt_index = target.squeeze()
        pred_p = torch.max(probs, dim=1)[0]
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        # loss = -((1-Pt) ** self.gamma * torch.log2(Pt) + (torch.log2(Pt/pred_p)))     # Focal_loss + Ratio loss
        # loss = -3*(1-Pt)**2 * torch.log2((Pt/pred_p) * Pt)
        # loss = -((1-Pt) ** (pred_p - Pt)*(self.gamma)) * (torch.log2(Pt))
        # loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)                               # Focal_loss
        # loss = -torch.log2(Pt/pred_p)
        # loss = torch.log2(Pt/pred_p)**2 + (-torch.log2(pred_p))
        # loss = torch.log2(pred_p/Pt)**2 + (-torch.log2(pred_p))
        loss = -torch.log2((Pt/pred_p)*Pt)                                              # Ratio loss
        # loss = (1-Pt)**self.gamma * ((torch.log2((Pt**2)/pred_p))**2)
        # loss = ((1-Pt)) * (torch.log2((Pt/pred_p)*Pt))**2
        
        # Vanish-loss(效果不好)
        # ce_loss = -torch.log2(Pt)
        # f_loss = (1-Pt)**2 * (torch.log2(Pt))**2
        # loss = (1-((torch.e**Pt)-1)/((torch.e**pred_p)-1)) * f_loss + ((torch.e**Pt)-1)/((torch.e**pred_p)-1) * ce_loss
        # loss = ((1-Pt)) * (torch.log2(((Pt**2)/pred_p)*Pt))**2
        
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