import torch
import torch.nn as nn
import torch.nn.functional as functional


class Ensemble(nn.Module):
    def __init__(self, models,alpha):
        super(Ensemble, self).__init__()
        self.models = models
        self.alpha = alpha

    def forward(self, x):
        back_ground = []
        classes = []
        for mod in self.models:
            x_i,_,_ = mod(x)
            back_ground.append(x_i[:,[0]])
            classes.append(x_i[:,1:])
        out_0 = (sum(back_ground))*(self.alpha/len(self.models))
        out_1= torch.cat(classes,axis=1)
        out = torch.cat([out_0,out_1], axis=1)
        return out
