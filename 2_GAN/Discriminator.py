import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, attributes_num, attribute_dim, hide_dim, user_dim):
        super(Discriminator, self).__init__()
        self.attrNum = attributes_num
        self.attrDim = attribute_dim
        self.hideDim = hide_dim
        self.userDim = user_dim

        self.dAttrsMatrix = nn.Embedding(2*attributes_num, attribute_dim)
        self.dWb1 = nn.Linear(attributes_num*attribute_dim + user_dim, hide_dim)
        self.dWb2 = nn.Linear(hide_dim, hide_dim)
        self.dWb3 = nn.Linear(hide_dim, user_dim)

    def forward(self, attrId, user_emb):
        attr = self.dAttrsMatrix(attrId.long().view(-1, 18))
        attrFeature = attr.view(-1, self.attrNum*self.attrDim)
        emb = torch.cat([attrFeature, user_emb], 1)

        l1 = torch.sigmoid(self.dWb1(emb))
        l2 = torch.sigmoid(self.dWb2(l1))
        D_logit = self.dWb3(l2)
        D_prob = torch.sigmoid(D_logit)

        return D_prob, D_logit