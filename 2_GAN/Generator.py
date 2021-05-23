import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, attributes_num, attribute_dim, hide_dim, user_dim):
        super(Generator, self).__init__()
        self.attrNum = attributes_num
        self.attrDim = attribute_dim
        self.hideDim = hide_dim
        self.userDim = user_dim

        self.gAttrsMatrix = nn.Embedding(2*attributes_num, attribute_dim)
        self.gWb1 = nn.Linear(attributes_num*attribute_dim, hide_dim)
        self.gWb2 = nn.Linear(hide_dim, hide_dim)
        self.gWb3 = nn.Linear(hide_dim, user_dim)

    def forward(self, attrId):

        attr = self.gAttrsMatrix(attrId.long().view(-1, 18))
        attrFeature = attr.view(-1, self.attrNum * self.attrDim)

        l1 = torch.sigmoid(self.gWb1(attrFeature))
        l2 = torch.sigmoid(self.gWb2(l1))
        fakeUser = torch.sigmoid(self.gWb3(l2))

        return fakeUser