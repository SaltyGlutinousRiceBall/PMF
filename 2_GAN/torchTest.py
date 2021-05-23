import torch
import torch.nn as nn

a = nn.Embedding(4, 5)

b = torch.LongTensor([[0, 1, 2], [3, 2, 1]])

c = a(b)

print(a)
print("------------")


print(b)
print("------------")


print(c)
print("------------")