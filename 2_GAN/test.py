from Generator import *
from Discriminator import *
import support

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

attrNum = 18
attrDim = 5
hideDim = 100
userDim = 18

generator = Generator(attrNum, attrDim, hideDim, userDim)
discriminator = Discriminator(attrNum, attrDim, hideDim, userDim)

test_items, test_attrs = support.get_testdata()
test_attrs = torch.Tensor(test_attrs)

for i in range(30):
    generator.load_state_dict(torch.load('Generator/epoch' + str(i * 10) + '.ld'))
    test_G_user = generator(test_attrs)
    precision10, precision20, map10, map20, ndcg10, ndcndcg20 = support.test(
        test_items, test_G_user.detach())
    print(
        "epoch{}  precision_10:{:.4f},precision_20:{:.4f},map_10:{:.4f},map_20:{:.4f},ndcg_10:{:.4f},ndcg_20:{:.4f}".format(
            i * 10, precision10, precision20, map10, map20, ndcg10, ndcndcg20))
