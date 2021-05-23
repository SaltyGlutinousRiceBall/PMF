import numpy as np
import pandas as pd
import torch
import evall

train = np.array(pd.read_csv('train_data.csv', header=None))
neg = np.array(pd.read_csv('neg_data.csv', header=None))
user_emb_matrix = np.array(pd.read_csv(r'util/user_emb.csv', header=None))
test_item = np.array(pd.read_csv('test_item.csv', header=None).astype(np.int32))
test_attribute = np.array(pd.read_csv('test_attribute.csv', header=None).astype(np.int32))

user_attribute_matrix = np.array(pd.read_csv(r'util/user_attribute.csv', header=None))
ui_matrix = np.array(pd.read_csv(r'util/ui_matrix.csv', header=None))


def get_testdata():
    return test_item, test_attribute


def shuffle():
    np.random.shuffle(train)


def shuffle2():
    np.random.shuffle(neg)


def get_traindata(start_index, end_index):
    '''get train samples'''
    batch_data = train[start_index: end_index]

    #    print(  batch_data)
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    attr_batch = [int(i) for line in attr_batch for i in line]
    real_user_emb_batch = user_emb_matrix[user_batch]

    return user_batch, item_batch, attr_batch, real_user_emb_batch


def get_negdata(start_index, end_index):
    '''get negative samples'''
    batch_data = neg[start_index: end_index]

    #    print(  batch_data)
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    attr_batch = [int(i) for line in attr_batch for i in line]
    real_user_emb_batch = user_emb_matrix[user_batch]

    return user_batch, item_batch, attr_batch, real_user_emb_batch


# 获取训练数据
def get_data(begin, end):
    train_user_batch, train_item_batch, train_attr_batch, train_user_emb_batch = get_traindata(begin, end)
    counter_user_batch, counter_item_batch, counter_attr_batch, counter_user_emb_batch = get_negdata(begin, end)

    return torch.Tensor(train_attr_batch), torch.Tensor(train_user_emb_batch), torch.Tensor(
        counter_attr_batch), torch.Tensor(counter_user_emb_batch)


def get_intersection_similar_user(G_user, k):
    # G_user = np.round(G_user)
    #    G_user = np.around(G_user)
    user_emb_matrixT = np.transpose(user_attribute_matrix)
    A = np.matmul(G_user, user_emb_matrixT)
    intersection_rank_matrix = np.argsort(-A)
    #    print( intersection_rank_matrix[0,:k])
    #    print( intersection_rank_matrix[50,:k])
    return intersection_rank_matrix[:, 0:k]


def test(test_item_batch, test_G_user):
    k_value = 20
    test_BATCH_SIZE = np.size(test_item_batch)

    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value)

    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        for test_u in test_userlist:

            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
    p_at_20 = round(count / (test_BATCH_SIZE * k_value), 4)

    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        RS.append(r)
    #    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )
    M_at_20 = evall.mean_average_precision(RS)

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
    #    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE)
    G_at_20 = ans / test_BATCH_SIZE
    k_value = 10

    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        for test_u in test_userlist[:k_value]:

            if ui_matrix[test_u, test_i] == 1:
                count = count + 1
    p_at_10 = round(count / (test_BATCH_SIZE * k_value), 4)

    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        RS.append(r)
    #    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )
    M_at_10 = evall.mean_average_precision(RS)

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):
        r = []
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
    #    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE)
    G_at_10 = ans / test_BATCH_SIZE

    return p_at_10, p_at_20, M_at_10, M_at_20, G_at_10, G_at_20
