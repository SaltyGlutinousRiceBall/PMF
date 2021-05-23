import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv(r'data/train_data.csv', usecols=['userId', 'itemId', 'gerne']))
test_data = pd.read_csv(r'data/test_data.csv', usecols=['itemId', 'gerne'])
test_data.drop_duplicates(keep='first', inplace=True)
test_data = np.array(test_data)


def construct_traindata():
    #  这个方法是用来构建train数据的 构建完了 就用不到了  生成的是  train_data.csv

    for i in train_data:
        i[2] = i[2][1:-1]
    for i in train_data:
        tmp = np.linspace(0, 34, 18)
        li = np.int32(i[2].split(','))
        for j in li:
            tmp[j] = tmp[j] + 1
        i[2] = np.array(tmp, dtype=np.int32)
    print(train_data)
    train = pd.DataFrame(train_data)
    train.to_csv('train_data.csv', header=None, index=0)


def construct_testdata():
    for i in test_data:
        i[1] = i[1][1:-1]
    print(test_data)
    item_batch = [x[0] for x in test_data]
    attribute = []
    for i in test_data:
        tmp = np.linspace(0, 34, 18)
        li = np.int32(i[1].split(','))
        print(li)
        for j in li:
            tmp[j] = tmp[j] + 1
        attribute.append(tmp)
    print(len(item_batch))
    print(len(attribute))
    item = pd.DataFrame(item_batch)
    item.to_csv('test_item.csv', header=None, index=0)
    attribute = pd.DataFrame(attribute)
    attribute.to_csv('test_attribute.csv', header=None, index=0)


construct_traindata()
construct_testdata()