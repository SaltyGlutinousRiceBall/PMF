import numpy as np


def load_data(data_dir, ratio=0.8):
    # 输入: u.data路径 训练集占比
    # 返回：用户数量 电影数量 训练集[用户索引，电影索引，评分] 测试集[用户索引，电影索引，评分]
    # 其中用户及电影在R矩阵中的索引，是用户及电影在数据中出现的顺序，而不是用户及电影ID
    user_set = dict()
    item_set = dict()
    data = list()

    u_index, it_index = 0, 0

    with open(data_dir) as f:
        for line in f.readlines():
            u_id, it_id, rat, _ = line.split()

            if u_id not in user_set:
                user_set[u_id] = u_index
                u_index += 1

            if it_id not in item_set:
                item_set[it_id] = it_index
                it_index += 1

            data.append([user_set[u_id], item_set[it_id], float(rat)])  # [用户索引，电影索引，评分]

    f.close()

    np.random.shuffle(data)
    train = data[0:int(len(data) * ratio)]
    test = data[int(len(data) * ratio):]

    return u_index, it_index, train, test, user_set, item_set, data

