import numpy as np
import dataset

data_dir = 'dataset/ml-100k/u.data'


class PMF(object):
    def __init__(self,
                 train_data,
                 test_data,
                 n,
                 m,
                 k=10,
                 learning_rate=0.015,
                 epoch=50,
                 hyper_paramter=0.1):
        self._train = train_data
        self._test = test_data
        self.user_num = n
        self.item_num = m
        self.factor_num = k
        self.lr = learning_rate
        self.epoch = epoch
        self.lamda = hyper_paramter

    def train(self):
        U = np.random.normal(0, 0.05, (self.user_num, self.factor_num))  # 初始化用户隐含特征矩阵U
        V = np.random.normal(0, 0.1, (self.item_num, self.factor_num))  # 初始化电影隐含特征矩阵V
        record_list = list()  # 用来存下训练过程中的损失值及RMSE

        for step in range(0, self.epoch):
            E = 0.0  # 目标函数的值
            i = 0

            for data in self._train:
                user, movie, rate = data

                predict_rate = np.dot(U[user], V[movie].T)  # 根据对应用户隐含特征与电影隐含特征预测出的评分
                err = rate - predict_rate  # 真实评分与预测评分的误差
                e1 = err * err   # 目标函数E的第一部分
                e2 = self.lamda * np.square(U[user]).sum()  # 目标函数E的第二部分
                e3 = self.lamda * np.square(V[movie]).sum()  # 目标函数E的第三部分

                U[user] += self.lr * (err * V[movie] - self.lamda * U[user])  # 根据梯度对用户特征向量更新
                V[movie] += self.lr * (err * U[user] - self.lamda * V[movie])  # 根据梯度对电影特征向量更新

                E += 0.5 * (e1 + e2 + e3)  # 目标函数
                # rmse = self.caculate_rmse(U, V)  # 均方根误差
                # record_list.append([E, rmse])
                #
                # if i % 20 == 0:  # 每训练20次输出一次结果
                #     print('iteration:%d  loss:%.3f  rmse:%.5f' % (i, E, rmse))
                # i += 1

            # 每次都计算rmse，训练一个epoch需要一个多小时，因此改成一个epoch输出一次rmse
            rmse = self.caculate_rmse(U, V)
            record_list.append([E, rmse])
            print('step:%d  loss:%.3f  rmse:%.5f' % (step, record_list[-1][0], record_list[-1][1]))

        print('训练结束  loss:%.3f  rmse:%.5f' % (record_list[-1][0], record_list[-1][1]))
        return U, V, record_list

    def caculate_rmse(self, U, V):
        rmse = 0.0
        for ttt in self._test:
            user, movie, rate = ttt
            err = rate - np.dot(U[user], V[movie].T)
            rmse += err * err
        rmse = np.sqrt(rmse / len(self._test))
        return rmse


if __name__ == '__main__':
    user_num, item_num, train_data, test_data, user_set, movie_set, all_data = \
        dataset.load_data(data_dir=data_dir, ratio=0.8)
    pmf = PMF(train_data, test_data, user_num, item_num)
    U, V, record_list = pmf.train()



