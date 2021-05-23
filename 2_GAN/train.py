import support
import time

import torch.optim as optim

from Generator import *
from Discriminator import *

device = torch.device('cuda')


def train(attrNum, attrDim, hideDim, userDim, batch_size, epochs, learning_rate, alpha):
    generator = Generator(attrNum, attrDim, hideDim, userDim)  # 生成器
    discriminator = Discriminator(attrNum, attrDim, hideDim, userDim)  # 判别器
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 定义优化器
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=alpha)
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=alpha)
    BCELoss = nn.BCELoss()

    support.shuffle()
    support.shuffle2()

    # 开始训练
    for epoch in range(epochs+1):
        start = time.time()

        # 训练生成器
        index = 0
        while index < 253236:
            if index + batch_size <= 253236:
                train_attr_batch, _, _, _ = support.get_data(
                    index, index + batch_size)

            index += batch_size

            train_attr_batch = train_attr_batch.to(device)

            fake_user_emb = generator(train_attr_batch)
            fake_user_emb = fake_user_emb.to(device)
            D_fake, D_logit_fake = discriminator(train_attr_batch, fake_user_emb)

            G_loss = BCELoss(D_fake, torch.ones_like(D_fake)).mean()

            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

        # 训练判别器
        index = 0
        while index < 253236:
            # discriminator.zero_grad()

            if index + batch_size <= 253236:
                train_attr_batch, train_user_emb_batch, counter_attr_batch, counter_user_emb_batch = support.get_data(
                    index, index + batch_size)
            index = index + batch_size

            train_attr_batch = train_attr_batch.to(device)
            train_user_emb_batch = train_user_emb_batch.to(device)
            counter_attr_batch = counter_attr_batch.to(device)
            counter_user_emb_batch = counter_user_emb_batch.to(device)

            fake_user_emb = generator(train_attr_batch)
            fake_user_emb.to(device)

            D_real, D_logit_real = discriminator(train_attr_batch, train_user_emb_batch)
            D_fake, D_logit_fake = discriminator(train_attr_batch, fake_user_emb)
            D_counter, D_logit_counter = discriminator(counter_attr_batch, counter_user_emb_batch)

            D_loss_real = BCELoss(D_real, torch.ones_like(D_real)).mean()
            D_loss_fake = BCELoss(D_fake, torch.zeros_like(D_fake)).mean()
            D_loss_counter = BCELoss(D_counter, torch.zeros_like(D_counter)).mean()
            D_loss = D_loss_real + D_loss_fake + D_loss_counter
            optimizerD.zero_grad()

            D_loss.backward()
            optimizerD.step()

        end = time.time()

        # 记录训练状态
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), "Generator/epoch" + str(epoch) + ".ld")
            torch.save(discriminator.state_dict(), "Discriminator/" + str(epoch) + ".ld")

        print("epoch: {}/{}, G_loss:{:.4f}, D_loss:{:.4f}, time:{:.4f}".format(epoch, epochs, G_loss, D_loss,
                                                                               end - start))


if __name__ == '__main__':
    print("开始训练")
    attrNum = 18
    attrDim = 5
    batch_size = 1024
    hideDim = 100
    userDim = 18
    learning_rate = 0.0001
    alpha = 0.0001
    epochs = 300
    train(attrNum, attrDim, hideDim, userDim, batch_size, epochs, learning_rate, alpha)
