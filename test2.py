# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_loss = 5
    val_loss = 5
    train_acc = 0.0
    val_acc = 0.0

    x = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(200):
        # 生成数据，此处应根据实际训练过程获取训练集loss和acc
        # 以及验证集loss和acc
        train_loss -= epoch * 0.1
        val_loss -= epoch * 0.11

        train_acc += epoch * 0.01
        val_acc += epoch * 0.011

        x.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        plt.figure(figsize=(3, 6), dpi=100)
    # 创建两行一列的图，并指定当前使用第一个图
    plt.subplot(2, 1, 1)
    try:
        train_loss_lines.remove(train_loss_lines[0])  # 移除上一步曲线
        val_loss_lines.remove(val_loss_lines[0])
    except Exception:
        pass

    train_loss_lines = plt.plot(x, train_loss_list, 'r', lw=1)  # lw为曲线宽度
    val_loss_lines = plt.plot(x, val_loss_list, 'b', lw=1)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train_loss",
                "val_loss"])

    # # 创建两行一列的图，并指定当前使用第二个图
    plt.subplot(2, 1, 2)
    try:
        train_acc_lines.remove(train_acc_lines[0])  # 移除上一步曲线
        val_acc_lines.remove(val_acc_lines[0])
    except Exception:
        pass

    train_acc_lines = plt.plot(x, train_acc_list, 'r', lw=1)  # lw为曲线宽度
    val_acc_lines = plt.plot(x, val_acc_list, 'b', lw=1)
    plt.title("acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend(["train_acc",
                "val_acc"])

    plt.show()
    plt.pause(0.1)  # 图片停留0.1s

