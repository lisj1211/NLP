import numpy as np
import matplotlib.pyplot as plt


def save_loss_curve(train_loss, model_name):
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)  # 去除上边框
    ax.spines['right'].set_visible(False)  # 去除右边框

    plt.xlabel('iters')
    plt.ylabel('loss')

    x = np.array(list(range(len(train_loss[0]))))*100
    for i, name in enumerate(model_name):
        plt.plot(x, train_loss[i], label=name)
        plt.legend()
    plt.title('Loss curve')
    plt.savefig('loss_curve1.png')


if __name__ == "__main__":
    model_name = ["BertFreeze", "BertUnfreeze", "LSTM", "FastText", "TextCNN"]
    loss_list = [np.load("{}_loss.npy".format(n)) for n in model_name]
    save_loss_curve(loss_list, model_name)
