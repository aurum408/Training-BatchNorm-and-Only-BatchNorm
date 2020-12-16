import os
import matplotlib
from matplotlib import pyplot as plt
from typing import List

names = ["Test", "Epoch", "Time", "Data", "Loss", "Prec"]


def parse_metric(metric: str) -> dict:
    for name in names:
        if metric.startswith(name):
            metric_name = name
            break
    if metric_name == "Epoch":
        n_epoch, n_batch = metric.split("[")[1:]
        n_epoch = int(n_epoch.split("]")[0])
        n_batch = n_batch.split(']')[0]
        batch_index, n_batches = list(map(lambda x: int(x), n_batch.split("/")))

        value = [n_epoch, batch_index, n_batches]

    elif metric_name == "Test":
        val = metric.split('/')
        batch_index = int(val[0][-1])
        n_batches = int(val[1][0])
        value = [batch_index, n_batches]
    else:
        val = float(metric.split("(")[1].split(")")[0])
        value = [val]

    data = {metric_name: value}
    return data


def parse_line(line: str) -> dict:
    metrics = line.split("\t")
    data = [parse_metric(metric) for metric in metrics]
    data = {k: v for i in data for k, v in i.items()}
    return data


def parse_log(fpath: str) -> dict:
    # logs = {name:[] for name in names}
    # logs = {"Test": {name:[] for name in names}}
    test_logs = {name:[] for name in names}
    train_logs = {name:[] for name in names}
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    print("ok")
    for line in lines[1:]:
        if line.startswith("\n"):
            continue
        else:
            data = parse_line(line)
            if "Epoch" in data.keys():
                for k, v in data.items():
                    train_logs[k].append(v[0])
            else:
                for k, v in data.items():
                    test_logs[k].append(v[0])
            # info.append(data)

    n_epoch = max(train_logs["Epoch"])

    test_loss = test_logs["Loss"]
    train_loss = train_logs["Loss"]
    test_acc = test_logs["Prec"]
    train_acc = train_logs["Prec"]

    ntrain = len(train_loss) // n_epoch
    ntest = len(test_loss) // n_epoch


    train_acc = [train_acc[i] for i in range(0, len(train_acc), ntrain)]
    train_loss = [train_loss[i] for i in range(0, len(train_loss), ntrain)]

    val_acc = [test_acc[i] for i in range(0, len(test_acc), ntest)]
    val_loss = [test_loss[i] for i in range(0, len(test_loss), ntest)]

    info = {"train_acc": train_acc, "train_loss": train_loss,
            "val_acc": val_acc, "val_loss": val_loss}

    return info

def _plot(data: list, label: str, color: str) -> None:
    plt.plot(data, color, label=label)


def plot(configs: list, xlabel: str, ylabel: str,
         title: str, legend: str, fname: str) -> None:

    fig = plt.figure()
    ax = plt.subplot(111)

    # plt.figure()
    plt.title(title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    for cfg in configs:
        _plot(*cfg)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig("tmp/{}.png".format(fname))


if __name__ == '__main__':
    SGDbn = parse_log("/Users/anastasia/Downloads/resnet32SGDbn_Dec-15-2020 (3).txt")
    SGDall = parse_log("/Users/anastasia/Downloads/resnet32SGDall_Dec-15-2020 (5).txt")

    LBFGSall = parse_log("/Users/anastasia/Downloads/resnet32LBFGSall_Dec-15-2020 (3).txt")
    LBFGSbn = parse_log("/Users/anastasia/Downloads/resnet32LBFGSbn_Dec-15-2020 (8).txt")

    plot([[SGDbn["train_loss"][1:], "SGD_bn", "r"],
          [SGDall["train_loss"][1:], "SGD_all", "b"],
          [LBFGSbn["train_loss"][1:], "LBFGS_bn", "g"],
          [LBFGSall["train_loss"][1:], "LBFGS_all", "y"]],
         "epoch", "loss", "train_loss", "best", "train_loss")

    plot([[SGDbn["train_loss"][1:], "SGD_bn_train", "r"],
          [SGDall["train_loss"][1:], "SGD_all_train", "b"],
          [LBFGSbn["train_loss"][1:], "LBFGS_bn_train", "g"],
          [LBFGSall["train_loss"][1:], "LBFGS_all_train", "y"],
          [SGDbn["val_loss"][1:], "SGD_bn_val", "purple"],
          [SGDall["val_loss"][1:], "SGD_all_val", "orange"],
          [LBFGSbn["val_loss"][1:], "LBFGS_bn_val", "pink"],
          [LBFGSall["val_loss"][1:], "LBFGS_all_val", "olive"]],
         "epoch", "loss", "loss", "best", "loss")

    plot([[SGDbn["val_loss"][1:], "SGD_bn", "r"],
          [SGDall["val_loss"][1:], "SGD_all", "b"],
          [LBFGSbn["val_loss"][1:], "LBFGS_bn", "g"],
          [LBFGSall["val_loss"][1:], "LBFGS_all", "y"]],
         "epoch", "loss", "val_loss", "best", "val_loss")

    plot([[SGDbn["train_acc"][1:], "SGD_bn", "r"],
          [SGDall["train_acc"][1:], "SGD_all", "b"],
          [LBFGSbn["train_acc"][1:], "LBFGS_bn", "g"],
          [LBFGSall["train_acc"][1:], "LBFGS_all", "y"]],
         "epoch", "accuracy", "train_acc", "best", "train_acc")

    plot([[SGDbn["train_acc"][1:], "SGD_bn_train", "r"],
          [SGDall["train_acc"][1:], "SGD_all_train", "b"],
          [LBFGSbn["train_acc"][1:], "LBFGS_bn_train", "g"],
          [LBFGSall["train_acc"][1:], "LBFGS_all_train", "y"],
          [SGDbn["val_acc"][1:], "SGD_bn_val", "purple"],
          [SGDall["val_acc"][1:], "SGD_all_val", "orange"],
          [LBFGSbn["val_acc"][1:], "LBFGS_bn_val", "pink"],
          [LBFGSall["val_acc"][1:], "LBFGS_all_val", "olive"]],
         "epoch", "accuracy", "accuracy", "best", "acc")

    plot([[SGDbn["val_acc"][1:], "SGD_bn", "r"],
          [SGDall["val_acc"][1:], "SGD_all", "b"],
          [LBFGSbn["val_acc"][1:], "LBFGS_bn", "g"],
          [LBFGSall["val_acc"][1:], "LBFGS_all", "y"]],
         "epoch", "accuracy", "val_acc", "best", "val_acc")



    print("ok")