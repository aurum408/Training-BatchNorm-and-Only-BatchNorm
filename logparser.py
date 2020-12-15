import os
import matplotlib
from matplotlib import pyplot as plt

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
    data = { k:v for i in data for k, v in i.items()}
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


if __name__ == '__main__':
    fpath_SGDbn = "/Users/anastasia/Downloads/resnet32SGDbn_Dec-15-2020 (3).txt"
    fpath_SGDall = "/Users/anastasia/Downloads/resnet32SGDall_Dec-15-2020 (5).txt"
    fpath_LBFGSall = "/Users/anastasia/Downloads/resnet32LBFGSall_Dec-15-2020.txt"
    fpath_LBFGSbn = "/Users/anastasia/Downloads/resnet32LBFGSbn_Dec-15-2020 (2).txt"

    data = [parse_log(fpath) for fpath in [fpath_SGDbn, fpath_SGDall, fpath_LBFGSbn, fpath_LBFGSall]]
    print("ok")