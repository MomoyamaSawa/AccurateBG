import argparse
import json
import os

import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader import DataReader


def personalized_train_ohio(epoch, ph, path="../output"):
    """
    用于个性化训练模型。函数接受三个参数：epoch（训练轮数），ph（预测时长），和path（输出路径）。
    """
    # read in all patients data
    # 数据读取和处理，读取2018年和2020年各个病人的训练数据，并存储在字典train_data中。
    pid_2018 = [559, 563, 570, 588, 575, 591]
    pid_2020 = [540, 552, 544, 567, 584, 596]
    pid_year = {2018: pid_2018, 2020: pid_2020}
    train_data = dict()
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            reader = DataReader(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )
            train_data[pid] = reader.read()
    # add test data of 2018 patient
    # 测试数据处理，读取2018年各个病人的测试数据，并存储在列表test_data_2018中。
    use_2018_test = True
    standard = False  # do not use standard
    test_data_2018 = []
    for pid in pid_2018:
        reader = DataReader(
            "ohio", f"../data/OhioT1DM/2018/test/{pid}-ws-testing.xml", 5
        )
        test_data_2018 += reader.read()

    # a dumb dataset instance
    # 数据集实例化，创建一个虚拟的train_dataset实例，用于后续的训练过程。设置一些基本参数如sampling_horizon, prediction_horizon, scale, 和outtype。
    train_dataset = CGMSDataSeg(
        "ohio", "../data/OhioT1DM/2018/train/559-ws-training.xml", 5
    )
    sampling_horizon = 7
    prediction_horizon = ph
    scale = 0.01
    outtype = "Same"
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    # 读取配置文件，超参数
    with open(os.path.join(path, "config.json")) as json_file:
        config = json.load(json_file)
    argv = (
        config["k_size"],
        config["nblock"],
        config["nn_size"],
        config["nn_layer"],
        config["learning_rate"],
        config["batch_size"],
        epoch,
        config["beta"],
    )
    l_type = config["loss"]
    # test on patients data
    # 训练和测试模型
    outdir = os.path.join(path, f"ph_{prediction_horizon}_{l_type}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    all_errs = []
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            # only check results of 2020 patients
            if pid not in pid_2020:
                continue
            # 100 is dumb if set_cutpoint is used
            train_pids = set(pid_2018 + pid_2020) - set([pid])
            local_train_data = []
            if use_2018_test:
                local_train_data += test_data_2018
            for k in train_pids:
                local_train_data += train_data[k]
            print(f"Pretrain data: {sum([sum(x) for x in local_train_data])}")
            train_dataset.data = local_train_data
            train_dataset.set_cutpoint = -1
            train_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                100,
                False,
                outtype,
                1,
                standard,
            )
            # 训练
            regressor(train_dataset, *argv, l_type, outdir)
            # fine-tune on personal data
            # 微调
            target_test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            target_test_dataset.set_cutpoint = 1
            target_test_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                0.01,
                False,
                outtype,
                1,
                standard,
            )
            target_train_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )

            target_train_dataset.set_cutpoint = -1
            target_train_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                100,
                False,
                outtype,
                1,
                standard,
            )
            err, labels = test_ckpt(target_test_dataset, outdir)
            errs = [err]
            transfer_res = [labels]
            for i in range(1, 4):
                # 训练微调模型力
                err, labels = regressor_transfer(
                    target_train_dataset,
                    target_test_dataset,
                    config["batch_size"],
                    epoch,
                    outdir,
                    i,
                )
                errs.append(err)
                transfer_res.append(labels)
            transfer_res = np.concatenate(transfer_res, axis=1)
            np.savetxt(
                f"{outdir}/{pid}.txt",
                transfer_res,
                fmt="%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
            )
            all_errs.append([pid] + errs)
    all_errs = np.array(all_errs)
    np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%d %.4f %.4f %.4f %.4f")


def main():
    """
    解析命令行参数，并调用personalized_train_ohio函数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--prediction_horizon", type=int, default=6)
    parser.add_argument("--outdir", type=str, default="../ohio_results")
    args = parser.parse_args()

    personalized_train_ohio(args.epoch, args.prediction_horizon, args.outdir)


if __name__ == "__main__":
    main()
