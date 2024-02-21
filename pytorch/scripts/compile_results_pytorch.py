# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse


import pandas as pd

# naming convention
# key: config name
# value: ([version, num_gpus], rename)
# version: 0 for pytorch:20.01-py3, and 1 for pytorch:20.10-py3
# num_gpus: sometimes num_gpus can't be inferred from config name (for example p3.16xlarge) or missing from the result log. So we ask for user to specify it here.
# rename: renaming the system so it is easier to read
# watt per gpu
# price per gpu

list_system_single = {
    "8x24GB": ([1, 8], "Runpod 8x 24GB", 100, 100),
}

# the price and wattage are all arbitrary but we can definitely use it in the future
list_system_multiple = {
    "16GB": ([1, 1], "Runpod 1x 16GB", 100, 100),
    "2x16GB": ([1, 2], "Runpod 1x 16GB", 100, 100),
    "4x16GB": ([1, 4], "Runpod 1x 16GB", 100, 100),
    "8x16GB": ([1, 8], "Runpod 8x 16GB", 100, 100),

    "24GB": ([1, 1], "Runpod 1x 24GB", 100, 100),
    "2x24GB": ([1, 2], "Runpod 1x 24GB", 100, 100),
    "4x24GB": ([1, 4], "Runpod 1x 24GB", 100, 100),
    "8x24GB": ([1, 8], "Runpod 8x 24GB", 100, 100),

    "40GB": ([1, 1], "Runpod 1x 40GB", 100, 100),
    "2x40GB": ([1, 2], "Runpod 1x 40GB", 100, 100),
    "4x40GB": ([1, 4], "Runpod 1x 40GB", 100, 100),
    "8x40GB": ([1, 8], "Runpod 8x 40GB", 100, 100),

    "48GB": ([1, 1], "Runpod 1x 48GB", 100, 100),
    "2x48GB": ([1, 2], "Runpod 1x 48GB", 100, 100),
    "4x48GB": ([1, 4], "Runpod 1x 48GB", 100, 100),
    "8x48GB": ([1, 8], "Runpod 8x 48GB", 100, 100),

    "80GB": ([1, 1], "Runpod 1x 80GB", 100, 100),
    "2x80GB": ([1, 2], "Runpod 1x 80GB", 100, 100),
    "4x80GB": ([1, 4], "Runpod 1x 80GB", 100, 100),
    "8x80GB": ([1, 8], "Runpod 8x 80GB", 100, 100),
}
list_test_fp32 = [
    # version 0: nvcr.io/nvidia/pytorch:20.01-py3 and 20.10-py3
    {
        "PyTorch_SSD_FP32": ("ssd", "^.*Training performance =.*$", -2),
        "PyTorch_ncf_FP32": ("ncf", "^.*best_train_throughput:.*$", -1),
        "PyTorch_bert_large_squad_FP32": (
            "bert_large_squad",
            "^.*training throughput:.*$",
            -1,
        ),
        "PyTorch_bert_base_squad_FP32": (
            "bert_base_squad",
            "^.*training throughput:.*$",
            -1,
        ),
    },
    # version 1: nvcr.io/nvidia/pytorch:22.09-py3
    {
        "PyTorch_SSD_FP32": ("ssd", "^.*Average images/sec:.*$", -1),
        "PyTorch_ncf_FP32": ("ncf", "^.*best_train_throughput.*$", 7),
        "PyTorch_bert_large_squad_FP32": (
            "bert_large_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
        "PyTorch_bert_base_squad_FP32": (
            "bert_base_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
    },
]

list_test_fp16 = [
    # version 0: nvcr.io/nvidia/pytorch:20.01-py3 and 20.10-py3
    {
        "PyTorch_SSD_AMP": ("ssd", "^.*Training performance =.*$", -2),
        "PyTorch_ncf_FP16": ("ncf", "^.*best_train_throughput:.*$", -1),
        "PyTorch_bert_large_squad_FP16": (
            "bert_large_squad",
            "^.*training throughput:.*$",
            -1,
        ),
        "PyTorch_bert_base_squad_FP16": (
            "bert_base_squad",
            "^.*training throughput:.*$",
            -1,
        ),
    },
    # version 1: nvcr.io/nvidia/pytorch:22.09-py3
    {
        "PyTorch_SSD_AMP": ("ssd", "^.*Average images/sec:.*$", -1),
        "PyTorch_ncf_FP16": ("ncf", "^.*best_train_throughput.*$", 7),
        "PyTorch_bert_large_squad_FP16": (
            "bert_large_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
        "PyTorch_bert_base_squad_FP16": (
            "bert_base_squad",
            "^.*training_sequences_per_second :.*$",
            -6,
        ),
    },
]


def gather_throughput(
    list_test, list_system, name, system, config_name, df, version, path_result
):
    column_name, key, pos = list_test[version][name]
    pattern = re.compile(key)
    path = path_result + "/" + system + "/" + name
    count = 0.000000001
    total_throughput = 0.0
    print(path)
    if os.path.exists(path):
        print(path, "exists")
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for i, line in enumerate(open(os.path.join(path, filename))):
                    for match in re.finditer(pattern, line):
                        # print(match.group().split(' ')) # for debug
                        try:
                            throughput = float(match.group().split(" ")[pos])
                            print(throughput, "throughput")
                        except:
                            pass

                if throughput > 0:
                    count += 1
                    total_throughput += throughput
                    flag = True

                if not flag:
                    print(system + "/" + name + " " + filename + ": something wrong")

        # return int(round(total_throughput / count, 2))
        print(config_name, int(round(total_throughput / count, 2)))
        return int(round(total_throughput / count, 2)), key
        # df.at[config_name, column_name] = int(round(total_throughput / count, 2))
    else:
        print(config_name, column_name)
        df.at[config_name, column_name] = 0
        return -1, key


def gather_bs(
    list_test, list_system, name, system, config_name, df, version, path_result
):
    column_name, key, pos = list_test[version][name]
    path = path_result + "/" + system + "/" + name

    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".para"):
                with open(os.path.join(path, filename)) as f:
                    first_line = f.readline()
                    df.at[config_name, column_name] = int(first_line.split(" ")[1])

    df.at[config_name, "num_gpu"] = list_system[system][0][1]
    df.at[config_name, "watt"] = list_system[system][2] * int(list_system[system][0][1])
    df.at[config_name, "price"] = list_system[system][3] * int(
        list_system[system][0][1]
    )

def remove_regex_from_desc(desc):
    pattern = re.compile(r'\^\.\*(.*?)\.\*\$$')
    match = pattern.search(desc)

    return " ".join(match.group(1).strip().rstrip(":").strip().split("_")) if match else None

def main():
    parser = argparse.ArgumentParser(description="Gather benchmark results.")

    parser.add_argument(
        "--path", type=str, default="results", help="path that has the results"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Choose becnhmark precision",
    )

    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["single", "multiple", "all"],
        help="Choose system type (single or multiple GPUs)",
    )

    parser.add_argument(
        "--version",
        type=int,
        default=1,
        choices=[0, 1],
        help="Choose benchmark version",
    )

    args = parser.parse_args()

    if args.precision == "fp32":
        list_test = list_test_fp32
    elif args.precision == "fp16":
        list_test = list_test_fp16
    else:
        sys.exit(
            "Wrong precision: " + args.precision + ", choose between fp32 and fp16"
        )

    if args.system == "single":
        list_system = list_system_single
    elif args.system == "multiple":
        list_system = list_system_multiple
    else:
        list_system = {}
        list_system.update(list_system_single)
        list_system.update(list_system_multiple)

    columns = []
    columns.append("num_gpu")
    columns.append("watt")
    columns.append("price")

    for test_name, value in sorted(list_test[args.version].items()):
        columns.append(list_test[args.version][test_name][0])
    list_configs = [list_system[key][1] for key in list_system]

    df_throughput = pd.DataFrame(index=list_configs, columns=columns)
    df_throughput = df_throughput.fillna(-1.0)

    df_bs = pd.DataFrame(index=list_configs, columns=columns)

    throughputs = {}
    for key in list_system:
        if key == os.environ["GPU_TYPE"]:
            version = list_system[key][0][0]
            config_name = list_system[key][1]
            for test_name, value in sorted(list_test[version].items()):
                throughput, throughput_description = gather_throughput(
                    list_test,
                    list_system,
                    test_name,
                    key,
                    config_name,
                    df_throughput,
                    version,
                    args.path,
                )
                if throughput == -1:
                    throughputs[test_name] = f"Model { test_name } did not run in latest run"
                else:
                    throughputs[test_name] = f"{throughput} {remove_regex_from_desc(throughput_description)}"
                gather_bs(
                    list_test,
                    list_system,
                    test_name,
                    key,
                    config_name,
                    df_bs,
                    version,
                    args.path,
                )

    print(throughputs)

    df_throughput.index.name = "name_gpu"
    df_throughput.to_csv("pytorch-train-throughput-" + args.precision + ".csv")

    df_bs.index.name = "name_gpu"
    df_bs.to_csv("pytorch-train-bs-" + args.precision + ".csv")


if __name__ == "__main__":
    main()
