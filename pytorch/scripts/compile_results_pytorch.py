# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import re
import argparse
import requests

import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

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

test_name_to_camel_case = {
    "PyTorch_SSD_AMP": "ssdAMP",
    "PyTorch_ncf_FP16": "ncf16",
    "PyTorch_bert_large_squad_FP16": "bertLarge",
    "PyTorch_bert_base_squad_FP16": "bertBase",
    "PyTorch_SSD_FP32": "ssd32",
    "PyTorch_ncf_FP32": "ncf32",
    "PyTorch_bert_large_squad_FP32": "bertLarge32",
    "PyTorch_bert_base_squad_FP32": "bertBase32",
}


def gather_throughput(
    list_test, list_system, name, system, config_name, df, version, path_result
):
    column_name, key, pos = list_test[version][name]
    pattern = re.compile(key)
    path = path_result + "/" + system + "/" + name
    count = 0.000000001
    total_throughput = 0.0
    errors = []
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                flag = False
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for i, line in enumerate(open(os.path.join(path, filename))):
                    for match in re.finditer(pattern, line):
                        try:
                            throughput = float(match.group().split(" ")[pos])
                        except:
                            pass

                if throughput > 0:
                    count += 1
                    total_throughput += throughput
                else:
                    errors.append(
                        f"unknown error - could not parse results for the test and file: { name + ' ' + filename  }"
                    )

        return int(round(total_throughput / count, 2)), key, errors
    else:
        errors.append(f"unknown error - test did not run: {name}")
        return -1, key, errors


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


def parse_desc(desc):
    pattern = re.compile(r"\^\.\*(.*?)\.\*\$$")
    match = pattern.search(desc)

    return (
        " ".join(match.group(1).strip().rstrip(":").strip().split("_"))
        if match
        else None
    )


def compile_results():
    precision = "fp32"
    system = "multiple"
    version = 1
    path = "/results"

    try:
        if precision == "fp32":
            list_test = list_test_fp32
        elif precision == "fp16":
            list_test = list_test_fp16
        else:
            sys.exit("Wrong precision: " + precision + ", choose between fp32 and fp16")

        if system == "single":
            list_system = list_system_single
        elif system == "multiple":
            list_system = list_system_multiple

        columns = []
        columns.append("num_gpu")
        columns.append("watt")
        columns.append("price")

        for test_name, value in sorted(list_test[version].items()):
            columns.append(list_test[version][test_name][0])
        list_configs = [list_system[key][1] for key in list_system]

        df_throughput = pd.DataFrame(index=list_configs, columns=columns)
        df_throughput = df_throughput.fillna(-1.0)

        throughputs = {}
        throughput_errors = []
        for key in list_system:
            if key == os.environ["GPU_TYPE"]:
                version = list_system[key][0][0]
                config_name = list_system[key][1]
                for test_name, value in sorted(list_test[version].items()):
                    throughput, throughput_description, errors = gather_throughput(
                        list_test,
                        list_system,
                        test_name,
                        key,
                        config_name,
                        df_throughput,
                        version,
                        path,
                    )
                    if errors:
                        for err in errors:
                            throughput_errors.append(err)
                    else:
                        throughputs[test_name_to_camel_case[test_name]] = (
                            f"{throughput} {parse_desc(throughput_description)}"
                        )

        return throughputs, throughput_errors
    except Exception as e:
        return {}, [str(e)]

def send_throughput_resp(throughputs, errors):
    api_key = os.environ["API_KEY"]
    if not api_key:
        print("API key not found")
        return

    if os.environ["ENV"] == "prod":
        url = f"https://api.runpod.io/graphql?api_key={api_key}"
    else:
        # TODO: update this before merging
        url = f"https://pierre-bastola-api.runpod.dev/graphql?api_key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }

    input_fields = [f'machineId: "{ os.environ["MACHINE_ID"] }"',]
    input_fields.append(f'benchmarkConfig: "{ os.environ["BENCHMARK_CONFIG"]}"')
    for test_name, test_result in throughputs.items():
        input_fields.append(f'{test_name}: "{ test_result }"')

    mutation = f"""mutation RecordBenchmark{{
        machineRecordBenchmark(input: {{
            {
                ", ".join(input_fields)
            }
        }})
    }}"""

    print("sending mutation", mutation)

    data = json.dumps({"query": mutation})
    response = requests.post(url, headers=headers, data=data, timeout=30)

    print(response.text)
    print(response.status_code)


# TODO: update this
files = ["/data/bert_base", "/data/bert_large", "/data/squad"]

# tests_to_run = ["bert_base_squad_fp32", "bert_large_squad_fp32", "ssd_fp32", "ncf_fp32"]

# use this for testing
tests_to_run = ["bert_base_squad_fp32"]

def run_tests():
    benchmark_config = os.environ["BENCHMARK_CONFIG"]
    if not benchmark_config:
        send_throughput_resp({}, ["Test config not found"])
        return

    for file in files:
        if not os.path.exists(file):
            send_throughput_resp({}, [f"File not found: { file }"])
            return

    errors = []
    for test in tests_to_run:
        # TODO: fetch this from the env var. Format for this command:
        # -- ./run_benchmark.sh: the script that runs the benchmark
        # -- 8x24GB: the system configuration to be tested. This is in the format gpu_count x VRAM size
        # -- bert_base_squad_fp32: the name of the model to test it with
        print(f"starting test {test}")
        command = ["./run_benchmark.sh", benchmark_config, test, "2500"]

        # result = subprocess.run(command, text=True, capture_output=True)
        # if result.returncode != 0:
        #     print("something errored", result.stderr)
        #     send_throughput_resp({}, [result.stderr])
        #     return
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                print(output, "breaking")
                break
            if output:
                print("output:", output.strip())

        err = process.stderr.read()
        if err:
            print("something errored", err.strip())
            errors.append(err.strip())
            # send_throughput_resp({}, [err.strip()])
            # return

    throughputs, runtime_errors = compile_results()
    print(runtime_errors + errors)
    send_throughput_resp(throughputs, runtime_errors + errors)

if __name__ == "__main__":
    run_tests()