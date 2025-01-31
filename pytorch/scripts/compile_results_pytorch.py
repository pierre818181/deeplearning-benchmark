# -*- coding: utf-8 -*-
# we always have to make sure to insert the right AWS credentials and topic ARN. The lambda will 
# have no way of knowing that the pod needs to be killed if we do not do that.
import json
import os
import subprocess
import re
import time
import requests
import re
import pandas as pd
from gql.transport.requests import RequestsHTTPTransport
from loguru import logger
from infer import install_packages_for_inference, run_inference
import boto3

gpu_type_to_inference_model = {
    '16GB': 'tiiuae/falcon-7b',
    '2x16GB': 'tiiuae/falcon-7b',
    '4x16GB': 'tiiuae/falcon-7b',
    '8x16GB': 'tiiuae/falcon-7b',

    '24GB': 'tiiuae/falcon-7b',
    '2x24GB': 'tiiuae/falcon-7b',
    '4x24GB': 'tiiuae/falcon-40b',
    '8x24GB': 'tiiuae/falcon-40b',

    '40GB': 'tiiuae/falcon-7b',
    '2x40GB': 'tiiuae/falcon-40b',
    '4x40GB': 'tiiuae/falcon-40b',
    '8x40GB': 'tiiuae/falcon-40b',

    '48GB': 'tiiuae/falcon-7b',
    '2x48GB': 'tiiuae/falcon-40b',
    '4x48GB': 'tiiuae/falcon-180B',
    '8x48GB': 'tiiuae/falcon-180B',

    '80GB': 'tiiuae/falcon-40b',
    '2x80GB': 'tiiuae/falcon-40b',
    '4x80GB': 'tiiuae/falcon-180B',
    '8x80GB': 'tiiuae/falcon-180B',
}

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
    "PyTorch_bert_large_squad_FP16": "bertLarge",
    "PyTorch_bert_base_squad_FP16": "bertBase",
    "PyTorch_SSD_FP32": "ssd32",
    "PyTorch_bert_large_squad_FP32": "bertLarge32",
    "PyTorch_bert_base_squad_FP32": "bertBase32",
}


def gather_throughput(
    list_test, list_system, name, system, version, path_result
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
                throughput = 0
                # Sift through all lines and only keep the last occurrence
                for _, line in enumerate(open(os.path.join(path, filename))):
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

def parse_desc(desc):
    pattern = re.compile(r"\^\.\*(.*?)\.\*\$$")
    match = pattern.search(desc)

    return (
        " ".join(match.group(1).strip().rstrip(":").strip().split("_"))
        if match
        else None
    )


def compile_results(list_test):
    system = "multiple"
    version = 1
    path = "/results"
    
    logger.info("list test")
    logger.info(list_test)
    try:
        list_system = list_system_multiple
        throughputs = {}
        throughput_errors = []
        for key in list_system:
            if key == os.environ["CONFIG"]:
                version = list_system[key][0][0]
                for test_name, value in sorted(list_test[version].items()):
                    throughput, throughput_description, errors = gather_throughput(
                        list_test,
                        list_system,
                        test_name,
                        key,
                        version,
                        path,
                    )
                    logger.info("throughput")
                    logger.info(throughput)
                    logger.info("throughput_description")
                    logger.info(throughput_description)
                    logger.info("errors")
                    logger.info(errors)

                    if errors:
                        for err in errors:
                            logger.error(err)
                            throughput_errors.append(err)
                    else:
                        throughputs[test_name_to_camel_case[test_name]] = (
                            f"{throughput} {parse_desc(throughput_description)}"
                        )

        return throughputs, throughput_errors
    except Exception as e:
        return {}, [str(e)]


def send_throughputs(client, topic_arn, message):
    message["podId"] = os.environ.get("POD_ID", "missing_pod_id")
    response = client.publish(
        TopicArn=topic_arn,
        Message=json.dumps(message)
    )

    logger.info(f"response from sns: {response}")
    # sleeping so that the lambda gets time to kill the pod.
    time.sleep(50)

if os.environ.get("TEST_TYPE", "heavy") == "lite":
    fp_32_tests, fp_16_tests, datasets = ["bert_base_squad_fp32"], ["bert_base_squad_fp16"], ["bert"]
elif os.environ.get("TEST_TYPE", "heavy") == "inference_only":
    fp_32_tests, fp_16_tests, datasets = [], [], []
else:
    fp_32_tests, fp_16_tests, datasets = [
                "bert_base_squad_fp32",
                "bert_large_squad_fp32",
                "ssd_fp32",
            ], [
                "bert_base_squad_fp16",
                "bert_large_squad_fp16",
                "ssd_amp",
            ], [
                "bert", 
                "object_detection"
            ]

def initialize_aws_client(access_key, secret_key, region):
    return boto3.client(
        'sns',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

def run_tests():
    try:
        config = os.environ.get("CONFIG", None)
        timeout = os.environ.get("TIMEOUT", None)
        precision = os.environ.get("PRECISION", None)
        pod_id = os.environ.get("POD_ID", None)
        test_type = os.environ.get("TEST_TYPE", "heavy")
        aws_key_id =  os.environ.get("MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID", None)
        aws_secret_access_key  = os.environ.get("MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY", None)
        aws_topic_arn = os.environ.get("MACHINE_BENCHMARK_TOPIC_ARN", None)
        if not config or not precision or not pod_id or not aws_key_id or not aws_secret_access_key or not aws_topic_arn:
            if not aws_key_id or not aws_secret_access_key or not aws_topic_arn or not pod_id:
                logger.error("""One of the critical environment variables is missing: MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
                                MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY, MACHINE_BENCHMARK_TOPIC_ARN. 
                                Cannot send request to terminate the pod.""")
            else:
                client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")
                send_throughputs(client, aws_topic_arn, {"errors": ["""One of the environment variables is missing: FINETUNING_CONFIG, INFERENCE_CONFIG, 
                                        TIMEOUT, PRECISION, POD_ID, TEST_TYPE, INFERENCE_PROMPTS_SIZE, MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
                                        MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY, MACHINE_BENCHMARK_TOPIC_ARN"""],
                                    })
            return
        logger.info(f"CONFIG: {config}, TIMEOUT: {timeout}, PRECISION: {precision}, POD_ID: {pod_id}, TEST_TYPE: {test_type}, AWS_ACCESS_KEY_ID: {aws_key_id}, AWS_SECRET_ACCESS_KEY: {aws_secret_access_key}, AWS_TOPIC_ARN: {aws_topic_arn}")

        client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")
        throughputs = {}
        runtime_errors = []
        errors = []

        run_training = test_type != "inference_only"
        if run_training:
            for ds in datasets:
                logger.info(f"downloading dataset: {ds}")
                cmd = ["/workspace/run_prepare.sh", ds]
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    logger.info(f"Successfully downloaded dataset: {ds}")
                else:
                    send_throughputs(client, aws_topic_arn, {"errors": [f"Failed to download dataset: {ds}"]})
                    return
            
            move_dataset_cmd = ["cp", "-r", "/data", "/workspace/data"]
            result = subprocess.run(
                move_dataset_cmd, capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info(f"Successfully downloaded dataset: {ds}")
            else:
                logger.error(f"Failed to copy dataset: {ds}")
                send_throughputs(client, aws_topic_arn, {"errors": [f"Failed to copy dataset: {ds}"]})
                return

            list_test = None  
            if precision == "fp32":
                tests_to_run = fp_32_tests
                list_test = list_test_fp32
            elif precision == "fp16":
                tests_to_run = fp_16_tests
                list_test = list_test_fp16
            else:
                tests_to_run = fp_32_tests + fp_16_tests
                list_test = list_test_fp32

            for test in tests_to_run:
                # TODO: fetch this from the env var. Format for this command:
                # -- ./run_benchmark.sh: the script that runs the benchmark
                # -- 8x24GB: the system configuration to be tested. This is in the format gpu_count x VRAM size
                # -- bert_base_squad_fp32: the name of the model to test it with
                logger.info(f"starting test {test}")
                command = ["./run_benchmark.sh", config, test, timeout]

                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        logger.info(output)
                        logger.info(f"completed command")
                        logger.info(command)
                        break
                    if output:
                        if "CUDA error: invalid device ordinal" in output:
                            send_throughputs(client, aws_topic_arn, {"errors": "CUDA error: invalid device ordinal. This most likely means that the system does not have the required number of GPUs"})
                            return
                        logger.info(output.strip())

                err = process.stderr.read()
                if err:
                    logger.error("something errored")
                    logger.error(err.strip())
                    errors.append(err.strip())
                    # send_throughput_resp({}, [err.strip()])
                    # return

            throughputs, runtime_errors = compile_results(list_test)
            logger.info("throughputs")
            logger.info(throughputs)
            logger.info("runtime errors")
            logger.info(runtime_errors)

            errors.extend(runtime_errors)
        
        do_inference = os.environ.get("TEST_TYPE", "heavy") == "inference_only" or os.environ.get("TEST_TYPE", "heavy") == "heavy"
        if do_inference:
            err = install_packages_for_inference()
            logger.info("error from setup")
            logger.info(err)
            if err != None:
                errors.append(err)
                logger.info("errored during setup. Skipping inference tests.")
            else:
                total_time, err = run_inference(gpu_type_to_inference_model.get(config, "tiiuae/falcon-7b"))
                throughputs["falconInferenceTime"] = int(total_time)
                logger.info(total_time)
                logger.info(err)

        err_string = "\n\n".join(errors)
        throughputs["errors"] = err_string
        send_throughputs(client, aws_topic_arn, throughputs)
    except Exception as e:
        logger.exception(e)
        client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")
        send_throughputs(client, aws_topic_arn, {"errors": str(e)})


if __name__ == "__main__":
    run_tests()