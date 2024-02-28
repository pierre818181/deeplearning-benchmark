FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt-get install git

RUN mkdir /scripts

WORKDIR /workspace

# The purpose is to set the directory structure in the right way to be able to run these tests.
# There is a better structure for the directories but that will require changing the script
# which is a high-bandwidth task
RUN mkdir benchmark && \
    git clone -b lambda/benchmark https://github.com/LambdaLabsML/DeepLearningExamples.git && \
    git clone https://github.com/pierre818181/deeplearning-benchmark.git && \
    cp -r DeepLearningExamples/PyTorch/* benchmark && \
    cp -r deeplearning-benchmark/pytorch/scripts/* /scripts && \
    cp -r deeplearning-benchmark/pytorch/scripts/* .

RUN pip install -r requirements.txt

# Test it with this. The bert finetuning model is a lot smaller than the ones for ncf and object detection
# RUN ./run_prepare.sh bert
# RUN ./run_prepare.sh object_detection

# Initialize datasets for the three models we will run
# RUN ./run_prepare.sh bert && ./run_prepare.sh ncf && ./run_prepare.sh object_detection
# TODO: test without this
# RUN cp -r /data /workspace/data

CMD ["python3", "compile_results_pytorch.py"]

# run a blocking call, get inside the container with exec and then test stuff out
# CMD ["python3", "-m", "http.server"]

# sudo docker run -e "GPU_TYPE=8x24GB" -e "MACHINE_ID=ghnlbwbe44923" -e "API_KEY=7699e9fe969179fcce8f3c04e91c195f9a4e0971644ea391af37d7fcacac2d448c9a71525117d8ef44b20edd2deba1abd318150cbeb9a146683102bdf6db354a" -e "ENV=dev" -e "BENCHMARK_CONFIG=8x24GB" -e "PRECISION=fp32" -e "TIMEOUT=1500" --shm-size=3000g --gpus all -it host /bin/bash
# GPU_TYPE="8x24GB" MACHINE_ID="ghnlbwbe44923" API_KEY="7699e9fe969179fcce8f3c04e91c195f9a4e0971644ea391af37d7fcacac2d448c9a71525117d8ef44b20edd2deba1abd318150cbeb9a146683102bdf6db354a" ENV="dev" BENCHMARK_CONFIG="8x24GB" PRECISION="fp32" TIMEOUT=1500 python3 compile_results_pytorch.py