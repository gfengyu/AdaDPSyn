# AdaDPSyn

Codes for "Data-adaptive Differentially Private Prompt Synthesis for In-Context Learning".


## Requirements

#### 1. vLLM OpenAI Compatible Server

To use the vLLM server for model inference, follow the steps below:

##### Install vLLM

```bash
pip install vllm==0.3.1
```

##### Start the vLLM OpenAI-compatible API server

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --dtype auto --api-key EMPTY --port 8000
```

For further details, please refer to the [[vLLM Docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)].

#### 2. Install Required Packages

```bash
pip install openai==1.12.0
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install random
```

## Experiments

You can download the datasets from [[Zhao's repository](https://github.com/tonyzhaozh/few-shot-learning/tree/main/data)]. 



The following example shows how to run an experiment on the TREC dataset at epsilon = 8:


```
./trec_eps_8.sh
```

## Acknowledgments

This code has been built upon the code accompanying the papers

"[Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation](https://arxiv.org/abs/2309.11765)" [[code](https://github.com/microsoft/dp-few-shot-generation)].

"[Calibrate Before Use: Improving Few-shot Performance of Language Models](https://arxiv.org/abs/2102.09690)" [[code](https://github.com/tonyzhaozh/few-shot-learning)].

## Questions

If anything is unclear, please contact Fengyu Gao (itsmefengyu@gmail.com).

