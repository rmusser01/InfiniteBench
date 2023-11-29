<div align="center">

# InfiniteBench: 100k+ Long-Context Benchmark for Large Language Models

<p align="center">
  <a href="./README_ZH.md">中文</a> •
  <a href="./README.md">English</a> •
  <a href="">(Paper Upcoming)</a>
</p>

</div>

## Introduction

Welcome to InfiniteBench, a cutting-edge benchmark tailored for evaluating the capabilities of language models to process, understand, and reason over super long contexts (100k+ tokens). Long contexts are crucial for enhancing applications with LLMs and achieving high-level interaction. InfiniteBench is designed to push the boundaries of language models by testing them against a context length of 100k+, which is 10 times longer than traditional datasets.

## Features

- **Loooong Context:** InfiniteBench is a pioneer in testing language models with a context length of 100k+, offering an unparalleled challenge in the field.
- **Diverse Domain:** The benchmark comprises 12 unique tasks, each crafted to assess different aspects of language processing and comprehension in extended contexts.
- **Specialized Test:** InfiniteBench consists of tasks that state-of-the-art LLMs are known to be capable of when using shorter context. This ensures that the performance degradation is only caused by the length of the contexts.
- **Real-World and Synthetic Scenarios:** The tasks are a mix of real-world scenarios and synthetic constructs, ensuring a comprehensive evaluation of models. Real-world scenarios make the test pragmatic, and synthetic ones leave the space for extending the context length further with ease.

## Task Composition

<div align="center">
  <img src="figs/data_pie.png" width="480px">
</div>

| Task Name            | Context       | # Examples | Avg Input Tokens | Avg Output Tokens | Description                                                                     |
| -------------------- | ------------- | ---------- | ---------------- | ----------------- | ------------------------------------------------------------------------------- |
| En.Sum               | Fake Book     | 148        | 131.7k           | 1.1k              | Summarization of a fake book created with core entity substitution.              |
| En.QA                | Fake Book     | 466        | 155.3k           | 4.9               | Free-form question answering based on the fake book.                            |
| En.MC                | Fake Book     | 322        | 142.4k           | 5.3               | Multiple choice questions derived from the fake book.                           |
| En.Dia               | Script        | 200        | 103.6k           | 3.4               | Identification of talkers in partially anonymized scripts.                      |
| Zh.QA                | New Book      | 189        | 1918.1k          | 6.2               | Question answering on a set of newly collected books.                           |
| Code.Debug           | Code Document | 394        | 114.7k           | 4.8               | Finding which function in a code repo contains an crashing error (in multiple choice form). |
| Code.Run             | Synthetic     | 400        | 75.2k            | 1.3               | Simulating execution of multiple simple, synthetic functions.                     |
| Math.Calc            | Synthetic     | 50         | 43.9k            | 43.9k             | Calculations involving super-long arithmetic equations.                         |
| Math.Find            | Synthetic     | 350        | 87.9k            | 1.3               | Finding special integers in a lengthy list.                                     |
| Retrieve.PassKey[^1] | Synthetic     | 590        | 122.4k           | 2.0               | Retrieving hidden keys in a noisy long context.                                 |
| Retrieve.Number      | Synthetic     | 590        | 122.4k           | 4.0               | Locating repeated hidden numbers in a noisy long context.                       |
| Retrieve.KV[^2]      | Synthetic     | 500        | 89.9k            | 22.7              | Finding the corresponding value from a dictionary and a key.                    |

## How to Download Data

Click here to download data from 🤗 Huggingface directly: <https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench>

### Using 🤗 Datasets

Alternatively, you can download using the 🤗 Datasets library as follows.

```python
from datasets import load_dataset
dataset = load_dataset("xinrongzhang2022/InfiniteBench")
```
### Using scripts
```shell
cd InfiniteBench
bash scripts/download_dataset.sh
```
And the data can be found under data/InfiniteBench/
## Evaluation Result

We evaluate SOTA proprietary and open-source LLMs, the result is as follows.

| Task Name        | GPT-4       | Yarn-Mistral-7B | Kimi-Chat | Claude 2    | RWKV-4-World-7B |
| ---------------- | ----------- | --------------- | --------- | ----------- | --------------- |
| Retrieve.PassKey | 100%        | 92.71%          | 98.14%    | coming      | < 5%            |
| Retrieve.Number  | 100%        | 56.61%          | 95.42%    | 67.12%      | -               |
| Retrieve.KV      | 89.00%      | < 5%            | 40.40%    | 67.00%      | -               |
| En.Sum           | 8.76%       | < 5%            | 18.02%    | 15.13%     | -               |
| En.QA            | 21.45%      | 10.22%          | 14.98%    | coming      | -               |
| En.MC            | 69.88%      | 26.71%          | 71.74%    | coming      | -               |
| En.Dia           | 8.50%       | 7.50%           | 11.50%    | 46.50%     | -               |
| Zh.QA            | 24.80%      | 14.25%          | 17.63%    | coming      | -               |
| Code.Debug       | 39.59%      | < 5%            | 18.02%    | < 5%        | -               |
| Code.Run         | 23.25%      | < 5%            | < 5%      | < 5%        | -               |
| Math.Calc        | < 5%        | < 5%            | < 5%      | < 5%        | -               |
| Math.Find        | 60.00%      | 17.14%          | 12.57%    | 32.29%      | -               |

Note:

1. The evaluation code for YaRN-Mistral-7B and RWKV-4-World-7B is implemented by ourselves, and please contact us or submit an issue if there are any problems.
2. Kimi-Chat, Claude 2, and GPT-4 are evaluated using the official API with default configuration.
3. For Math.Calc, the values in the parentheses have a measurement unit of 0.01%. This is because it is easy to get a very low score on this task.
4. The metric for task Math.Find, Math.Calc, Code.Run, Code.Debug, En.Dia, En.MC, Retrieve.KV, Retrieve.Number, and Retrieve.PassKey is accuracy; 

   The metric for task Zh.QA and En.QA are ROUGE F1 score; 
   
   The metric for En.Sum is the `rougeLsum` score from the 🤗 Evaluate library.

> Since RWKV-World-4-7B got none of the examples in Retrieve.PassKey correct, we did not consider testing it on other tasks, which are more challenging for other models. We emphasize that RWKV-World-4-7b was apparently never trained on 128k, and was not advertised as being capable of contexts of such lengths, and this result is therefore not a sign that the RWKV architecture is incapable of extending to 128k context lengths. Moreover, there is a chance that our implementation is faulty, we greatly appreciate notifying us of any issues.

<div align="center">
  <img src="figs/radar_res.png" width="480px">
</div>

## Installation

```shell
pip install -r requirements.txt
```

## How to Run

Download the dataset the `data` folder (or set the `--data_dir` argument to the location of the dataset). The data folder structure should be:

```
├── InfiniteBench
│   ├── code_debug.jsonl
│   ├── code_run.jsonl
│   ├── kv_retrieval.jsonl
│   ├── longbook_choice_eng.jsonl
│   ├── longbook_qa_chn.jsonl
│   ├── longbook_qa_eng.jsonl
│   ├── longbook_sum_eng.jsonl
│   ├── longdialogue_qa_eng.jsonl
│   ├── math_calc.jsonl
│   ├── math_find.jsonl
│   ├── number_string.jsonl
│   └── passkey.jsonl
└── construct_synthetic_dataset.py
```

Then, in the `src` folder, execute:

```shell
python eval_yarn_mistral.py --task kv_retrieval
python eval_gpt4.py --task longbook_sum_qa
python eval_rwkv.py --task passkey
```

## Contributions

## Citation

> This will be updated when our preprint paper is released.

```bibtex
@misc{zhang2023infinitebench,
  title  = {InfiniteBench: 128k Long-Context Benchmark for Language Models},
  author = {Zhang, Xinrong and Chen, Yingfa and Hu, Shengding and Wu, Qihao and Chen, Junhao and Xu, Zihang and Dai, Zhenning and Han, Xu and Wang, Shuo and Liu, Zhiyuan and Sun, Maosong},
  year   = {2023}
}
```

## Acknowledgement

Thanks to Cong Feng, Zhongwu Zhai, Guoyang Zeng, Chenyang Song, Renjie Luo, Chaoqun He, Yuge Tu, Bowen Ping, Yujie Huang, Yudong Mei, Kaihuo Zhang, Weilin Zhao, Ao Sun, Yulin Chen, Ganqu Cui.

## References
[^1]: Mohtashami, Amirkeivan and Martin Jaggi. "Landmark Attention: Random-Access Infinite Context Length for Transformers." ArXiv abs/2305.16300 (2023): n. pag.
[^2]: Liu, Nelson F. et al. "Lost in the Middle: How Language Models Use Long Contexts." ArXiv abs/2307.03172 (2023): n. pag.
