---
title: "Accelerating the Development of Large Multimodal Models with LMMs-Eval"
date: 2024-03-07
draft: false
language: en
featured_image: ../assets/images/pages/lmms-eval.png
summary: One command evaluation API for fast and thorough evaluation of LMMs, providing multi-faceted insights on model performance with over 40 datasets.
# author: TailBliss
# authorimage: ../assets/images/global/author.webp
categories: Blog
tags: Blog
---

## Introduction

In today's world, we're on a thrilling quest for Artificial General Intelligence (AGI), driven by a passion that reminds us of the excitement surrounding the 1960s moon landing. At the heart of this adventure are the incredible large language models (LLMs) and large multimodal models (LMMs). These models are like brilliant minds that can understand, learn, and interact with a vast array of human tasks, marking a significant leap toward our goal.

To truly understand how capable these models are, we've started to create and use a wide variety of evaluation benchmarks. These benchmarks help us map out a detailed chart of abilities, showing us how close we are to achieving true AGI. However, this journey is not without its challenges. The sheer number of benchmarks and datasets we need to look at is overwhelming. They're all over the place - tucked away in someone's Google Drive, scattered across Dropbox, and hidden in the corners of various school and research lab websites. It's like embarking on a treasure hunt where the maps are spread far and wide.

In the field of language models, there has been a valuable precedent set by the work of `lm-evaluation-harness` <d-cite key="eval-harness"></d-cite>. They offer integrated data and model interfaces, enabling rapid evaluation of language models and serving as the backend support framework for the [open-llm-leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and has gradually become the underlying ecosystem of the era of foundation models.

However, the evaluation of multi-modality models is still in its infancy, and there is no unified evaluation framework that can be used to evaluate multi-modality models across a wide range of datasets. To address this challenge, we introduce **lmms-eval**<d-cite key="lmms_eval2024"></d-cite>, an evaluation framework meticulously crafted for consistent and efficient evaluation of Large Multimoal Models (LMMs).

We humbly obsorbed the exquisite and efficient design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Building upon its foundation, we implemented our `lmms-eval` framework with performance optimizations specifically for LMMs.

## Necessity of lmms-eval

We believe our effort could provide an efficient interface for the detailed comparison of publicly available models to discern their strengths and weaknesses. It’s also useful for research institutions and production-oriented companies to accelerate the development of Large Multimoal Models. With the `lmms-eval`, we have significantly accelerated the lifecycle of model iteration. Inside the LLaVA team, the utilization of `lmms-eval` largely improves the efficiency of the model development cycle, as we are able to evaluate weekly trained hundreds of checkpoints on 20-30 datasets, identifying the strengths and weaknesses, and then make targeted improvements.

---

## Important Features

![](https://lmms-lab.github.io/blog-images/lmms-eval-1.0/teaser.webp)

<!-- {% include figure.liquid loading="eager" path="assets/img/teaser.png" class="img-fluid rounded z-depth-1" zoomable=true %} -->

<!-- <p align="center" width="100%">
<img src="assets/img/teaser.png"  width="100%" height="100%">
</p> -->

For more usage guidance, please visit our [GitHub repo](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main).

### One-command evaluation, with detailed logs and samples.

You can evaluate the models on multiple datasets with a single command. No model/data preparation is needed, just one command line, few minutes, and get the results. Not just a result number, but also the detailed logs and samples, including the model args, input question, model response, and ground truth answer.

### Accelerator support and Tasks grouping.

We support the usage of `accelerate` to wrap the model for distributed evaluation, supporting multi-gpu and tensor parallelism. With **Task Grouping**, all instances from all tasks are grouped and evaluated in parallel, which significantly improves the throughput of the evaluation.

Below are the total runtime on different datasets using 4 x A100 40G.

| Dataset (#num)          | LLaVA-v1.5-7b      | LLaVA-v1.5-13b     |
| :---------------------- | :----------------- | :----------------- |
| mme (2374)              | 2 mins 43 seconds  | 3 mins 27 seconds  |
| gqa (12578)             | 10 mins 43 seconds | 14 mins 23 seconds |
| scienceqa_img (2017)    | 1 mins 58 seconds  | 2 mins 52 seconds  |
| ai2d (3088)             | 3 mins 17 seconds  | 4 mins 12 seconds  |
| coco2017_cap_val (5000) | 14 mins 13 seconds | 19 mins 58 seconds |

### All-In-One HF dataset hubs.

We are hosting more than 40 (and increasing) datasets on [huggingface/lmms-lab](https://huggingface.co/lmms-lab), we carefully converted these datasets from original sources and included all variants, versions and splits. Now they can be directly accessed without any burden of data preprocessing. They also serve for the purpose of visualizing the data and grasping the sense of evaluation tasks distribution.

<!-- {% include figure.liquid loading="eager" path="assets/img/ai2d.png" class="img-fluid rounded z-depth-1" zoomable=true %} -->

![](https://lmms-lab.github.io/blog-images/lmms-eval-1.0/ai2d.webp)

### Detailed Logging Utilites

We provide detailed logging utilities to help you understand the evaluation process and results. The logs include the model args, generation parameters, input question, model response, and ground truth answer. You can also record every details and visualize them inside runs on Weights & Biases.

![](https://lmms-lab.github.io/blog-images/lmms-eval-1.0/wandb_table.webp)

<!-- <p align="center" width="100%">
<img src="https://i.postimg.cc/W1c1vBDJ/Wechat-IMG1993.png"  width="100%" height="80%">
</p> -->

## Model Results

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<!-- {% include figure.liquid loading="eager" path="assets/img/sheet_table.png" class="img-fluid rounded z-depth-1" zoomable=true %} -->

![](https://lmms-lab.github.io/blog-images/lmms-eval-1.0/sheet_table.webp)

<!-- <p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p> -->

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

## Supported Models/Datasets

Different models perform best at specific prompt strategies and require models developers skilled knowledge to implement. We prefer not to hastily integrate a model without a thorough understanding to it. Our focus is on more tasks, and we encourage model developers to incorporate our framework into their development process and, when appropriate, integrate their model implementations into our framework through PRs (Pull Requests). Within this strategy, we support a wide range of datasets and selective model series, including:

### Supported models

- GPT4V (API, only generation-based evaluation)
- LLaVA series (ppl-based, generation-based)
- Qwen-VL series (ppl-based, generation-based)
- Fuyu series (ppl-based, generation-based)
- InstructBLIP series (generation-based)
- MiniCPM-V series (ppl-based, generation-based)

### Supported datasets

Names inside `()` indicate the actual task name referred in the config file.

- AI2D `(ai2d)`
- ChartQA `(chartqa)`
- CMMMU `(cmmmu)`
  - CMMMU Validation `(cmmmu_val)`
  - CMMMU Test `(cmmmu_test)`
- COCO Caption `(coco_cap)`
  - COCO 2014 Caption `(coco2014_cap)`
    - COCO 2014 Caption Validation `(coco2014_cap_val)`
    - COCO 2014 Caption Test `(coco2014_cap_test)`
  - COCO 2017 Caption `(coco2017_cap)`
    - COCO 2017 Caption MiniVal `(coco2017_cap_val)`
    - COCO 2017 Caption MiniTest `(coco2017_cap_test)`
- DOCVQA `(docvqa)`
  - DOCVQA Validation `(docvqa_val)`
  - DOCVQA Test `(docvqa_test)`
- Ferret `(ferret)`
- Flickr30K `(flickr30k)`
  - Ferret Test `(ferret_test)`
- GQA `(gqa)`
- HallusionBenchmark `(hallusion_bench_image)`
- Infographic VQA `(info_vqa)`
  - Infographic VQA Validation `(info_vqa_val)`
  - Infographic VQA Test `(info_vqa_test)`
- LLaVA-Bench `(llava_bench_wild)`
- LLaVA-Bench-COCO `(llava_bench_coco)`
- MathVista `(mathvista)`
  - MathVista Validation `(mathvista_testmini)`
  - MathVista Test `(mathvista_test)`
- MMBench `(mmbench)`
  - MMBench English `(mmbench_en)`
    - MMBench English Dev `(mmbench_en_dev)`
    - MMBench English Test `(mmbench_en_test)`
  - MMBench Chinese `(mmbench_cn)`
    - MMBench Chinese Dev `(mmbench_cn_dev)`
    - MMBench Chinese Test `(mmbench_cn_test)`
- MME `(mme)`
- MMMU `(mmmu)`
  - MMMU Validation `(mmmu_val)`
  - MMMU Test `(mmmu_test)`
- MMVet `(mmvet)`
- Multi-DocVQA `(multidocvqa)`
  - Multi-DocVQA Validation `(multidocvqa_val)`
  - Multi-DocVQA Test `(multidocvqa_test)`

... and more.

## Citations

```bib
@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaicheng Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
```
