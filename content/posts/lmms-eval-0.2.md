---
title: Embracing Video Evaluations with LMMs-Eval
date: 2024-06-10
draft: false
language: en
featured_image: ../assets/images/pages/lmms-eval-video.png
summary: We introduce a video evaluation feature to lmms-eval, supporting video model evaluations with over most popular datasets.
# author: TailBliss
# authorimage: ../assets/images/global/author.webp
categories: Blog
tags: video models
---

<p style="font-size: 16px; line-height: 1.4;">
  <span style="color: gray; font-size: 18px;">
    <a href="https://kairuihu.github.io/">Kairui Hu<sup>*</sup></a>, &nbsp;
    <a href="https://pufanyi.github.io/">Fanyi Pu<sup>*</sup></a>, &nbsp;
    <a href="https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg">Kaichen Zhang<sup>*</sup></a>, &nbsp;
    <a href="https://github.com/choiszt">Shuai Liu<sup>*</sup></a>, &nbsp;
    <a href="https://zhangyuanhan-ai.github.io/">Yuanhan Zhang<sup>*</sup></a> &nbsp;
    <br>
    <a href="https://brianboli.com/">Bo Li<sup>*;&dagger;</sup></a>, &nbsp;
    <a href="https://liuziwei7.github.io/">Ziwei Liu</a>
  </span>
</p>
<p style="font-size: 16px; line-height: 1.2;">
Nanyang Technical University, Singapore
<br>
<br>
<sup>*</sup>indicates equal contribution.
<sup>&dagger;</sup>development lead.
</p>

## Table of Contents

<ul style="color: gray; font-size: 14px">
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#video-evaluation-in-lmms">Video Evaluation in LMMs</a>
    <ul>
      <li><a href="#frame-extraction-for-evaluation">Frame Extraction for Evaluation</a></li>
      <li><a href="#challenges-with-audio-integration">Challenges with Audio Integration</a></li>
      <li><a href="#meta-information-for-video-datasets">Meta Information for Video Datasets</a></li>
      <li><a href="#alignment-check-for-video-datasets">Alignment Check for Video Datasets</a></li>
    </ul>
  </li>
  <li><a href="#more-details-and-feature-updates-with-v020">More Details and Feature Updates with <code>v0.2.0</code></a>
    <ul>
      <li><a href="#improved-pipeline-for-video-evaluations">Improved Pipeline for Video Evaluations</a></li>
      <li><a href="#improved-overall-evaluation-pipeline">Improved Overall Evaluation Pipeline</a></li>
      <li><a href="#supported-video-tasks">Supported Video Tasks</a></li>
      <li><a href="#supported-video-models">Supported Video Models</a></li>
    </ul>
  </li>
</ul>

## Introduction

In the journey towards multimodal intelligence, the development of LMMs has progressed remarkably, transitioning from handling static images to processing complex video inputs. This evolution is crucial, enabling models to understand and interpret dynamic scenes with temporal dependencies, motion dynamics, and contextual continuity. The importance of video evaluation is also increasing across various applications. However, there has been a noticeable absence of comprehensive benchmarks to evaluate the diverse array of video tasks. The introduction of `lmms-eval/v0.2.0` is both necessary and significant as it addresses this critical gap in the evaluation of video-based LMMs.

Building upon the success of `lmms-eval/v0.1.0`, `lmms-eval/v0.2.0` makes major upgrades on incorporating video tasks and models, and more feature updates on improved pipelines for both image and video tasks, more image models, and fixed previous community issues.

## Video Evaluation in LMMs

### Frame Extraction for Evaluation

In our framework, video evaluation can be viewed as extracting multiple frames to represent a video and then feeding the multi-frame input to the model for inference. This perspective allows us to enhance the zero-shot video understanding capability of image models by utilizing their potential to process sequences of frames as visual tokens. When each frame is considered as part of a concatenated sequence, image-only-trained models like LLaVA-Next can achieve impressive performance on video-related tasks. This highlights the significant advancement in the zero-shot video understanding capability of image models, representing a notable step forward for LMMs.

When defining these models, we also specify the number of frames to be extracted. Extracting more frames typically enhances the model's understanding of the entire video.

Besides, some models like Gemini and Reka do not expose the video processing interface. Our interface accommodates this by directly uploading the raw data files for evaluation.

### Challenges with Audio Integration

One of the most concerning issues with these evaluations is the lack of focus on audio inputs. The majority of current evaluations overlook audio, which is a significant drawback. Audio plays a pivotal role in video content, offering supplementary context and information. At present, only the WorldQA dataset explicitly necessitates audio information to answer questions accurately. This underscores a critical gap in the evaluation process that future frameworks must address to ensure a more comprehensive evaluation of video understanding.

### **Meta Information for Video Datasets**

Table 1: Video Dataset Meta Information

<table style="white-space: nowrap; display: flex; justify-content: center; align-items: center;">
  <tr class="bg-white-100">
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Dataset</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Split</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Task Name</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Task Format</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Evaluation Metric</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Video Source</th>
    <th class="bg-blue-100 border text-left" style="padding: 16px 0.25em">Average Length</th>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">ActivityNet-QA</td>
    <td class="border" style="padding: 16px 0.25em">Test</td>
    <td class="border" style="padding: 16px 0.25em">activitynetqa</td>
    <td class="border" style="padding: 16px 0.25em">Open-ended</td>
    <td class="border" style="padding: 16px 0.25em">GPT-Eval</td>
    <td class="border" style="padding: 16px 0.25em">Internet</td>
    <td class="border" style="padding: 16px 0.25em">117.3s</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">EgoSchema</td>
    <td class="border" style="padding: 16px 0.25em">Full</td>
    <td class="border" style="padding: 16px 0.25em">egoschema</td>
    <td class="border" style="padding: 16px 0.25em">MCQ</td>
    <td class="border" style="padding: 16px 0.25em">Submission</td>
    <td class="border" style="padding: 16px 0.25em">Ego4D</td>
    <td class="border" style="padding: 16px 0.25em">180s</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">YouCook2</td>
    <td class="border" style="padding: 16px 0.25em">Validation</td>
    <td class="border" style="padding: 16px 0.25em">youcook2_val</td>
    <td class="border" style="padding: 16px 0.25em">MCQ</td>
    <td class="border" style="padding: 16px 0.25em">Bleu; METEOR; ROUGE_L; CIDEr</td>
    <td class="border" style="padding: 16px 0.25em">YouTube</td>
    <td class="border" style="padding: 16px 0.25em">311.6s</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">Vatex</td>
    <td class="border" style="padding: 16px 0.25em">Test</td>
    <td class="border" style="padding: 16px 0.25em">vatex_test</td>
    <td class="border" style="padding: 16px 0.25em">Caption Matching</td>
    <td class="border" style="padding: 16px 0.25em">Bleu; METEOR; ROUGE_L; CIDEr</td>
    <td class="border" style="padding: 16px 0.25em">YouTube</td>
    <td class="border" style="padding: 16px 0.25em">147.6s</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">Vatex-ZH</td>
    <td class="border" style="padding: 16px 0.25em">Validation</td>
    <td class="border" style="padding: 16px 0.25em">vatex_val_zh</td>
    <td class="border" style="padding: 16px 0.25em">    Caption Matching</td>
    <td class="border" style="padding: 16px 0.25em">Bleu; METEOR; ROUGE_L; CIDEr</td>
    <td class="border" style="padding: 16px 0.25em">YouTube</td>
    <td class="border" style="padding: 16px 0.25em">165s</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border" style="padding: 16px 0.25em">VideoChatGPT</td>
    <td class="border" style="padding: 16px 0.25em">Test</td>
    <td class="border" style="padding: 16px 0.25em">videochatgpt</td>
    <td class="border" style="padding: 16px 0.25em">Open-ended</td>
    <td class="border" style="padding: 16px 0.25em">GPT_Eval</td>
    <td class="border" style="padding: 16px 0.25em">ActivityNet-200</td>
    <td class="border" style="padding: 16px 0.25em">108s</td>
  </tr>
    <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">VideoDetailCaptions</td>
        <td class="border" style="padding: 16px 0.25em">Test</td>
        <td class="border" style="padding: 16px 0.25em">video_dc499</td>
        <td class="border" style="padding: 16px 0.25em">Open-ended</td>
        <td class="border" style="padding: 16px 0.25em">GPT_Eval</td>
        <td class="border" style="padding: 16px 0.25em">ActivityNet-200</td>
        <td class="border" style="padding: 16px 0.25em">108s</td>
    </tr>
    <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">NextQA</td>
        <td class="border" style="padding: 16px 0.25em">OE (Text / Validation), MC (Test)</td>
        <td class="border" style="padding: 16px 0.25em">nextqa</td>
        <td class="border" style="padding: 16px 0.25em">MCQ / Open-ended</td>
        <td class="border" style="padding: 16px 0.25em">MC: Exact Match; OE: WUPS</td>
        <td class="border" style="padding: 16px 0.25em">YFCC-100M</td>
        <td class="border" style="padding: 16px 0.25em">44s</td>
    </tr>
    <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">CVRR-ES</td>
        <td class="border" style="padding: 16px 0.25em">Default</td>
        <td class="border" style="padding: 16px 0.25em">cvrr</td>
        <td class="border" style="padding: 16px 0.25em">Open-ended</td>
        <td class="border" style="padding: 16px 0.25em">GPT_Eval</td>
        <td class="border" style="padding: 16px 0.25em">Internet; Public dataset</td>
        <td class="border" style="padding: 16px 0.25em">22.3s</td>
    </tr>
        <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">Perception Test</td>
        <td class="border" style="padding: 16px 0.25em">MC</td>
        <td class="border" style="padding: 16px 0.25em">perceptiontest_val_mc</td>
        <td class="border" style="padding: 16px 0.25em">MCQ</td>
        <td class="border" style="padding: 16px 0.25em">Accuracy</td>
        <td class="border" style="padding: 16px 0.25em">Internet</td>
        <td class="border" style="padding: 16px 0.25em">23s</td>
    </tr>
    <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">TempCompass</td>
        <td class="border" style="padding: 16px 0.25em">Default</td>
        <td class="border" style="padding: 16px 0.25em">tempcompass</td>
        <td class="border" style="padding: 16px 0.25em">MCQ; Y/N; Captioning; Caption Matching</td>
        <td class="border" style="padding: 16px 0.25em">Accuracy</td>
        <td class="border" style="padding: 16px 0.25em">Internet</td>
        <td class="border" style="padding: 16px 0.25em">11.9s</td>
    </tr>
    <tr class="hover:bg-gray-50">
        <td class="border" style="padding: 16px 0.25em">Video-MME</td>
        <td class="border" style="padding: 16px 0.25em">Test</td>
        <td class="border" style="padding: 16px 0.25em">videomme</td>
        <td class="border" style="padding: 16px 0.25em">MCQ</td>
        <td class="border" style="padding: 16px 0.25em">Accuracy</td>
        <td class="border" style="padding: 16px 0.25em">YouTube</td>
        <td class="border" style="padding: 16px 0.25em">1017s</td>
    </tr>
    <!-- A,B,C,D QA	Accuracy	YouTube	1017.0s -->
</table>

### Alignment Check for Video Datasets

Table 2. Alignment Check for Video Datasets
<table style="white-space: nowrap; display: flex; justify-content: center; align-items: center;">
  <tr class="bg-white-100">
    <th class="bg-blue-100 border text-left px-8 py-4">Dataset</th>
    <th class="bg-blue-100 border text-left px-8 py-4">Subset</th>
    <th class="bg-blue-100 border text-left px-8 py-4">Model</th>
    <th class="bg-blue-100 border text-left px-8 py-4">Original Reported</th>
    <th class="bg-blue-100 border text-left px-8 py-4">LMMs-Eval</th>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">EgoSchema(0-shot)</td>
    <td class="border px-8 py-4">egoschema_subset_mc_ppl</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">-</td>
    <td class="border px-8 py-4">50.60%</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">CVRR-ES</td>
    <td class="border px-8 py-4">cvrr_multiple_actions_in_a_single_video</td>
    <td class="border px-8 py-4">Video-ChatGPT</td>
    <td class="border px-8 py-4">27.67%</td>
    <td class="border px-8 py-4">28.31%</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">CVRR-ES</td>
    <td class="border px-8 py-4">cvrr</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">-</td>
    <td class="border px-8 py-4">44.29%</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">TempCompass</td>
    <td class="border px-8 py-4">tempcompass_caption_matching</td>
    <td class="border px-8 py-4">LLaVA-1.5-13B</td>
    <td class="border px-8 py-4">59.50%</td>
    <td class="border px-8 py-4">59.35%</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">VideoChatGPT</td>
    <td class="border px-8 py-4">videochatgpt_temporal</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">Score: 2.60 / 5</td>
    <td class="border px-8 py-4">Score: 2.67 / 5</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">NextQA</td>
    <td class="border px-8 py-4">nextqa_oe_test</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">26.90%</td>
    <td class="border px-8 py-4">26.61%</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">VATEX</td>
    <td class="border px-8 py-4">vatex_test</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">-</td>
    <td class="border px-8 py-4">CIDEr: 39.28</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">ActivityNetQA</td>
    <td class="border px-8 py-4">activitynetqa</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">53.5% </td>
    <td class="border px-8 py-4">52.72% </td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">VideoDetailCaptions</td>
    <td class="border px-8 py-4">video_dc499</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">Score: 3.32 / 5</td>
    <td class="border px-8 py-4">Score: 3.50 / 5</td>
  </tr>
  <tr class="hover:bg-gray-50">
    <td class="border px-8 py-4">Video-MME (wo/subs)</td>
    <td class="border px-8 py-4">videomme</td>
    <td class="border px-8 py-4">LLaVA-NeXT-Video-7B</td>
    <td class="border px-8 py-4">-</td>
    <td class="border px-8 py-4">41.98%</td>
  </tr>
</table>

## More Details and Feature Updates with `v0.2.0`

### **Improved Pipeline for Video Evaluations**

Here’s a breakdown of adding video datasets support, especially on how we implement the process from video caching, loading and feed to model to get response.

1.  **Download and Load Videos:** Video are being loaded during generation phase. We will host different video datasets on the huggingface and preprocess the video path for you in your huggingface cache folder. It is recommended to set `HF_HOME` before you use our evaluation suite so that you can manage the download place. After downloading the videos from huggingface hub, we unzip them into a local cache dir, where by default is `HF_HOME`.

    - The code specifically demonstrates the logic of how we handle video datasets in lmms-eval.

        ```python
            @retry(stop=(stop_after_attempt(5) | stop_after_delay(60)), wait=wait_fixed(2))
            def download(self, dataset_kwargs=None) -> None:
                # If the dataset is a video dataset,
                # Recursively search whether their is a zip and unzip it to the huggingface home
                if dataset_kwargs is not None and "video" in dataset_kwargs and dataset_kwargs["video"]:
                    hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
                    cache_dir = dataset_kwargs["cache_dir"]
                    cache_dir = os.path.join(hf_home, cache_dir)
                    accelerator = Accelerator()
                    if accelerator.is_main_process:
                        cache_path = snapshot_download(repo_id=self.DATASET_PATH, repo_type="dataset")
                        zip_files = glob(os.path.join(cache_path, "**/*.zip"), recursive=True)

                        if not os.path.exists(cache_dir) and len(zip_files) > 0:
                            for zip_file in zip_files:
                                eval_logger.info(f"Unzipping {zip_file} to {cache_dir}")
                                shutil.unpack_archive(zip_file, cache_dir)

                    accelerator.wait_for_everyone()

                    if "builder_script" in dataset_kwargs:
                        builder_script = dataset_kwargs["builder_script"]
                        self.DATASET_PATH = os.path.join(cache_path, builder_script)
                        dataset_kwargs.pop("builder_script")

                    dataset_kwargs.pop("cache_dir")
                    dataset_kwargs.pop("video")
        ```

2.  **Format questions:** For each task, questions are formatted in the `<taskname>/utils.py` file. We parse each document from the Huggingface dataset, retrieve the questions, and formulate the input with any specified model-specific prompts.

    - The code specifically demonstrates the logic of how to implement question format.

        ```python
        # This is the place where you format your question
        def perceptiontest_doc_to_text(doc, model_specific_prompt_kwargs=None):
            if model_specific_prompt_kwargs is None:
                model_specific_prompt_kwargs = {}
            pre_prompt = ""
            post_prompt = ""
            if "pre_prompt" in model_specific_prompt_kwargs:
                pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            if "post_prompt" in model_specific_prompt_kwargs:
                post_prompt = model_specific_prompt_kwargs["post_prompt"]

            question = doc["question"]
            if "options" in doc:
                index = 0
                for op in doc["options"]:
                    if index == 0:
                        question += "\n" + "A. " + op
                    elif index == 1:
                        question += "\n" + "B. " + op
                    else:
                        question += "\n" + "C. " + op
                    index += 1
                post_prompt = "\nAnswer with the option's letter from the given choices directly."
            print("question\n")
            print(question)

            return f"{pre_prompt}{question}{post_prompt}"
        ```

3.  **Process results:** After the model generates results, each result is parsed and evaluated based on the corresponding evaluation metric. The choice of metric is based on the dataset’s official implementation on their official project website. We primarily use three types of metrics:

    **a. Accuracy:**
    For datasets with ground truth answers, we generate a score by comparing the model’s results with the ground truth. This metric is commonly used in multiple-choice QA tasks such as PerceptionTest-val and EgoSchema-subset.

    **b. GPT Evaluation:**
    For open-ended answers generated by the model, we apply OpenAI GPT API to evaluate the responses. This metric is often used in generation tasks like ActivityNetQA and VideoChatGPT.

    **c. Submission:**
    If the dataset does not provide ground truth answers and requires submission of inference results to a server for evaluation, we provide a submission file according to the dataset's official template. This metric is used in tasks like EgoSchema, Perception Test.

4.  **Aggregate results:**
    After evaluating each data instance, we aggregate the individual results to generate the overall evaluation metrics. Finally, we provide a summary table that consolidates all the evaluation results, similar to the one in Google’s Gemini report.
5.  **Grouped Tasks:**
    For tasks with multiple subsets, we group all subset tasks together. For example, the VideoChatGPT dataset includes three subsets: generic, temporal, and consistency. By running `--task videochatgpt`, all three subsets can be evaluated together, eliminating the need to specify each subset individually. We summarize all the grouped task names in Table 1. This pipeline ensures a thorough and standardized evaluation process for video LMMs, facilitating consistent and reliable performance assessment across various tasks and datasets. - This code denotes how we organize the group of tasks together.
    ```yaml
    group: videochatgpt
    task:
    - videochatgpt_gen
    - videochatgpt_temporal
    - videochatgpt_consistency
    ```


### **Improved Overall Evaluation Pipeline**

 1. For newly added tasks with different splits and metrics, we have adopted a naming rule in the format `{name}_{split}_{metric}`. For instance, `perceptiontest_val_mcppl` refers to the validation split of the PerceptionTest evaluation dataset, using multiple choice perplexity as the evaluation metric.
   
 2. We support `llava-next` series models with sglang. You can use the following command to launch evaluation with sglang support, that’s much more efficient when running on `llava-next-72b/110b`

    ```bash
    python3 -m lmms_eval \
    	--model=llava_sglang \
    	--model_args=pretrained=lmms-lab/llava-next-72b,tokenizer=lmms-lab/llavanext-qwen-tokenizer,conv_template=chatml-llava,tp_size=8,parallel=8 \
    	--tasks=mme \
    	--batch_size=1 \
    	--log_samples \
    	--log_samples_suffix=llava_qwen \
    	--output_path=./logs/ \
    	--verbosity=INFO
    ```

 3. We add a `force_download` mode to robustly handle the case that videos are not fully cached in local folder. You could add the args to task yaml file as the following commands. To support the evaluations that in machines that do not have access to internet, we add the `local_files_only` to support this feature.

    ```yaml
    dataset_path: lmms-lab/ActivityNetQA
    dataset_kwargs:
      token: True
      video: True
      force_download: False
      local_files_only: False
      cache_dir: activitynetqa
    model_specific_prompt_kwargs:
      default:
        pre_prompt: ""
        post_prompt: " Answer the question using a single word or phrase."

    metadata:
      version: 0.0
      gpt_eval_model_name: gpt-3.5-turbo-0613
    ```

 4. We found that sometimes the dataset downloading process will throw `Retry` or `HTTP Timedout` errors. To prevent this, we recommend disabling `hf_transfer` mechanism by setting this in your environment `export HF_HUB_ENABLE_HF_TRANSFER="0"`
   
 5. mmmu_group_img_val

    We aligned the results of LLaVA-NeXT 34B with previously reported values. In our previous evaluation, for the questions with multiple images, we concatenated them into one. When tested separately (`mmmu_val`), the score was 46.7, and after we do the concatenation operation (in lmms-eval, you could switch to use `tasks=mmmu_group_img_val`), the score was 50.1 for LLaVA-NeXT 34B.

    <p align="center">
    <a href="https://postimg.cc/2brN0TMk">
        <img src="https://i.postimg.cc/JnB742Qk/mmmu-group.png" alt="mmmu-group.png">
        </a>
    </p>

    ```bash
    A collimated beam containing two different frequencies of light travels through vacuum and is incident on a piece of glass. Which of the schematics below depicts the phenomenon of dispersion within the glass in a qualitative correct manner? Select (e) if none of the options are qualitatively correct.
    (A) <image 1>
    (B) <image 2>
    (C) <image 3>
    (D) <image 4>

    Answer with the option's letter from the given choices directly.
    ```


  6. **Predict Only Mode - Only Inference, No Evaluation**

        In some cases, you may want to obtain only the inference results, without triggering the evaluation process. For this purpose, we have integrate the **`predict_only`** mode from the original `lmms-eval`. This feature allows you to obtain model inference results without performing further evaluation. **It is particularly useful when you do not need to evaluate your model results, for instance, if the dataset requires ChatGPT-based evaluation but you do not want to use the OpenAI API.**

        To use the **`predict_only`** mode, add the **`--predict_only`** flag to your command. This will override the original evaluation process with a bypass function after obtaining the model inference results and simply save the results as logs. 

  7. **From-Logs Mode - Evaluation Based on Log Files from `predict_only` mode**

        ```bash
        accelerate launch -m lmms_eval \
        --model=from_log \
        --tasks=<taskname> \ 
        --batch_size=1 \
        --log_samples \
        --output_path=./logs/ \
        --model_args=logs=<path_to_log_directory>,model_name=<model_name>,model_args \
        --verbosity=DEBUG
        ```

        In some cases, you may want to evaluate model performance using pre-existing inference results. For this purpose, we have designed the **`from_log`** mode. This feature allows you to evaluate model performance directly from inference results recorded in the `logs/` directory. This mode saves time and enhances portability, consistency, and reproducibility. **It is particularly useful when you already have model inference results and wish to avoid running the inference process again.** 

        Currently, we only support inference results stored in the `logs/` directory, which is the log file generated from our pipeline. Hence currently, the file template of these results is pre-defined. If you need to evaluate the inference result generated by yourself, you may have to convert your file into the same template as those under `logs/`.

        To use the **`from_log`** mode for performance evaluation based on existing log files, you can run the following command. You can specify the `task_name`, `path_to_log_directory`, `model_name`and  `model_args`(if specified). Our framework will traverse through all the log files within the specified directory and find the most recent log file for evaluation.

   8. **Combined Use of Two Modes**
   
        You can use the two modes together: using the **`predict_only`** mode to obtain model inference results, and using the **`from_log`** mode to evaluate the generated inference results. This enhances the overall consistency and reproducibility of our framework.

        You can use the following command to run the two modes together:

        ```bash
        # Open predict_only mode to inference
        accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
            --model <model_name> \
            --tasks $TASK \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix <model_name> \
            --output_path ./logs/ \
            --predict_only

        # Open from_log mode to evaluate
        accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
            --model from_log \
            --model_args model_name=<model_name>\
            --tasks $TASKS \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix <model_name> \
            --output_path ./logs/
        ```

### **Newly Supported Video Tasks**
  1. <a href="https://github.com/MILVLG/activitynet-qa">ActivityNet-QA</a>
  2. <a href="https://github.com/egoschema/EgoSchema/">EgoSchema</a>
  3. <a href="http://youcook2.eecs.umich.edu/">YouCook2</a>
  4. <a href="https://eric-xw.github.io/vatex-website/index.html">VATEX</a>
  5. <a href="https://eric-xw.github.io/vatex-website/index.html">VATEX-ZH</a>
  6. <a href="https://github.com/mbzuai-oryx/Video-ChatGPT/">VideoChatGPT</a>
  7. <a href="https://github.com/EvolvingLMMs-Lab/lmms-eval-internal/blob/internal_main_dev/lmms_eval/tasks/video_detail_description/README.md">VideoDetailCaptions</a>
  8. <a href="https://github.com/doc-doc/NExT-QA">NextQA</a>
  9. <a href="https://github.com/mbzuai-oryx/CVRR-Evaluation-Suite/">CVRR-ES</a>
  10. <a href="https://github.com/google-deepmind/perception_test">Perception Test</a>
  11. <a href="https://github.com/llyx97/TempCompass">TempCompass</a>
  12. <a href="https://github.com/BradyFU/Video-MME">Video-MME</a>
  
### **Newly Supported Video Models**

We have supported more video models that can be used in LMMs-Eval. We now support evaluating video datasets using a one line command.

1. [LLaVA-NeXT-Video](https://huggingface.co/collections/lmms-lab/llava-next-video-661e86f5e8dabc3ff793c944)
2. [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
3. [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
4. [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
5. [MPLUG-OWL](https://github.com/X-PLUG/mPLUG-Owl)

### **Community Support**

During this period, we received the following Pull Requests (PRs):

> Details are in [lmms-eval/v0.2.0 release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/untagged-9057ff0e9a72d5a5846f)

**Datasets:**

- VCR: Vision_Caption_Restoration (officially from the authors, MILA)
- ConBench (officially from the authors, PKU/Bytedance)
- MathVerse (officially from the authors, CUHK)
- MM-UPD (officially from the authors, University of Tokyo)
- Multi-lingual MMMU (officially from the authors, CUHK)
- WebSRC (from Hunter Heiden)
- ScreeSpot (from Hunter Heiden)
- RealworldQA (from Fanyi Pu, NTU)
- Multi-lingual LLaVA-W (from Gagan Bhatia, UBC)

**Models:**

- LLaVA-HF (officially from Huggingface)
- Idefics-2 (from the lmms-lab team)
- microsoft/Phi-3-Vision (officially from the authors, Microsoft)
- LLaVA-SGlang (from the lmms-lab team)
