---
title: "Long Context Transfer from Language to Vision"
date: 2024-06-20
draft: false
language: en
featured_image: ../assets/images/pages/heatmap.png
summary: Our paper explores the long context transfer phenomenon and validates this property on both image and video benchmarks. We propose the Long Video Assistant (LongVA) model, which can process up to 2000 frames or over 2000K visual tokens without additional complexities.
description: Our paper explores the long context transfer phenomenon and validates this property on both image and video benchmarks. We propose the Long Video Assistant (LongVA) model, which can process up to 2000 frames or over 2000K visual tokens without additional complexities.
categories: Blog
tags: video models

---
<p style="font-size: 16px; line-height: 1.8;">
  <span style="color: gray; font-size: 18px;">
    <a href="https://veiled-texture-20c.notion.site/Peiyuan-Zhang-ab24b48621c9491db767a76df860873a">Peiyuan Zhang<sup>*;&dagger;1;2</sup></a>, &nbsp;
    <a href="https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg">Kaichen Zhang<sup>*;1;2</sup></a>, &nbsp;
    <a href="https://brianboli.com/">Bo Li<sup>*;1;2</sup></a>, &nbsp;
    <a href="https://openreview.net/profile?id=~Guangtao_Zeng1">Guangtao Zeng<sup>3</sup></a>, &nbsp;
    <a href="https://jingkang50.github.io/">Jingkang Yang<sup>1;2</sup></a>, &nbsp;
    <a href="https://zhangyuanhan-ai.github.io/">Yuanhan Zhang<sup>1;2</sup></a>, &nbsp;
    <a href="https://openreview.net/profile?id=~Ziyue_Wang5">Ziyue Wang<sup>2</sup></a>, &nbsp;
    <a href="https://www.ntu.edu.sg/s-lab">Haoran Tan<sup>2</sup></a>, &nbsp;
    <a href="https://chunyuan.li/">Chunyuan Li<sup>1</sup></a>, &nbsp;
    <a href="https://liuziwei7.github.io/">Ziwei Liu<sup>1;2</sup></a>, &nbsp;
  </span>
</p>
<p style="font-size: 16px; line-height: 1.2;">
<sup>1</sup>LMMs-Lab Team &nbsp; <sup>2</sup>NTU, Singapore &nbsp; <sup>3</sup>SUTD, Singapore
<br>
<br>
<sup>*</sup>equal contribution.
<sup>&dagger;</sup>project lead.
</p>

# Table of Contents
<ul style="line-height: 1.2;">
  <li><a href="#abstract">Abstract</a></li>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#long-video-assistant">Long Video Assistant</a>
    <ul>
      <li><a href="#training-long-language-model">Training Long Language Model</a></li>
      <li><a href="#aligning-long-language-model-using-short-vision-data">Aligning Long Language Model Using Short Vision Data</a></li>
    </ul>
  </li>
  <li><a href="#experiments">Experiments</a>
    <ul>
      <li><a href="#v-niah-results">V-NIAH Evaluations</a></li>
      <li><a href="#video-evaluations">Video Evaluations</a></li>
      <li><a href="#image-evaluations">Image Evaluations</a></li>
    </ul>
  </li>
  <li><a href="#conclusion">Conclusion</a></li>
</ul>

# Abstract

Video sequences offer valuable temporal information, but existing large multimodal models (LMMs) fall short in understanding extremely long videos. Many works address this by reducing the number of visual tokens using visual resamplers. Alternatively, in this paper, we approach this problem from the perspective of the language model. By simply extrapolating the context length of the language model with text data alone, we enable LMMs to comprehend up to 200K visual tokens without training on any video data. 

Our paper makes the following contributions:
**a)** We explore the \textit{long context transfer} phenomenon and validate this property on both image and video benchmarks.
**b)** To effectively measure the LMM’s ability to generalize to long contexts in the vision modality, we develop V-NIAH (Visual Needle-In-A-Haystack), a purely synthetic long vision benchmark inspired by the language model’s NIAH test.
**c)** We propose the Long Video Assistant (LongVA) model, which can process up to 2000 frames or over 2000K visual tokens without additional complexities. With its extended context length, LongVA achieves state-of-the-art performance on Video-MME among 7B-scale models by densely sampling more input frames. 

We opensource the code and models at:
- [Github](https://github.com/EvolvingLMMs-Lab/LongVA)
- [Demo](https://longva-demo.lmms-lab.com/)
- [Checkpoints](https://huggingface.co/lmms-lab)
  - [LongVA-7B](https://huggingface.co/lmms-lab/LongVA-7B)
  - [LongVA-7B-DPO](https://huggingface.co/lmms-lab/LongVA-7B-DPO)


# Introduction

Driven by the progress of Large Language Models (LLMs), multiple studies are conducted to extend their capability to understand images and videos. With modality alignment and visual instruction tuning, these Large Multimodal Models (LMMs) have shown impressive abilities such as captioning and visual question-answering. While current LMMs have demonstrated promising performance on tasks involving single images and short videos, effectively processing and understanding extremely long videos remains a significant challenge.

One of the primary reasons for this challenge is the excessive number of visual tokens generated by the vision encoder. For instance, LLaVA-1.6 can produce 576 to 2880 visual tokens for a single image. The number of visual tokens increases significantly with the addition of more frames. To address this problem, numerous methods have been proposed to reduce the number of visual tokens. One popular direction is to modify the visual resampler that connects the vision encoder and LLM, aiming to extract fewer tokens. Alternative approaches employ heuristic techniques to prune or merge the visual features. However, despite these efforts, Table 1 demonstrates that the majority of current LMMs are still limited in their ability to process a large number of frames effectively.

Another issue hindering the development of high-performance long video LMMs is the lack of high-quality long video datasets. In Table 1, we list the average video length of existing video instruction tuning data. Most datasets consist of video clips within 1 minute. Even if some datasets do contain longer videos, the corresponding text pairs are generated by annotating only several frames within that video, lacking long and dense supervision signals. 

This work addresses both of the aforementioned problems.
For modeling, instead of reducing the visual tokens, we identify the more critical issue limiting the visual context length in existing LMMs: the context length of the language model backbone. Given a language model, we first extend its context length by training on longer text data. We then use this context-extended LM as the backbone to perform modality alignment and visual instruction tuning without any long video text pairs. By training this way, the context length of the language model is directly transferred to that of the LMMs. 

To facilitate benchmarking and accurately assess the context length in the visual domain, we created V-NIAH, a synthetic visual benchmark based on the Needle-in-a-haystack test used in language models. Our model, Long Video Assistant (LongVA), is capable of accurately retrieving visual information from 2000 frames or more than 224K visual tokens. Experiments show that additional frames during inference lead to improved performance on long video question-answering benchmarks, and LongVA achieves state-of-the-art performance among 7B models on the Video-MME dataset. 
In summary, our paper makes the following contributions:

**(1) Long Context Transfer (LCT)**: We discovered the LCT phenomenon where the context of the language model can be directly transferred to the modality-aligned multi-modal models.

**(2) Long Video Assistant (LongVA)**: Leveraging the LCT property, we developed LongVA model that can perceive up to 224K visual tokens without any long video data during training.

**(3) Visual Needle-In-A-Haystack (V-NIAH)**: We proposed V-NIAH benchmark testify LMMs ability in locating and retrieving visual information over extremely long contexts.

[![longva-table1.png](https://i.postimg.cc/NfH07NkQ/longva-table1.png)](https://postimg.cc/RNv9vLYY)

# Long Video Assistant

As in Figure 1, this paper centers around the hypothesis that *if the modality of vision and language can be truly aligned, the capability to handle long contexts could also transfer from text to vision*, and this could happen even without explicit long video training. Our methodology is thus very straightforward. Given a language model, we first perform long context training purely on language to extend its text context in [Training Long Language Model](#training-long-language-model). We then detailed how we augment this language model with long visual capabilities by training solely on short image data in [Visual-Language Alignment](#vl-alignment).

![LongVA Plot Main](https://i.postimg.cc/SQZZ7dN1/longva-figure1.png)

## Training Long Language Model
We use Qwen2-7B-Instruct as the backbone language model and perform continued pretraining with a context length of 224K[^1] over a total of 900M tokens. We follow [Xiong et al. (2023)](https://arxiv.org/abs/2309.16039) to increase RoPE base frequency during the continued pertaining and specifically set it to 1B. A constant learning rate of 1e-5 is maintained for a batch size of one million tokens across 1,000 training steps. Following [Fu et al. (2024)](https://arxiv.org/abs/2402.10171), we construct the dataset used for long context training from Slimpajama by upsampling documents longer than 4096 and keeping the domain mixture ratio unchanged. Multiple documents are packed into a single sequence separated by a BOS token.

We employed several optimization strategies to perform training on such long sequences. These include FlashAttention-2, Ring Attention, activation checkpointing, and parameter offload. To balance the load across different GPUs, we shard the sequence in a zigzag way in ring attention. The resulting training framework is memory efficient and maintains very high GPU occupancy. Note that we do not use any parameter-efficient methods such as LoRA or approximate attention. With those optimizations, the compute used in long context training is minimal compared to that of language model pretraining, making it feasible for academic budgets. The long context training can finish in 2 days with 8 A100 GPUs.

## Aligning Long Language Model Using Short Vision Data

Inspired by the *AnyRes* encoding scheme in LLaVA-NeXT, we designed *UniRes* that provides a unified encoding scheme for both images and videos, as shown in Figure 2. Unlike *AnyRes* which retains a small base image and flattens ViT patches across the grids, *UniRes* removes the base image and flattens patches within each grid. This approach allows us to maintain consistent representation when extending image data into videos where multiple frames are viewed as multiple grids in a row. To clearly ablate the long context transfer phenomenon from language to vision, we only use image-text data during training (and thus no long video data during training).

![LongVA Figure 2](https://i.postimg.cc/V6z8MgLL/longva-figure2.png)

# Experiments

## V-NIAH Evaluations

To measure the context length of language models on extremely long input, earlier works calculate perplexity scores over long documents. Recently, many have started using the Needle-in-a-Haystack (NIAH) test to benchmark LLMs' ability to retrieve long context information precisely. We note that there is so far no benchmark to measure the visual context length of LMMs.
To evaluate LongVA's capacity to locate and retrieve long-range visual information, we extend the NIAH test from text to video and propose V-NIAH. 

As shown in Figure 3, we designed 5 video question-answering problems as the needle and inserted each as a single frame into hours-long videos. We sampled the videos at 1 FPS as the visual input.  The image of the needle is sourced from existing VQA benchmarks or AI-generated to avoid any contamination. The AI-generated images and questions are purposely chosen to be "counterfactual" or "counter-commonsense", ensuring the model cannot answer based on language knowledge alone. Each question includes a "locating prompt" so that a capable system or human can locate the needle frame from the video haystack and answer the question.


When testing LongVA with visual inputs of up to 3000 frames, one difficulty we encountered was that processing a 200K-token input requires up to 100GB of GPU memory for the KV cache for a 7B LM like LLaMA. Even with advanced LM serving systems like vLLM with tensor parallelism to shard the KV cache across multiple GPUs, the sampling process remains extremely slow due to limited memory and batchsize. To address this, we used "perplexity-based" evaluation to measure the correctness of the model output. We first encode all frames and save their corresponding visual embeddings. During the evaluation, we only load the language model from LongVA and concatenate the visual embeddings, question tokens, and answer tokens for a single forward pass with ring attention. This approach makes the workload compute-bound and eliminates the need to cache the KV state. The model's output is considered correct only if the highest output logits index of all tokens in the answer span matches the correct answer.

![LongVA Figure 3](https://i.postimg.cc/FzRXCKLM/longva-figure3.png)

Figure 4 shows the V-NIAH performance of LongVA. We also include LLaVA-NeXT-Video-32K, a 7B video LMM trained on Mistral-Instruct-v0.2 with a 32K context length, and LongVA trained with *AnyRes* encoding scheme as baselines. In the bottom left plot of Figure 4, we observe that the visual context length of LLaVA-NeXT-Video-32K is limited by the context length of its language model backbone. Once the LM training length is exceeded, V-NIAH accuracy drops significantly. To extend the context length of the language model, we used training-free length extrapolation by increasing the RoPE base frequency. We tested frequencies of [3M, 10M, 30M, 100M, 300M, 1B] to find the best configuration. As shown in the bottom right plot, training-free extrapolation allows the language model to retrieve information over a longer context, but the performance boost is still limited. This motivated us to train a long context language model on text and then augment it with vision, resulting in LongVA. As shown in the top left plot, LongVA can almost perfectly retrieve information and answer the needle question for input frames fewer than 2000. Although we only trained LongVA's language backbone on a context length of 224K (equivalent to 1555 frames), it generalizes well beyond that, maintaining satisfactory performance within 3000 frames.

We also present the V-NIAH heatmap of LongVA trained with *AnyRes* encoding scheme, keeping all other factors unchanged. As shown at the top right of Figure 4, LongVA-*AnyRes* demonstrates strong retrieval capabilities. However, its performance still lags behind LongVA trained with UniRes. We believe that the unified representation of images and videos in UniRes, where a video is encoded in the same way as a long image, enhances the long context transfer from language to vision. This approach also facilitates effective training with short vision data (images) and enables zero-shot understanding of long videos during inference.

## Video Evaluations

V-NIAH is suitable for providing quick validation signals during the model development cycle. Being synthetic, constructing such benchmarks is generally low-cost and can extend to arbitrary context lengths without extra human annotation. However, it only tests the model's ability to retrieve information and does not cover other abilities necessary for a long video assistant. Therefore, we also evaluate LongVA on real-world long and short video datasets in this section. Note that LongVA is trained without any video data. Unless otherwise specified, results listed in this section can be considered *zero-shot*. 

In Table 4, we present LongVA's performance on Video-MME, a comprehensive evaluation suite for video LMMs that includes diverse data types and qualitative annotations. Video-MME is an ideal benchmark for assessing LMMs' ability to handle long videos in real-world scenarios, given its average video duration of 1017 seconds and the inclusion of short, medium, and long subsets. On Video-MME, LongVA achieves state-of-the-art performance among LMMs under 10B parameters and rivals the performance of much larger ones like LLaVA-NeXT-Video-34B and InternVL-Chat-V1.5. Notably, as the number of frames during inference increases, LongVA consistently shows improved performance on the long subset. However, its performance on the short and medium subset saturates at 32 frames and 64 frames, respectively. These results demonstrate the necessity of having long video evaluation benchmarks, as short videos do not fully capture the model's capabilities with long-range inputs. The strong results achieved by LongVA without any video data during training confirm the effectiveness of *long context transfer from language to vision*, where LongVA is able to understand long videos by transferring the long context capability from the language model.

![LongVA Table 4](https://i.postimg.cc/zBx9h1c0/longva-table4.png)

## Image Evaluations

We further evaluate our model on various image benchmarks to investigate the image performance of LongVA (Table 6). We include LLaVA-1.6-Vicuna, LLaVA-NeXT-LLaMA3 and LLaVA-NeXT-Qwen2 as strong baselines, where LLaVA-NeXT-Qwen2 model is trained by us following the  LLaVA-NeXT training strategy and data recipe. To ablate the influence of *AnyRes* vs. *UniRes*, we also report the performance of LongVA trained with LLaVA-NeXT's original *AnyRes* strategy. Note that *UniRes* operate 2x2 average pooling on each image, reducing to *1/4* visual tokens per image grid. However, the grid upper bound is set to 49 for *UniRes* while 4 for *AnyRes*, so *UniRes* may produce more image grids if the input images are of higher resolution. Compared to *AnyRes*, *UniRes* achieved significantly increased performance InfoVQA, while the scores drop to some extent on AI2D and ChartQA.

![LongVA Table 6](https://i.postimg.cc/SsnwcPb4/longva-table5.png)

# Conclusion

This work addresses the challenges of understanding long videos in Large Multimodal Models. By extending the language model on text and then aligning this extended model with visual inputs, we significantly improved the capability of LMMs to handle long videos thanks to the long context transfer phenomenon. Our model, LongVA, shows improved performance with more input frames and achieves state-of-the-art results on Video-MME. Additionally, we introduce a synthetic benchmark, V-NIAH, to effectively measure the visual context length of video LMMs.  We hope this work inspires further research in the field of long-vision models and multimodal agents.

[^1]: 224K is the maximum we can fit with 8 A100-80G for Qwen-2-7B. We find that the embedding size significantly impacts the maximum sequence length in our optimized codebase. Qwen2 has a huge vocabulary of 152K tokens. For LLaMA2 with 32K vocabulary, we can train it with 700K context length.