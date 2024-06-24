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
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#long-video-assistant">Long Video Assistant</a>
    <ul>
      <li><a href="#example-demonstrations">Example Demonstrations</a></li>
      <li><a href="#v-niah-results">V-NIAH Evaluations</a></li>
      <li><a href="#video-evaluations">Video Evaluations</a></li>
      <li><a href="#image-evaluations">Image Evaluations</a></li>
    </ul>
  </li>
  <li><a href="#conclusion">Conclusion</a></li>
</ul>

<!-- # Abstract

Video sequences offer valuable temporal information, but existing large multimodal models (LMMs) fall short in understanding extremely long videos. Many works address this by reducing the number of visual tokens using visual resamplers. Alternatively, in this paper, we approach this problem from the perspective of the language model. By simply extrapolating the context length of the language model with text data alone, we enable LMMs to comprehend up to 200K visual tokens without training on any video data. 

Our paper makes the following contributions:
**a)** We explore the \textit{long context transfer} phenomenon and validate this property on both image and video benchmarks.
**b)** To effectively measure the LMM’s ability to generalize to long contexts in the vision modality, we develop V-NIAH (Visual Needle-In-A-Haystack), a purely synthetic long vision benchmark inspired by the language model’s NIAH test.
**c)** We propose the Long Video Assistant (LongVA) model, which can process up to 2000 frames or over 2000K visual tokens without additional complexities. With its extended context length, LongVA achieves state-of-the-art performance on Video-MME among 7B-scale models by densely sampling more input frames.  -->

# Introduction & Results
Gemini has amazed the world with its capability to understand hour-long videos. However, we still lack an open-source alternative with similar capabilities. Our latest research introduces an innovative solution towards long video Large Multimodal Models (LMMs), shifting the focus from reducing visual tokens per frame to leveraging the long context capabilities of language models. In this blog post, we present our SoTA video model, **Long Video Assistant (LongVA)**, and our novel benchmark, **Visual Needle-In-A-Haystack (V-NIAH)**.

**Long Context Transfer** We discovered and verified that the long context capability of language models can be directly transferred to the video domain in modality-aligned multi-modal models. On V-NIAH, LongVA is capable of accurately retrieving visual information from inputs with 2000 frames or more than 200K visual tokens.

**SoTA Performance** LongVA achieves state-of-the-art performance on the Video-MME benchmarks among 7B models. Its performance increases with denser sampling of video frames. Notably, it is the only opensource model on Video-MME that can handle 384 input frames (same as GPT4-o).

We opensource the code and models at:
- [Github](https://github.com/EvolvingLMMs-Lab/LongVA)
- [Demo](https://longva-demo.lmms-lab.com/)
- [Checkpoints](https://huggingface.co/lmms-lab)
  - [LongVA-7B](https://huggingface.co/lmms-lab/LongVA-7B)
  - [LongVA-7B-DPO](https://huggingface.co/lmms-lab/LongVA-7B-DPO)

<p align="center">
    <figure>
        <img src="https://i.postimg.cc/5tTBq2Gd/longva.png" width="800">
        <figcaption>V-NIAH helps us measure the superiour long context capability of LongVA in visual domain.</figcaption>
    </figure>
</p>


<p align="center">
    <figure>
        <img src="https://i.postimg.cc/ncWhL9QJ/Screenshot-2024-06-24-at-3-06-25-PM.png" width="800">
        <figcaption>LongVA achieves SoTA performance among 7B LMMs and is the only opensource model with 384 input frames.</figcaption>
    </figure>
</p>

# Methods

Current open source LMMs show promising performance on tasks involving single images and short videos. However, effectively processing and understanding extremely long videos remains a significant challenge. One primary difficulty is the excessive number of visual tokens generated by the vision encoder. For example, LLaVA-1.6 can produce visual tokens anywhere from 576 to 2880 for a single image. The number of visual tokens increases significantly with the addition of more frames in videos. To tackle this issue, various methods have been proposed to reduce the number of visual tokens. One popular approach is to modify the visual resampler that connects the vision encoder to the language model, aiming to extract fewer tokens. Other strategies involve heuristic techniques to prune or merge the visual features. However, despite these efforts, most current language models for multimedia still struggle to process a large number of frames effectively. At the day of writing this blog post, mojority of opensource LMMs can only handle 8 to 64 frames.

As shown in the below figure, our method shifts the focus from reducing the number of visual tokens to increasing upper bound of the visual tokens that a LMM can handle.  We  hypothesize that *if the modality of vision and language can be truly aligned, the capability to handle long contexts could also transfer from text to vision*, and this could happen even without explicit long video training. Our methodology is thus straightforward. We start with a language model and perform long context training purely on text to extend its text context capabilities. Once this is achieved, we augment the language model with visual capabilities by training it solely on short image data. If our hypothesis is true, this two-step process would ensure that the model can handle both extended text contexts and visual information effectively.

<p align="center">
    <figure>
        <img src="https://i.postimg.cc/SQZZ7dN1/longva-figure1.png" width="800">
    </figure>
</p>



## Training Long Language Model
We use Qwen2-7B-Instruct as the backbone language model and perform continued pretraining with a context length of 224K[^1] over a total of 900M tokens. We follow increase RoPE base frequency during the continued pertaining and specifically set it to 1B. A constant learning rate of 1e-5 is maintained for a batch size of one million tokens across 1,000 training steps. Following [Fu et al. (2024)](https://arxiv.org/abs/2402.10171), we construct the dataset used for long context training from Slimpajama by upsampling documents longer than 4096 and keeping the domain mixture ratio unchanged. Multiple documents are packed into a single sequence separated by a BOS token.

We employed several optimization strategies to perform training on such long sequences. These include FlashAttention-2, Ring Attention, activation checkpointing, and parameter offload. To balance the load across different GPUs, we shard the sequence in a zigzag way in ring attention. The resulting training framework is memory efficient and maintains very high GPU occupancy. Note that we do not use any parameter-efficient methods such as LoRA or approximate attention. With those optimizations, the compute used in long context training is minimal compared to that of language model pretraining, making it feasible for academic budgets. The long context training can finish in 2 days with 8 A100 GPUs.

## Aligning Long Language Model Using Short Vision Data

Inspired by the *AnyRes* encoding scheme in LLaVA-NeXT, we designed *UniRes* that provides a unified encoding scheme for both images and videos, as shown below. Unlike *AnyRes* which retains a small base image and flattens ViT patches across the grids, *UniRes* removes the base image, flattens patches within each grid, and 2x2 pool the visual features by defaul.This approach allows us to maintain consistent representation when extending image data into videos where multiple frames are viewed as multiple grids in a row. 

To clearly ablate the long context transfer phenomenon from language to vision, we adopt a *train short, test long* protocol where we only use image-text data during training, but test on long videos. Specifically, we trained our model using the same data recipe and two-stage training approach as LLaVA-1.6.

<p align="center">
    <figure>
        <img src="https://i.postimg.cc/SQZZ7dN1/longva-figure1.png" width="800">
    </figure>
</p>

<!-- # Experiments -->

## Example Demonstrations

We provide the following examples to demonstrate LongVA's capabilities on real-world long and short videos, including some extremely long videos up to 30 minutes.

For more interactive demonstrations, please refer to the [LongVA Demo](https://longva-demo.lmms-lab.com/).

## V-NIAH Evaluations

To measure the context length of language models on extremely long inputs, earlier works used perplexity scores over long documents. We propose a new benchmark, V-NIAH, to evaluate the visual context length of LMMs.

<!-- Recently, the Needle-in-a-Haystack (NIAH) test has become popular for precisely benchmarking LLMs' ability to retrieve long context information. Recognizing a gap, we adapted the NIAH test for visual contexts, creating the V-NIAH to evaluate our Long Video Assistant (LongVA) model's ability to locate and retrieve information from extensive video inputs. -->

In V-NIAH, we embedded five video question-answering challenges, termed *needles* into hours-long videos sampled at 1 FPS. These needles, sourced from existing VQA benchmarks or generated by AI to avoid biases, are designed to be counterfactual, ensuring answers rely solely on visual cues rather than language knowledge. Each needle is accompanied by a locating prompt to aid in identifying the relevant frame within the video haystack.

Testing LongVA with up to 3000 frames presented challenges, notably the requirement of up to 100GB of GPU memory for processing 200K-token inputs using a 7B LM like LLaMA. To manage this, we employed a perplexity-based evaluation, encoding all frames and saving their visual embeddings. During evaluation, we load only the necessary components of LongVA, combining the saved embeddings

![LongVA Figure 3](https://i.postimg.cc/FzRXCKLM/longva-figure3.png)

Figure 4 illustrates the performance of LongVA in the V-NIAH test, comparing it with LLaVA-NeXT-Video-32K and LongVA using the *AnyRes* encoding scheme. The bottom left plot shows that LLaVA-NeXT-Video-32K's performance declines sharply once the input exceeds its language model's training length. To address this, we experimented with training-free length extrapolation by adjusting the RoPE base frequency, testing various frequencies from 3M to 1B. Although this method extended the context length, the performance improvements were modest, as depicted in the bottom right plot. LongVA, on the other hand, can effectively retrieves information and answers questions within 2000 frames and maintains good performance up to 3000 frames, despite being trained on a context length of 224K tokens.

<!-- Additionally, the top right plot presents the V-NIAH heatmap for LongVA trained with the *AnyRes* encoding scheme. While it shows strong retrieval capabilities, it underperforms compared to LongVA trained with the UniRes scheme. UniRes's unified representation of images and videos, treating videos as long images, enhances the transfer of long context knowledge from language to vision, facilitating effective training and zero-shot understanding of long videos. -->

<!-- ## Video Evaluations

V-NIAH provides a cost-effective and scalable way to quickly validate model performance during development, as it is synthetic and does not require human annotation for extending context lengths. However, it focuses solely on information retrieval capabilities and does not assess other skills needed for a comprehensive long video assistant. Consequently, we also tested LongVA on real-world video datasets, both long and short, to evaluate its broader capabilities in a *zero-shot* setting, as LongVA is trained without video data.

In Table 4, we detail LongVA's performance on Video-MME, a robust evaluation suite for video LMMs featuring diverse data types and qualitative annotations. With an average video duration of 1017 seconds, Video-MME is well-suited for testing LMMs on long video content. Here, LongVA not only sets a new standard for LMMs under 10B parameters but also competes closely with larger models like LLaVA-NeXT-Video-34B and InternVL-Chat-V1.5. LongVA's performance notably improves with the number of frames in the long video subset, while it plateaus at 32 frames for short videos and 64 frames for medium videos. These findings highlight the importance of long video benchmarks in fully assessing a model's capabilities, as shorter videos may not challenge the model's long-range processing skills. LongVA's strong performance across these tests, without any video training data, underscores the successful transfer of *long-context understanding from language to vision*.



## Image Evaluations

We further evaluate our model on various image benchmarks to investigate the image performance of LongVA (Table 6). We include LLaVA-1.6-Vicuna, LLaVA-NeXT-LLaMA3 and LLaVA-NeXT-Qwen2 as strong baselines, where LLaVA-NeXT-Qwen2 model is trained by us following the  LLaVA-NeXT training strategy and data recipe. To ablate the influence of *AnyRes* vs. *UniRes*, we also report the performance of LongVA trained with LLaVA-NeXT's original *AnyRes* strategy. Note that *UniRes* operate 2x2 average pooling on each image, reducing to *1/4* visual tokens per image grid. However, the grid upper bound is set to 49 for *UniRes* while 4 for *AnyRes*, so *UniRes* may produce more image grids if the input images are of higher resolution. Compared to *AnyRes*, *UniRes* achieved significantly increased performance InfoVQA, while the scores drop to some extent on AI2D and ChartQA.

![LongVA Table 6](https://i.postimg.cc/SsnwcPb4/longva-table5.png) -->

# Conclusion

This work addresses the challenges of understanding long videos in Large Multimodal Models. By extending the language model on text and then aligning this extended model with visual inputs, we significantly improved the capability of LMMs to handle long videos thanks to the long context transfer phenomenon. Our model, LongVA, shows improved performance with more input frames and achieves state-of-the-art results on Video-MME. Additionally, we introduce a synthetic benchmark, V-NIAH, to effectively measure the visual context length of video LMMs.  We hope this work inspires further research in the field of long-vision models and multimodal agents.

[^1]: 224K is the maximum we can fit with 8 A100-80G for Qwen-2-7B. We find that the embedding size significantly impacts the maximum sequence length in our optimized codebase. Qwen2 has a huge vocabulary of 152K tokens. For LLaMA2 with 32K vocabulary, we can train it with 700K context length.