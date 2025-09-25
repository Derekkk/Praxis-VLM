

<div align="center">


<h1>Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning</h1>

<div>
    <a target='_blank'>Zhe Hu<sup>1</sup>,</a>&emsp;
    <a target='_blank'>Jing Li<sup>1</sup>,</a>&emsp;
    <a target='_blank'>Zhongzhu Pu<sup>2,3</sup>,</a>&emsp;
    <a target='_blank'>Hou Pong Chan<sup>4</sup>,</a>&emsp;
    <a target='_blank'>Yu Yin<sup>5</sup></a>
</div>

<div>
    <em><sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>Tsinghua University, <sup>3</sup>InspireOmni AI</em>&emsp; 
</div>
<em><sup>4</sup>Alibaba Group, <sup>5</sup>Case Western Reserve University</em>
<div>
</div>

<div align="center">
  <a href="https://arxiv.org/pdf/2503.16965v2">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/collections/zhehuderek/praxis-vlm-67f5d8b3e077bdde7ec24baa">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collections-blue">
  </a>
</div>


---

</div>


#### 🌟 This Repo contains code and data for Praxis-VLM, which leverages textual GRPO training for vision-grounded decison making.

## 🎉 Updates
- **[2025-09]** Praxis-VLM is accepted at NeurIPS 2025!
- **[2025-06]** Training code of Praxis-VLM is released.
- **[2025-05]** Check out our [paper](https://arxiv.org/pdf/2503.16965v2) on arxiv.


## Overview
We introduce Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Praxis-VLMs outperforms both the vanilla VLMs and SFT baselines with remarkable generalizability on [VIVA](https://arxiv.org/pdf/2407.03000), [PCA-Bench](https://arxiv.org/pdf/2402.15527), and [EgoNormia](https://arxiv.org/pdf/2502.20490) benchmarks.

<div align='left'><img src="./assets/intro_figure.jpg"  alt="NAME" width="90%"/></div>


## 📚 Training Data Curation

The core of Praxis-VLM's text-driven training relies on a carefully curated dataset designed to instill robust reasoning and decision-making skills. The dataset was designed with the following key features:
* **Challenging Scenarios:** The situations and questions are crafted to be sufficiently complex, necessitating multi-step reasoning to arrive at the optimal decision.
* **Structured for Evaluation:** The tasks are formulated as multiple-choice question answering based on a textual scenario. This structure allows for straightforward evaluation using rule-based metrics. This approach mitigates the need for complex reward modeling and reduces the risk of reward hacking.
* **Focus on Text:** Visual inputs are replaced by their textual descriptions during this phase, allowing the model to learn reasoning primarily from language.

## ✨ Model Training

We employ Qwen2.5-VL 3b and 7b as the base models. For model training, we leverage [Easy-R1](https://github.com/hiyouga/EasyR1/tree/main) for GRPO implementation. For installation, please refer to the original Easy-R1 library.

For model training:

```
bash examples/qwen2_5_vl_3b_mcq_grpo.sh
```

## Model Inference
Here we use [VIVA benchmark](https://huggingface.co/datasets/zhehuderek/VIVA_Benchmark_EMNLP24) as an example. For PCA-Bench and Egonormia, you can download the data from the original hub.

```
cd scripts
python3 predict_praxis_vlm_vllm.py
```

## Evaluation

We use accuracy for model performance evaluation:
```
cd scripts
python3 evaluation.py
```

## Citation
```
@article{hu2025praxis,
  title={Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning},
  author={Hu, Zhe and Li, Jing and Pu, Zhongzhu and Chan, Hou Pong and Yin, Yu},
  journal={arXiv preprint arXiv:2503.16965},
  year={2025}
}
```
