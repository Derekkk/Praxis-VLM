

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
    <sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>Tsinghua University, <sup>3</sup>InspireOmni AI&emsp; 
</div>
<sup>4</sup>Alibaba Group, <sup>5</sup>Case Western Reserve University
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
- **[2025-05]** Check out our [paper](https://arxiv.org/pdf/2503.16965v2) on arxiv.
- **[2025-05]** Training code of Praxis-VLM is released.


## Overview
We introduce Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Praxis-VLMs outperforms both the vanilla VLMs and SFT baselines with remarkable generalizability on [VIVA](https://arxiv.org/pdf/2407.03000), [PCA-Bench](https://arxiv.org/pdf/2402.15527), and [EgoNormia](https://arxiv.org/pdf/2502.20490) benchmarks.

<div align='left'><img src="./assets/intro_figure.jpg"  alt="NAME" width="90%"/></div>


