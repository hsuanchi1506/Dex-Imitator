---
license: gpl-3.0
pipeline_tag: robotics
library_name: pytorch
---

# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

[![Paper](https://img.shields.io/badge/Paper-red?style=for-the-badge&labelColor=B31B1B&color=B31B1B)](https://arxiv.org/abs/2503.21860)
[![Project](https://img.shields.io/badge/Project-orange?style=for-the-badge&labelColor=D35400)](https://maniptrans.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-orange?style=for-the-badge&labelColor=FFD21E&color=FFD21E)](https://huggingface.co/datasets/LiKailin/DexManipNet)


This model is described in the paper [ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](https://huggingface.co/papers/2503.21860).  It's a two-stage method for efficiently transferring human bimanual skills to dexterous robotic hands in simulation.  The model first pre-trains a generalist trajectory imitator and then fine-tunes a specific residual module.

For code and usage instructions please see the project's Github repository: [ManipTrans](https://github.com/ManipTrans/ManipTrans).