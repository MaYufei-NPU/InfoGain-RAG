# InfoGain-RAG

Implementation of EMNLP Oral Paper: **InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering**

[![Paper](https://img.shields.io/badge/EMNLP-2025-red)](https://arxiv.org/abs/2509.12765)

## Overview

InfoGain-RAG is a novel framework that enhances Retrieval-Augmented Generation (RAG) systems by introducing Document Information Gain (DIG), a metric to quantify each document's contribution to correct answer generation. Our approach filters out irrelevant documents and prioritizes the most valuable ones through a specialized reranker trained with multi-task learning.

## Requirements

```bash
Python >= 3.8
PyTorch >= 1.10
Transformers >= 4.25.0
CUDA >= 11.3
```

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{wang2025infogain,
  title={InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering},
  author={Wang, Zihan and Liang, Zihan and Shao, Zhou and Ma, Yufei and Dai, Huangyu and Chen, Ben and Mao, Lingtao and Lei, Chenyi and Ding, Yuqing and Li, Han},
  booktitle={Proceedings of EMNLP},
  year={2025}
}
```
