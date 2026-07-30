# Scientific Abstract GPT
### Decoder-Only GPT for Scientific Abstract Generation using a Custom Byte-Level BPE Tokenizer

## Overview

Scientific Abstract GPT is an end-to-end implementation of a decoder-only Generative Pre-trained Transformer (GPT) designed to generate scientific abstracts from research paper titles and subjects.

Unlike many implementations that rely on pretrained tokenizers, this project trains a **Byte-Level Byte Pair Encoding (BPE) tokenizer from scratch** on a curated subset of arXiv research papers. The entire GPT architecture is implemented in **PyTorch** without using pretrained language models.

The project demonstrates the complete NLP pipeline including data preprocessing, tokenizer training, GPT implementation, model training, evaluation, and deployment through an interactive Gradio interface.

---

## Features

- Custom Byte-Level BPE tokenizer trained from scratch
- Decoder-only GPT architecture implemented in PyTorch
- Multi-Head Masked Self-Attention
- Positional & Token Embeddings
- Feed Forward Networks
- Layer Normalization
- Residual Connections
- Autoregressive Text Generation
- Mixed Precision Training
- Cosine Learning Rate Scheduler
- Structural Token Generation
- Interactive Gradio Demo

---

## Dataset

Source:

- arXiv Research Papers

Filtered Categories:

- Artificial Intelligence (cs.AI)
- Machine Learning (cs.LG)
- Computation and Language (cs.CL)

Each training sample follows the structure:

```text
<TITLE>
Research Paper Title

<SUBJECT>
Research Area

<ABSTRACT>
Paper Abstract

<END>
```

---

## Project Structure

```
Scientific-Abstract-GPT/

│
├── data/
│   ├── bpe_tokenizer/
│   └── tokenized_bpe_streams/
│
├── models/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── gpt_model_config.json
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Tokenization.ipynb
│   ├── 03_GPT_Model.ipynb
│   ├── 04_Training.ipynb
│   ├── 05_Text_Generation.ipynb
│   ├── 06_Evaluation.ipynb
│   └── 07_Demo.ipynb
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   └── gpt_components.py
│
├── requirements.txt
└── README.md
```

---

## GPT Architecture

The model consists of:

- Token Embedding Layer
- Positional Embedding Layer
- Decoder Transformer Blocks
    - Multi-Head Self Attention
    - Feed Forward Network
    - Residual Connections
    - Layer Normalization
- Linear Language Modeling Head

Training Objective:

- Next Token Prediction

---

## Tokenizer

This project **does not use GPT-2, TikToken, SentencePiece, or any pretrained tokenizer.**

Instead, it trains a **Byte-Level BPE tokenizer from scratch** using the Hugging Face Tokenizers library.

Vocabulary Size:

```
8000 tokens
```

Special Tokens:

```
<TITLE>
<SUBJECT>
<ABSTRACT>
<END>
```

---

## Training

Optimizer

- AdamW

Loss Function

- Cross Entropy Loss

Additional Techniques

- Mixed Precision Training
- Gradient Clipping
- Warmup
- Cosine Learning Rate Scheduler

Training Hardware

- NVIDIA Tesla T4 GPU (Google Colab)

---

## Evaluation Metrics

The model was evaluated using:

- Training Loss
- Validation Loss
- Test Loss
- Test Perplexity
- Structural Compliance
- END Token Completion
- Word Diversity
- Trigram Repetition

Example Results

| Metric | Value |
|---------|------:|
| Test Loss | 3.8807 |
| Test Perplexity | 48.46 |
| Structural Compliance | 95.83% |
| END Token Completion | 83.33% |
| Word Diversity | 0.6696 |
| Trigram Repetition | 0.0083 |

---

## Demo

The project includes an interactive Gradio application.

Users provide:

- Research Title
- Research Subject

The model generates:

- Scientific Abstract

Example

**Input**

```
Title:
Transformer Models for Medical Image Analysis

Subject:
Artificial Intelligence
```

↓

**Output**

```
Generated scientific abstract...
```

---

## Technologies Used

Programming Language

- Python

Deep Learning

- PyTorch

Tokenizer

- Hugging Face Tokenizers

Dataset

- Hugging Face Datasets

Interface

- Gradio

Development Environment

- Google Colab

Version Control

- Git
- GitHub

---

## Future Improvements

- Larger GPT architecture
- Longer context window
- Fine-tuning on additional scientific domains
- Beam Search decoding
- BLEU / ROUGE evaluation
- Citation generation
- Full research paper generation

---

## Acknowledgements

This project was developed as part of a graduate Natural Language Processing / Deep Learning course.

--



## License

This project is intended for educational and research purposes.
