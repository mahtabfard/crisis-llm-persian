# CrisisSense-LLM: Classification-Guided Retrieval-Augmented Generation for Disaster Situational Awareness

> A research implementation of a **Classification-Guided Retrieval-Augmented Generation (CG-RAG)** framework for disaster-related social media analysis using **Qwen2.5-7B-Instruct**, **LoRA fine-tuning**, **FAISS vector retrieval**, and **Retrieval-Augmented Generation (RAG)**.

---

## Overview

Disaster response organizations rely on timely and accurate situational information. Although social media platforms such as **X (formerly Twitter)** provide real-time reports during disasters, extracting actionable information from massive volumes of posts remains challenging.

This project extends the ideas of **CrisisSense-LLM** by introducing a **Classification-Guided RAG pipeline**, where multi-label classification results are used to guide document retrieval before response generation.

The system first classifies a query into multiple disaster-related categories, retrieves semantically relevant tweets using vector search and label-aware filtering, and finally generates a context-aware response using a Large Language Model.

---

## Key Features

* Multi-label disaster tweet classification
* LoRA fine-tuning of **Qwen2.5-7B-Instruct**
* FAISS-based dense vector retrieval
* Classification-Guided Retrieval-Augmented Generation (CG-RAG)
* Label-aware retrieval filtering
* Checkpoint ensemble using Majority Voting
* Evaluation with Accuracy, Precision, Recall, F1-score, and Confusion Matrices
* Interactive Gradio web interface

---

## System Architecture

```
User Query
      │
      ▼
Fine-tuned Qwen2.5 Classifier
      │
      ▼
Predicted Labels
(Event Type • Informativeness • Humanitarian Category)
      │
      ▼
Classification-Guided Retriever
      │
      ▼
FAISS Vector Database
      │
      ▼
Retrieved Disaster Tweets
      │
      ▼
Qwen2.5 Generator
      │
      ▼
Final Response
```

---

## Dataset

This project is built on the **CrisisBench** benchmark dataset.

The dataset contains disaster-related tweets annotated with three tasks:

* Event Type Classification
* Informativeness Classification
* Humanitarian Category Classification

Instruction tuning samples are generated using four prompt templates, producing more than **500,000 instruction-response pairs**.

---

## Training Configuration

| Parameter     | Value               |
| ------------- | ------------------- |
| Base Model    | Qwen2.5-7B-Instruct |
| Fine-tuning   | LoRA                |
| LoRA Rank     | 32                  |
| LoRA Alpha    | 64                  |
| LoRA Dropout  | 0.05                |
| Learning Rate | 2e-5                |
| Epochs        | 1                   |
| Max Length    | 512                 |
| Quantization  | 4-bit NF4           |
| Optimizer     | AdamW Fused         |

---

## Project Structure

```
project/
│
├── dataset_builder.py
├── create_splits.py
├── augment_train.py
├── build_instruction_splits_v2.py
│
├── train.py
├── inference.py
├── evaluate_results_colab.ipynb
│
├── rag_pipeline.py
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── figures/
│
├── gradio_app.py
│
└── README.md
```

---

## Pipeline

### 1. Dataset Preparation

* Clean CrisisBench
* Train/Validation/Test split
* Data augmentation
* Instruction generation

---

### 2. Fine-tuning

The Qwen2.5-7B-Instruct model is fine-tuned using LoRA for simultaneous prediction of:

* Event Type
* Informativeness
* Humanitarian Aid Type

---

### 3. Inference

The model predicts all three labels for each query.

Multiple checkpoints can be evaluated individually or combined using majority voting.

---

### 4. Retrieval

The predicted labels guide the retrieval stage.

Instead of searching the entire vector database, retrieval is filtered according to the predicted disaster categories, reducing irrelevant documents.

---

### 5. Response Generation

Retrieved tweets are used as external knowledge for the LLM to generate an informed disaster response.

---

## Evaluation

Evaluation includes:

* Overall Accuracy
* Per-label Accuracy
* Precision
* Recall
* Macro F1
* Micro F1
* Weighted F1
* Confusion Matrix
* Training Loss Curve
* Checkpoint Comparison
* Ensemble Evaluation

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* Sentence Transformers
* FAISS
* Scikit-learn
* Gradio
* Google Colab

---

## Citation

If you use this project in your research, please cite the corresponding paper.

```
Citation information will be added after publication.
```

---

## Acknowledgments

This project builds upon the **CrisisSense-LLM** framework and the **CrisisBench** dataset while extending it with a novel **Classification-Guided Retrieval-Augmented Generation (CG-RAG)** architecture for disaster situational awareness.

---

## License

This repository is intended for academic and research purposes.
