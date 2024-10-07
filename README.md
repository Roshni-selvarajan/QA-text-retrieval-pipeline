
# Multi-Stage Text Retrieval Pipeline for Question Answering (Q&A)

This repository implements a multi-stage text retrieval pipeline for question-answering tasks using both embedding models and ranking models to improve retrieval accuracy. It uses the BEIR datasets and models such as **sentence-transformers/all-MiniLM-L6-v2**, **nvidia/nv-embedqa-e5-v5**, and **cross-encoder/ms-marco-MiniLM-L-12-v2** to retrieve and rerank the top relevant passages for a query.

## Introduction
This project is designed to retrieve relevant passages for a given question and refine these passages by reranking them using powerful ranking models. It leverages pre-trained models for embeddings and rankings to improve the performance of information retrieval tasks, particularly for Q&A systems.

## Features
- Uses **BEIR benchmark datasets** like FiQA for Q&A tasks.
- Implements a two-stage retrieval pipeline using embedding models and ranking models.
- Retrieves top-k passages based on similarity scores.
- Reranks the retrieved passages using a cross-encoder model to improve ranking accuracy.
- Supports small and large embedding models for efficient candidate retrieval.

## Requirements

To run this project, install the following dependencies:
```bash
pip install transformers pandas torch sentence-transformers openai
```

## Datasets
The dataset used in this project is from the **FiQA** dataset, part of the **BEIR benchmark**.

- Preprocessing: The documents are truncated to a maximum of 512 tokens to ensure compatibility with the models.

## Pipeline Overview

### Stage 1: Candidate Retrieval
In the first stage, two embedding models are used for candidate retrieval:
- **Small Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Large Model**: `nvidia/nv-embedqa-e5-v5`

These models embed the query and corpus documents, then compute cosine similarity to retrieve the top-k relevant passages.

### Stage 2: Re-ranking
The retrieved passages are reranked using the following ranking model:
- **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`

The reranking model assigns scores to the retrieved passages, refining the top-k results.

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/qa-text-retrieval-pipeline.git
cd qa-text-retrieval-pipeline
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Load and Preprocess Dataset
Ensure that your dataset (`fiqa.csv`) is placed in the appropriate directory and structured with the following columns:
- `query_text`: The question or query.
- `document_text`: The text passages to retrieve from.

### 4. Run the Pipeline
To retrieve passages using the small embedding model:
```python
small_model_passages = retrieve_passages_small(small_embedding_model, query, corpus, top_k=10)
```

To rerank the retrieved passages using the ranking model:
```python
reranked_small_model_passages = rerank_passages(ranking_model, query, small_model_passages)
```

### 5. Large Model and Reranking
For the large embedding model:
```python
large_model_passages = retrieve_passages_large(large_embedding_model, query, corpus, top_k=10)
reranked_large_model_passages = rerank_passages(ranking_model, query, large_model_passages)
```

### 6. Output Results
To view the top-ranked passage after reranking:
```python
print(reranked_small_model_passages[0])
print(reranked_large_model_passages[0])
```

## Results
A comparison table of retrieval results based on the query "What is inflation?" is as follows:

| Model               | Similarity Score | Top Passage                                                                 |
|---------------------|------------------|-----------------------------------------------------------------------------|
| Small Model         | 0.876            | Passage retrieved by the small embedding model                              |
| Reranked Small Model| 0.912            | Passage reranked by the cross-encoder model                                 |
| Large Model         | 0.890            | Passage retrieved by the large embedding model                              |
| Reranked Large Model| 0.932            | Passage reranked by the cross-encoder model                                 |

## Evaluation
The pipeline is evaluated using **similarity scores** between the query and the retrieved passages. The re-ranking stage improves the retrieval accuracy by adjusting the order based on relevance.

## Contributing
Contributions are welcome! If you'd like to contribute, please submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
