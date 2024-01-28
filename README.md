# Text Similarity Model Comparison using TOPSIS

This repository contains Python code for comparing various text similarity models using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method. The code utilizes the Hugging Face `transformers` library for loading pre-trained models and calculating text embeddings, and the `scikit-learn` library for computing cosine similarity and implementing TOPSIS.

## Introduction

Text similarity is an important task in natural language processing (NLP), with applications such as duplicate detection, plagiarism detection, and semantic similarity analysis. This project aims to compare different pre-trained models for measuring text similarity using the TOPSIS method.

## Requirements

- Python 3.x
- `transformers` library from Hugging Face
- `torch` library
- `numpy` library
- `scikit-learn` library

## Installation

You can install the required libraries using pip:

```bash
pip install transformers torch numpy scikit-learn
```
## Usage

1. **Clone the Repository**: Use Git to clone the repository to your local machine.

    ```bash
    git clone https://github.com/yourusername/text-similarity-topsis.git
    ```

2. **Navigate to the Directory**: Move into the cloned directory.

    ```bash
    cd text-similarity-topsis
    ```

3. **Run the Script**: Execute the Python script to compare text similarity models using TOPSIS.

    ```bash
    python compare_similarity_models.py
    ```

## Explanation

The `compare_similarity_models.py` script performs the following steps:

1. **Loading Models**: It loads pre-trained models from Hugging Face (`bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`).

2. **Calculating Similarity**: It calculates the cosine similarity matrix between a given set of texts using each model.

3. **TOPSIS Method**: It implements the TOPSIS method to compare the models based on their similarity matrices.

4. **Printing Results**: Finally, it prints the TOPSIS scores and ranks of the models.

## Example Output

After running the script, you will see the output displaying the TOPSIS scores and ranks of the models:

