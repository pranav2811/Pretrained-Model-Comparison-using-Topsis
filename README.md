# Text Similarity Model Comparison using TOPSIS

## Overview:

This repository contains code written in Jupyter Notebook for comparing pretrained models of text similarity and ranking them using TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution). The models are evaluated based on several criteria including accuracy, precision, recall, F1 score, and mean squared error.

## Libraries Used:

-NLTK
-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-learn
-Gensim
-Transformers

## Introduction

Text similarity is an important task in natural language processing (NLP), with applications such as duplicate detection, plagiarism detection, and semantic similarity analysis. This project aims to compare different pre-trained models for measuring text similarity using the TOPSIS method.


## Installation

You can install the required libraries using pip:

```bash
pip install nltk pandas numpy matplotlib seaborn scikit-learn gensim transformers
```
## Files

1. **dataset_cleaning.ipynb**: Jupyter Notebook for cleaning the dataset.

2. **implementation_of_models.ipynb**: Jupyter Notebook for implementing pretrained models.

3. **topsis_results.ipynb**: Jupyter Notebook for TOPSIS evaluation and results.

4. **clean.csv**: Output file containing the cleaned dataset.

5. **summary_df.csv**: Output file containing summary data.

6. **topsis_results.csv**: Output file containing TOPSIS evaluation results.

7. **README.md**: Markdown file providing an overview of the repository and instructions for use.

## Instructions

1. **Python**: Ensure that you have Python installed on your machine along with the required libraries

2. **Clone the Repository**: Use Git to clone the repository to your local machine.

    ```bash
    git clone https://github.com/yourusername/text-similarity-topsis.git
    ```

3. **Navigate to the Directory**: Move into the cloned directory.

    ```bash
    cd text-similarity-topsis
    ```
Open the respective jupyter Notebooks ('dataset_cleaning.ipynb', 'implementation_of_models.ipynb', 'topsis_results.ipynb') using Jupyter Notebook or Jupyter Lab.

4. **Run the Script**:  Execute the cells in the notebooks sequentially to run the code


## Example Output

After running the cells in the jupyter notebook you will see how each model performed in terms of accuracy, precision, recall, F1 score, and mean squared error along with the topsis score and the rank. The output will be in the form of a table and a csv file will also be generated.
