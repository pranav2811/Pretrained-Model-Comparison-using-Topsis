# Model Comparison for Text Sentence Similarity using TOPSIS

This Python script compares different models for text sentence similarity using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method. It computes similarity scores between sentences using models obtained from Hugging Face's model hub and ranks the models based on their relative closeness to the ideal solution.

## Requirements

- Python 3.x
- `transformers` library from Hugging Face
- `torch` library
- `numpy` library
- `scikit-learn` library

You can install the required libraries using pip:

```bash
pip install transformers torch numpy scikit-learn
```

## Usage

1. Clone the repository or download the script.
2. Install the required libraries as mentioned above.
3. Modify the `model_names` list to include the names of the models you want to compare.
4. Optionally, adjust the evaluation sentences as needed.
5. Run the script.

## Description

- The script loads the specified models and tokenizers from Hugging Face's model hub.
- It defines a function to compute similarity scores between sentences using the cosine similarity metric.
- The similarity scores are normalized and used to calculate the ideal and negative-ideal solutions.
- Euclidean distances to these solutions are computed, and relative closeness is calculated.
- Finally, the models are ranked based on their relative closeness to the ideal solution.

## Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the models you want to compare
model_names = ["bert-base-uncased", "roberta-base"]

# Specify evaluation sentences
sentences = ["I am happy", "I am sad"]

# Load tokenizers and models
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in model_names]
models = [AutoModelForSequenceClassification.from_pretrained(model_name) for model_name in model_names]

# (Rest of the code goes here)
```

