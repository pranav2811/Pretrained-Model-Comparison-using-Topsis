from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def calculate_similarity(texts, model_name):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize the texts
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings of the texts
    with torch.no_grad():
        output = model(**tokenized_texts)
        embeddings = output.last_hidden_state[:, 0, :].numpy()

    # Calculate cosine similarity between the texts
    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix

def topsis(similarity_matrices):
    n_models = len(similarity_matrices)
    n_texts = similarity_matrices[0].shape[0]

    # Normalize the similarity matrices
    normalized_matrices = [(sim_mat - sim_mat.min(axis=1)[:, np.newaxis]) / (sim_mat.max(axis=1) - sim_mat.min(axis=1))[:, np.newaxis] for sim_mat in similarity_matrices]

    # Compute the weighted normalized decision matrix
    weights = np.ones(n_models) / n_models
    weighted_matrices = [normalized_mat * weight for normalized_mat, weight in zip(normalized_matrices, weights)]
    decision_matrix = np.stack(weighted_matrices, axis=-1).mean(axis=-1)

    # Calculate the ideal and anti-ideal solutions
    ideal_solution = decision_matrix.max(axis=0)
    anti_ideal_solution = decision_matrix.min(axis=0)

    # Calculate the distance to the ideal and anti-ideal solutions
    d_positive = np.sqrt(np.sum((decision_matrix - ideal_solution)**2, axis=1))
    d_negative = np.sqrt(np.sum((decision_matrix - anti_ideal_solution)**2, axis=1))

    # Calculate the TOPSIS score
    topsis_score = d_negative / (d_positive + d_negative)

    return topsis_score

if __name__ == "__main__":
    texts = ["Text 1", "Text 2", "Text 3"]  # Provide your list of texts
    model_names = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]  # List of model names to compare

    similarity_matrices = [calculate_similarity(texts, model_name) for model_name in model_names]

    topsis_score = topsis(similarity_matrices)
    print("TOPSIS scores and ranks:")
    sorted_indices = np.argsort(-topsis_score)  # Sort in descending order
    for rank, index in enumerate(sorted_indices):
        print(f"{rank + 1}. {model_names[index]}: {topsis_score[index]}")
