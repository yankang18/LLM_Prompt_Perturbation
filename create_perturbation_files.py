import json
import re

import numpy as np
import tqdm
# from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import llm_model_checkpoint, llm_model_name, data_save_dir
from utils import NumpyEncoder


def contains_english_chars(string):
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)


def contains_non_english_chars(string):
    pattern = r'[^a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)


# def filter_tokens(token2index):
#     filtered_index2token = {}
#     for key, idx in tqdm.tqdm(token2index.items()):
#         #     print(val)
#         if key.startswith('<'):
#             continue
#
#         if not key.startswith('Ġ'):
#             continue
#
#         val_ = key.replace("Ġ", "")
#
#         if val_ == val_.upper():
#             continue
#
#         if contains_non_english_chars(val_):
#             continue
#
#         if 3 < len(val_) < 16 and contains_english_chars(val_):
#             filtered_index2token[idx] = key
#     #         if len(filtered_index2token) > 10000:
#     #             break
#     return filtered_index2token

def filter_tokens(token2index):
    filtered_index2token = {}
    for token, idx in tqdm.tqdm(token2index.items()):
        #     print(val)
        if token.startswith('<'):
            continue

        if not token.startswith('▁'):
            continue

        val_ = token.replace("▁", "")

        if val_ == val_.upper():
            continue

        if contains_non_english_chars(val_):
            continue

        if 3 < len(val_) < 16 and contains_english_chars(val_):
            filtered_index2token[idx] = token

    return filtered_index2token


def cosine_similarity(embedding_matrix1, embedding_matrix2):
    # Compute dot product of the matrices
    dot_product = np.dot(embedding_matrix1, embedding_matrix2.T)

    # Compute the norm of the matrices
    norm_matrix1 = np.linalg.norm(embedding_matrix1, axis=1)
    norm_matrix2 = np.linalg.norm(embedding_matrix2, axis=1)

    # Compute cosine similarity
    similarity = dot_product / (np.outer(norm_matrix1, norm_matrix2))

    return similarity


def create_sorted_similarities_for_tokens(token_list, similarity_matrix):
    tokens_with_sorted_similarity = dict()
    token_array = np.array(token_list)
    for idx, token in tqdm.tqdm(enumerate(token_list)):
        similarity_array = similarity_matrix[idx]
        sorted_indices = np.argsort(similarity_array)[::-1]

        tokens_with_sorted_similarity[token] = [token_array[sorted_indices], similarity_array[sorted_indices]]
    return tokens_with_sorted_similarity


def create_sensitivity_of_embeddings(all_embedding_matrix):
    n_dimensions = all_embedding_matrix.shape[1]
    delta_f_new = np.zeros(n_dimensions)
    for dim in tqdm.trange(n_dimensions):
        dim_data = all_embedding_matrix[:, dim]
        sorted_dim_data = np.sort(dim_data)
        differences = sorted_dim_data[-1] - sorted_dim_data[0]
        delta_f_new[dim] = differences
    return delta_f_new


def get_embedding(model):
    # Get the embedding layer weights
    embedding_weights = model.get_input_embeddings().weight
    # Convert the embedding layer weights to numpy
    return embedding_weights.detach().numpy()


def compute_token_2_embedding(index_2_token_dict, embedding_weights, model_name, save_dir):
    token_2_embedding_dict = {}
    for idx, token in tqdm.tqdm(index_2_token_dict.items()):
        token_2_embedding_dict[token] = embedding_weights[idx].tolist()

    file_full_path = save_dir + 'token_2_embedding_{}.json'.format(model_name)
    with open(file_full_path, 'w') as f:
        json.dump(token_2_embedding_dict, f, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Saved token_2_embedding dictionary to {file_full_path}.")
    return token_2_embedding_dict


def compute_embedding_similarity_matrix(token_2_embedding, model_name, save_dir):
    embedding_matrix = np.array(list(token_2_embedding.values()))
    similarity_matrix = cosine_similarity(embedding_matrix, embedding_matrix)
    file_full_path = save_dir + "similarity_matrix_{}.npy".format(model_name)
    np.save(file_full_path, similarity_matrix, allow_pickle=True)
    print("similarity_matrix shape:", similarity_matrix.shape)
    print(f"Saved similarity_matrix to {file_full_path}.")
    return similarity_matrix


def compute_token_2_sorted_similarity(token_list, similarity_matrix, model_name, save_dir):
    _token_sorted_similarity_dict = create_sorted_similarities_for_tokens(token_list, similarity_matrix)
    file_full_path = save_dir + 'token_sorted_similarity_dict_{}.json'.format(model_name)
    with open(file_full_path, 'w') as f:
        json.dump(_token_sorted_similarity_dict, f, cls=NumpyEncoder)
    print(f"Saved token_sorted_similarity_dict to {file_full_path}.")
    return _token_sorted_similarity_dict


def compute_embedding_sensitivity(embedding_weights, model_name, save_dir):
    _delta_f_new = create_sensitivity_of_embeddings(embedding_weights)
    file_full_path = save_dir + 'sensitivity_of_embeddings_{}.json'.format(model_name)
    with open(file_full_path, 'w') as f:
        json.dump(_delta_f_new, f, cls=NumpyEncoder)
    print(f"Saved delta_f_new! to {file_full_path}.")
    return _delta_f_new


def create_perturbation_files(llm_model_name, llm_model_checkpoint, save_dir):

    tokenizer = AutoTokenizer.from_pretrained(llm_model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(llm_model_checkpoint)
    token2index = tokenizer.get_vocab()

    embedding_weights_np = get_embedding(model)

    # remove some tokens to save time
    # NOTE: you should create your own filtered tokens
    filtered_index2token = filter_tokens(token2index)

    token_2_embedding = compute_token_2_embedding(filtered_index2token, embedding_weights_np, llm_model_name, save_dir)

    similarity_matrix = compute_embedding_similarity_matrix(token_2_embedding, llm_model_name, save_dir)

    token_list = list(token_2_embedding.keys())
    compute_token_2_sorted_similarity(token_list, similarity_matrix, llm_model_name, save_dir)

    compute_embedding_sensitivity(embedding_weights_np, llm_model_name, save_dir)

    print("Perturbation Files Creation Is Completed!")


if __name__ == "__main__":
    create_perturbation_files(llm_model_name, llm_model_checkpoint, data_save_dir)
