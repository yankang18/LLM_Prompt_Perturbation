# from nltk.corpus import words
import json
from decimal import getcontext

import numpy as np
from transformers import AutoTokenizer

from config import llm_model_name, llm_model_checkpoint, data_save_dir

getcontext().prec = 100


def cosine_similarity_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


# def add_laplace_noise_to_vector(vector, epsilon, delta_f_new):
#     vector = np.asarray(vector, dtype=np.longdouble)
#     tt = 0
#     if (epsilon * 19.064721649556482 - 38.1294334077209) > 0:
#         tt = 0.01658160142016071 * np.log(epsilon * 19.064721649556482 - 38.1294334077209) + 9.311083811697406
#     if epsilon < 2:
#         beta_values = delta_f_new / epsilon
#     else:
#         beta_values = delta_f_new / tt
#
#     beta_values = beta_values.astype(np.longdouble)  # 确保使用高精度数据类型
#
#     noisy_vector = np.zeros_like(vector, dtype=np.longdouble)  # 使用高精度数据类型
#
#     for dim in range(len(vector)):
#         noise = np.random.laplace(0, beta_values[dim])
#         noisy_vector[dim] = vector[dim] + noise
#
#     return noisy_vector.astype(float)


def add_laplace_noise_to_vector(vector, epsilon, delta_f_new):
    vector = np.asarray(vector, dtype=np.longdouble)
    if epsilon == 0:
        beta_values = delta_f_new * 0
    else:
        beta_values = delta_f_new / (0.5 * epsilon)
    noise = np.random.laplace(loc=0, scale=beta_values, size=len(beta_values))
    noisy_vector = vector + noise

    return noisy_vector


def perturb_prompt(prompt,
                   epsilon,
                   tokenizer,
                   token_to_vector_dict,
                   sorted_distance_data,
                   delta_f_new):
    tokens = tokenizer.tokenize(prompt)

    # Pre-compute some constant values
    delta_u = 1.0  # Replace with the actual delta_u
    exp_factor = epsilon / (2 * delta_u)

    new_tokens = []
    for origin_token in tokens:
        if origin_token[0] == ' ':
            origin_token = origin_token[1:]
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            # if the token does not have embedding, we will not perturb this token.
            new_tokens.append(origin_token)
            continue

        # Add noise to the original embedding
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, delta_f_new)
        similarity = cosine_similarity_vectors(origin_embed, noise_embed)

        sorted_similarity = sorted_distance_data.get(origin_token, None)
        if sorted_similarity is None:
            continue

        token_only = sorted_similarity[0]
        similarity_only = sorted_similarity[1]

        # Reverse the array
        arr = np.flip(similarity_only)
        index = np.searchsorted(arr, similarity)
        # Adjust the index to get the correct position in the original descending array
        index = len(arr) - index

        close_tokens = token_only[:index]
        close_similarities = similarity_only[:index]
        if len(close_tokens) == 0:
            continue

        unnormalized_probabilities = np.exp(exp_factor * np.array(close_similarities))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)

    token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    # Convert token IDs back to sentence
    sentence = tokenizer.decode(token_ids)

    return sentence


def perturb_prompts(prompt_list, epsilon_list, model_name, model_checkpoint, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

    with open(save_dir + 'token_2_embedding_{}.json'.format(model_name), 'r') as f:
        token_2_embedding_dict = json.load(f)
    print("token_2_embedding loaded")

    with open(save_dir + 'token_sorted_similarity_dict_{}.json'.format(model_name), 'r',
              buffering=10 * 1024 * 1024) as f:
        token_sorted_distance_dict = json.load(f)
    print("token_sorted_similarity_dict loaded")

    with open(save_dir + 'sensitivity_of_embeddings_{}.json'.format(model_name), 'r') as f:
        delta_f_new = np.array(json.load(f))
    print("sensitivity_of_embeddings loaded")

    result_list = list()
    for prompt in prompt_list:
        print(f"Original prompt:{prompt}.")
        ptb_prompt_list = list()
        for eps in epsilon_list:
            print("=" * 100)
            print(f"Add noise with eps:{eps} to original prompt.")
            ptb_prompt = perturb_prompt(prompt=prompt,
                                        epsilon=eps,
                                        tokenizer=tokenizer,
                                        token_to_vector_dict=token_2_embedding_dict,
                                        sorted_distance_data=token_sorted_distance_dict,
                                        delta_f_new=delta_f_new)
            print(f"====> perturbed prompt: {ptb_prompt}")

            ptb_prompt_list.append({
                "ptb_prompt": ptb_prompt,
                "eps": eps
            })

        result_list.append({
            "original_prompt": prompt,
            "perturbed_prompts": ptb_prompt_list
        })

    with open(save_dir + 'perturb_result.json', 'w') as f:
        json.dump(result_list, f)
    print("perturb result saved")


if __name__ == "__main__":
    # sent = "There is this nasty intersection on my commute, I aways get stuck there waiting for a hook turn. Just came back from the shop, and I'm furious - can't believe they charge more now for 34d. I remember watching Twin Peaks after coming home from school. "
    prompt = "Lisa, a 28-year-old woman, went to the hospital for a thorough examination after experiencing " \
             "unexplained weight loss, fatigue, and frequent infections. Tests showed abnormal blood cell counts " \
             "and compromised immune function, leading to a diagnosis of Idiopathic Immunodeficiency Syndrome. " \
             "Treatment options were discussed to boost her immune system."

    # the larger, the less perturbation and more privacy leakage.
    epsilon = [1, 2, 4, 6, 8]
    prompts = [prompt]
    perturb_prompts(prompts, epsilon, llm_model_name, llm_model_checkpoint, data_save_dir)
