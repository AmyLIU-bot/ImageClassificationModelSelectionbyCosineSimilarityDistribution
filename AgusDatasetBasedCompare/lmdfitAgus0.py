#Compare with lmdfitAgus.py:return max_height instead of skewness in cosim_pairs_threads method
import os
import re
import time
import random
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from scipy.stats import skew
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate
from codecarbon import OfflineEmissionsTracker

model_bert = "google-bert/bert-base-uncased"
model_scibert = "allenai/scibert_scivocab_uncased"
model_legalbert = "nlpaueb/legal-bert-base-uncased"
model_financialbert = "ahmedrachid/FinancialBERT"
model_phambert = "Lianglab/PharmBERT-uncased"
model_agriculturebert = "recobo/agriculture-bert-uncased"
model_chemicalbert = "recobo/chemical-bert-uncased"

models = [
    model_bert,
    model_scibert,
    model_legalbert,
    model_financialbert,
    model_phambert,
    model_agriculturebert,
    model_chemicalbert
]

def data_grouping(csv_file, column, size, num_quantiles):

    data = pd.read_csv(csv_file).dropna().drop_duplicates()

    data["length"] = data[column].apply(len)

    quantiles = data["length"].quantile(
        [i / num_quantiles for i in range(1, num_quantiles)]
    ).tolist()

    quantiles = np.unique(quantiles).tolist()

    bins = [data["length"].min()] + quantiles + [data["length"].max()]

    labels = [f"Q{i+1}" for i in range(len(bins) - 1)]

    data["quantiles"] = pd.cut(
        data["length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop"
    )

    counts = data["quantiles"].value_counts().sort_index()
    print(counts)

    n_samples = size // num_quantiles

    min_samples_per_category = data["quantiles"].value_counts().min()

    if min_samples_per_category < n_samples:
        n_samples = min_samples_per_category

    return data, n_samples

def running_sizes(
    csv_file,
    sizes,
    x_column,
    num_repetitions=1,
    num_quantiles=100,
    max_token=True,
    sentence_padding=True,
    models=models,
    batch_size=64
):
    results_sizes = {}

    for repetition in range(num_repetitions):
        print("=================================================")
        print(f"Repetition {repetition + 1}/{num_repetitions}")

        random_seed = np.random.randint(1000)

        for size in sizes:
            print(f"Collecting data for sample size {size} with random seed {random_seed}...")

            grouped_data, n_samples = data_grouping(
                csv_file,
                x_column,
                size,
                num_quantiles
            )

            result_df = (
                grouped_data
                .groupby("quantiles")
                .apply(lambda x: x.sample(n_samples, random_state=random_seed))
                .reset_index(drop=True)
            )

            token_counts = result_df[x_column].str.split().apply(len)

            print(f"Sample Size: {size}")
            print(f"Actual Sample Size: {len(result_df)}")
            print(f"Minimum Token Count: {token_counts.min()}")
            print(f"Average Token Count: {token_counts.mean()}")
            print(f"Maximum Token Count: {token_counts.max()}")

            if max_token:
                token_length = 512
            else:
                token_length = token_counts.max()

            for model in models:
                if size not in results_sizes:
                    results_sizes[size] = {}

                if model not in results_sizes[size]:
                    results_sizes[size][model] = []

                result = cosim_pairs_threads(
                    result_df,
                    x_column,
                    model,
                    max_token=token_length,
                    sentence_padding=sentence_padding,
                    batch_size=batch_size
                )

                results_sizes[size][model].append(result)

    return results_sizes

def compute_embeddings_batch(
    texts,
    model,
    tokenizer,
    batch_size=32,
    max_token=512,
    sentence_padding=True
):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=sentence_padding,
            max_length=max_token,
            truncation=True
        )

        with torch.no_grad():
            batch_outputs = model(**inputs)

            # Mean pooling over token embeddings
            batch_embeddings = batch_outputs.last_hidden_state.mean(dim=1)

        embeddings.append(batch_embeddings)

    return torch.cat(embeddings)

def calculate_cosine_similarity_gpu(embeddings_data):
    similarity_scores = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_data = embeddings_data.to(device)

    num_samples = len(embeddings_data)

    for i in range(num_samples):
        embedding_1 = embeddings_data[i]

        for j in range(i + 1, num_samples):
            embedding_2 = embeddings_data[j]

            similarity = torch.nn.functional.cosine_similarity(
                embedding_1,
                embedding_2,
                dim=0
            )

            similarity_scores.append(similarity.item())

    return similarity_scores
def cosim_pairs_threads(
    df,
    x_column,
    model_name,
    split_ratio=0.5,
    max_token=512,
    sentence_padding=True,
    batch_size=64
):
    start_time = time.time()

    tracker = OfflineEmissionsTracker(
        country_iso_code="JPN",
        log_level="critical"
    )
    tracker.start()

    print("Cleaning punctuation ...")

    texts_data = df[x_column].tolist()

    def cleanPunc(sentence):
        cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", " ")
        return cleaned

    sentences_data = [cleanPunc(sentence) for sentence in texts_data]

    print("Preparing BERT Embeddings ...")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    print("Transforming Data ...")

    embeddings_data = compute_embeddings_batch(
        sentences_data,
        model,
        tokenizer,
        batch_size=batch_size,
        max_token=max_token,
        sentence_padding=sentence_padding
    )

    print("Calculating cosine similarities ...")

    similarity_scores = calculate_cosine_similarity_gpu(embeddings_data)

    df_pairs = pd.DataFrame({
        "Similarity Score": similarity_scores
    })

    description = df_pairs["Similarity Score"].describe()
    formatted_description = description.apply(lambda x: f"{x:.4f}")
    print(formatted_description)

    data = df_pairs["Similarity Score"]

    mean = np.mean(data)

    # Original Fisher-Pearson skewness
    skewness = skew(data, bias=False)

    counts, bin_edges = np.histogram(data, bins=30, density=True)
    max_height = np.max(counts)

    print(f"Mean: {mean:.3f}")
    print(f"Skewness: {skewness:.3f}")
    print(f"Max Height: {max_height:.3f}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    emissions = tracker.stop()
    print(f"{emissions * 1000:.5f} gCO2eq")

    return mean, max_height, elapsed_time, emissions * 1000

def compute_embeddings_batch(
    texts,
    model,
    tokenizer,
    batch_size=32,
    max_token=512,
    sentence_padding=True
):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=sentence_padding,
            max_length=max_token,
            truncation=True
        )

        with torch.no_grad():
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1)

        embeddings.append(batch_embeddings)

    return torch.cat(embeddings)
def do_clustering(models, data):
    start_time = time.time()

    tracker = OfflineEmissionsTracker(
        country_iso_code="JPN",
        log_level="critical"
    )
    tracker.start()

    print("=====================================================")
    print("Models:", models)
    print("Data:", data)

    data = np.array(data)

    n_clusters = 2

    cluster_labels = KMeans(
        n_clusters=n_clusters,
        random_state=0
    ).fit_predict(data)

    plt.figure(figsize=(10, 6))

    for i in range(n_clusters):
        plt.scatter(
            data[cluster_labels == i, 0],
            data[cluster_labels == i, 1],
            label=f"Cluster {i + 1}"
        )

    plt.xlabel("Max Height")
    plt.ylabel("Mean")
    plt.legend()
    plt.tight_layout()

    for i, model in enumerate(models):
        plt.annotate(model, (data[i, 0], data[i, 1]))

    plt.show()

    average_skewness_cluster_0 = np.mean(data[cluster_labels == 0, 0])
    average_skewness_cluster_1 = np.mean(data[cluster_labels == 1, 0])

    if abs(average_skewness_cluster_0) < abs(average_skewness_cluster_1):
        print("Cluster 1: more-fit")
        print("Cluster 2: less-fit")
        more_fit_cluster = 0
        less_fit_cluster = 1
    else:
        print("Cluster 1: less-fit")
        print("Cluster 2: more-fit")
        more_fit_cluster = 1
        less_fit_cluster = 0

    more_fit_indices = [
        i for i in range(len(models))
        if cluster_labels[i] == more_fit_cluster
    ]

    less_fit_indices = [
        i for i in range(len(models))
        if cluster_labels[i] == less_fit_cluster
    ]

    print("Models in the more-fit cluster:", more_fit_indices)

    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    emissions = tracker.stop()
    print(f"{emissions * 1000:.5f} gCO2eq")

    return (
        more_fit_cluster,
        more_fit_indices,
        less_fit_indices,
        elapsed_time,
        emissions * 1000
    )
def display_lmdfit(result_sizes, mynumber=200):
    averages = {}
    skewness_and_mean = []
    model_names = []

    for sample_size, model_data in result_sizes.items():
        averages[sample_size] = {}

        for model_name, model_results in model_data.items():

            avg_mean = sum(result[0] for result in model_results) / len(model_results)
            avg_skewness = sum(result[1] for result in model_results) / len(model_results)

            total_time = sum(result[2] for result in model_results)
            total_emission = sum(result[3] for result in model_results)

            averages[sample_size][model_name] = (
                avg_mean,
                avg_skewness,
                total_time,
                total_emission
            )

            model_names.append(model_name)
            skewness_and_mean.append([avg_skewness, avg_mean])

    data = averages[mynumber]

    headers = ["Model", "Mean", "Max Height", "Time", "Emissions"]

    table = [
        [model, avg_mean, avg_skewness, total_time, total_emission]
        for model, (avg_mean, avg_skewness, total_time, total_emission)
        in data.items()
    ]

    total_time_all = sum(row[3] for row in table)
    total_emissions_all = sum(row[4] for row in table)

    table.append(["Total", 0, 0, total_time_all, total_emissions_all])

    print(tabulate(
        table,
        headers=headers,
        floatfmt=".5f",
        tablefmt="simple"
    ))

    print()
    print("Clustering Based on Max Height")

    more_fit_cluster, more_fit_indices, less_fit_indices, time_cluster, em_cluster = do_clustering(
        model_names,
        skewness_and_mean
    )

    return more_fit_indices, less_fit_indices


