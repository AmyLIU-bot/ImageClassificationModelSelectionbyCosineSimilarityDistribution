#original agus code, only change the device capability section
import numpy as np
import pandas as pd
import time
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import re
from transformers import BertModel, BertTokenizer
import torch
from scipy.stats import skew
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from tabulate import tabulate
import matplotlib.pyplot as plt




model_bert = "google-bert/bert-base-uncased"
model_scibert = "allenai/scibert_scivocab_uncased"
model_legalbert ="nlpaueb/legal-bert-base-uncased"
model_financialbert = "ahmedrachid/FinancialBERT"
model_phambert ="Lianglab/PharmBERT-uncased"
model_biomedbert = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
model_agriculturebert = "recobo/agriculture-bert-uncased"
model_chemicalbert = "recobo/chemical-bert-uncased"

models=[model_bert,model_scibert,model_legalbert,model_financialbert,model_phambert,model_agriculturebert, model_chemicalbert]
#Insert my method about how to calculate the skewness
def asymmetry_eta(x, bandwidth=None):
    """
    Estimate eta using Patil et al.-style estimator:
    eta_hat = -Corr(f_hat(X_i), F_hat(X_i))

    f_hat(X_i): leave-one-out Gaussian KDE
    F_hat(X_i): leave-one-out empirical CDF
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 3:
        return np.nan

    # Silverman's bandwidth
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if np.isclose(sigma, 0):
        sigma = s if s > 0 else 1.0
    h = bandwidth if bandwidth is not None else 0.9 * sigma * (n ** (-1 / 5))#calculate bandwidth
    if h <= 0:
        h = 1.06 * (s if s > 0 else 1.0) * (n ** (-1 / 5))

    # Leave-one-out KDE
    diffs = (x[None, :] - x[:, None]) / h#cal culate pairwise difference
    K = stats.norm.pdf(diffs)#fit standardized distance into gaussian kernel
    np.fill_diagonal(K, 0.0)#leave one out
    f_hat = K.sum(axis=1) / ((n - 1) * h)#KED for pdf

    # Leave-one-out empirical CDF
    indicators = (x[None, :] < x[:, None]).astype(float)
    np.fill_diagonal(indicators, 0.0)
    F_hat = indicators.sum(axis=1) / (n - 1)

    if np.std(f_hat, ddof=1) == 0 or np.std(F_hat, ddof=1) == 0:
        return np.nan

    eta = -np.corrcoef(f_hat, F_hat)[0, 1]#calculate correlation

    return eta

def data_grouping(csv_file, column, size, num_quantiles):
    # Calculate lengths
    data = pd.read_csv(csv_file).dropna().drop_duplicates()  # Read the file and drop rows with missing values
    
    data['length'] = data[column].apply(len)

    # Determine quantiles for dynamic bins (according to num quantiles)
    quantiles = data['length'].quantile([i/num_quantiles for i in range(1, num_quantiles)]).tolist()
    
    
    # Ensure there are no duplicate bin edges
    quantiles = np.unique(quantiles).tolist()

    # Define bins based on unique quantiles
    bins = [data['length'].min()] + quantiles + [data['length'].max()]

    # Define labels for each quantile
    labels = [f'Q{i+1}' for i in range(len(bins) - 1)]

    # Group data based on the determined quantiles
    data['quantiles'] = pd.cut(data['length'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')
    
    
    # Count the number of samples in each quantile
    counts = data['quantiles'].value_counts().sort_index()
    print("Counts of quantiles:", counts)

    # Number of samples to pick from each group
    n_samples = size // num_quantiles 

    # Ensure we have enough samples in each category
    min_samples_per_category = data['quantiles'].value_counts().min()
    if min_samples_per_category < n_samples:
        n_samples = min_samples_per_category
        print(f"Reduced sample size to {n_samples} due to limited data in some categories.")
        
    return data, n_samples

def running_sizes(csv_file, sizes, x_column, num_repetitions=1, num_quantiles=50, max_token=True, sentence_padding=True, models=models, batch_size=64):
    # Initialize an empty dictionary to store the results
    results_sizes = {}
    sample_sizes = sizes
    # Iterate over each repetition
    for repetition in range(num_repetitions):
        print("=================================================")
        print(f"Repetition {repetition + 1}/{num_repetitions}")

        # Generate a new random seed for each repetition
        random_seed = np.random.randint(1000)

        # Collect data for all models with the same random seed
        for size in sample_sizes:
            print(f"Collecting data for sample size {size} with random seed {random_seed}...")
            
            grouped_data, n_samples = data_grouping(csv_file, x_column, size, num_quantiles)

            result_df = grouped_data.groupby('quantiles').apply(lambda x: x.sample(n_samples, random_state=random_seed)).reset_index(drop=True)

            # Calculate token counts
            token_counts = result_df[x_column].str.split().apply(len)

            # Display information about token counts
            print(f"Sample Size: {size}")
            print(f"Minimum Token Count: {token_counts.min()}")
            print(f"Average Token Count: {token_counts.mean()}")
            print(f"Maximum Token Count: {token_counts.max()}")
            
            if max_token:
                token_length = 512
            else:
                token_length = token_counts.max()    

            # Iterate over each model
            for model in models:
                # Check if the sample size key exists in the results dictionary
                if size not in results_sizes:
                    results_sizes[size] = {}

                # Check if the model key exists for this sample size
                if model not in results_sizes[size]:
                    results_sizes[size][model] = []

                # Append the results for this repetition to the corresponding list
                results_sizes[size][model].append(cosim_pairs_threads(result_df, x_column, model, 0.5, batch_size))

    return results_sizes

def cosim_pairs_threads(df,x_column,model_name,split_ratio=0.5, max_token= 512, sentence_padding=True, batch_size=64):
    start_time = time.time()
        
    # Start Code Carbon
    tracker = OfflineEmissionsTracker(country_iso_code="JPN", log_level="critical")
    
    tracker.start()

    # Start pyRAPL
    print("Cleaning punctuation ...")
    texts_data = df[x_column].tolist()
    
    
    def cleanPunc(sentence): 
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned

    # Function to split text into sentences and then clean punctuation
    def split_and_clean(text):
        sentences = sent_tokenize(text)
        cleaned_sentences = [cleanPunc(sentence) for sentence in sentences]
        return cleaned_sentences

    sentences_data = [cleanPunc(sentence) for sentence in texts_data]
    
    print("Preparing BERT Embeddings ...")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Tokenize and encode texts in the DataFrame
    embeddings_data = []
    
    batch_size =batch_size
    
    print("Transforming Data ...")

    embeddings_data = compute_embeddings_batch(sentences_data, model, tokenizer, batch_size, max_token,sentence_padding)

    
    def calculate_cosine_similarity_gpu(embeddings_data):
        similarity_scores = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings_data = embeddings_data.to(device)

        num_samples = len(embeddings_data)

        for i in range(num_samples):
            embedding_1_tensor = embeddings_data[i].clone().detach()

            for j in range(i + 1, num_samples):
                embedding_2_tensor = embeddings_data[j].clone().detach()

                similarity = torch.nn.functional.cosine_similarity(
                    embedding_1_tensor,
                    embedding_2_tensor,
                    dim=0
                )

                similarity_scores.append(similarity.item())

        return similarity_scores

    
    # Generate pairs of embeddings and calculate cosine similarity
    print("Calculating cosine similarities ...")
    similarity_scores = calculate_cosine_similarity_gpu(embeddings_data)


    # Create pairs DataFrame and calculate statistics
    df_pairs = pd.DataFrame({
        #"Data": np.repeat(sentences_data, len(sentences_data)),
        "Similarity Score": similarity_scores
    })
    
    
    description = df_pairs['Similarity Score'].describe()
    # Format the statistics with four decimal places
    formatted_description = description.apply(lambda x: f'{x:.4f}')
 
    # Print the formatted statistics
    print(formatted_description)
    
    data = df_pairs['Similarity Score']
    mean = np.mean(data)
    #original skwenss calculation
    skewness = skew(data, bias=False)

    eta = asymmetry_eta(data)
    
    
    print(f"Mean: {mean:.3f}")
    print(f"Skewness: {skewness:.3f}")
    print(f"Eta: {eta:.3f}")


    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the runtime in seconds
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    emissions = tracker.stop()
    print(f"{emissions * 1_000} gCO2eq") 
    
    return mean, eta, elapsed_time, emissions * 1000


def compute_embeddings_batch(texts, model, tokenizer, batch_size=32, max_token=512, sentence_padding=True):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=sentence_padding, max_length=max_token, truncation=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

def do_clustering(models, data):
    start_time = time.time()
        
    # Start Code Carbon
    tracker = OfflineEmissionsTracker(country_iso_code="JPN", log_level="critical")
    
    tracker.start()
    print("=====================================================")
    print(models)
    print(data)
    data = np.array(data)

    n_clusters = 2
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(data)
    #clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
    
    # Extract cluster labels and centroids
    cluster_labels = clustering #.labels_

    # Models list
    models = models

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label=f'Cluster {i + 1}')

    #plt.title(f'Clustering of BERT Models')
    plt.xlabel('Eta')
    plt.ylabel('Mean')
    plt.legend()
    #plt.grid(True)
    plt.tight_layout()

    # Annotate points with model names
    for i, model in enumerate(models):
        plt.annotate(model, (data[i, 0], data[i, 1]))

    plt.show()

    average_skewness_cluster1 = np.mean(data[cluster_labels == 0, 0])
    average_skewness_cluster2 = np.mean(data[cluster_labels == 1, 0])

    # Determine more-fit and less-fit clusters
    if abs(average_skewness_cluster1) < abs(average_skewness_cluster2):
        print("Cluster 1", "more-fit")
        print("Cluster 2", "less-fit")
        more_fit_cluster = 0
        less_fit_cluster = 1
    else:
        print("Cluster 1", "less-fit")
        print("Cluster 2", "more-fit")
        more_fit_cluster = 1
        less_fit_cluster = 0

    # Get the model indices in the more fit cluster
    more_fit_indices = [i for i in range(len(models)) if cluster_labels[i] == more_fit_cluster]
    less_fit_indices = [i for i in range(len(models)) if cluster_labels[i] == less_fit_cluster]
    print("Models in the more fit cluster:", more_fit_indices)
        
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the runtime in seconds
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    emissions = tracker.stop()
    print(f"{emissions * 1_000} gCO2eq") 

    return more_fit_cluster, more_fit_indices, less_fit_indices, elapsed_time, emissions * 1000


def display_lmdfit(result_sizes,mynumber=100):
    averages = {}
    skewness_and_mean = []
    models = []
    
    for model_id, model_data in result_sizes.items():
        averages[model_id] = {}
        for model_name, model_results in model_data.items():
            avg_mean = sum(result[0] for result in model_results) / len(model_results)
            avg_eta = sum(result[1] for result in model_results) / len(model_results)
            #avg_ks = sum(result[2] for result in model_results) / len(model_results)
            avg_time = sum(result[2] for result in model_results) 
            avg_emission = sum(result[3] for result in model_results) 
            
            averages[model_id][model_name] = (avg_mean, avg_eta, avg_time, avg_emission)
           
            models.append(model_name)
            skewness_and_mean.append([avg_eta,avg_mean])

    data = averages[mynumber]
        
    headers = ["Model", "Mean", "Eta", "Time", "Emissions"]
    table = [[model, avg1, avg2, avg3, avg4] 
             for index, (model, (avg1, avg2, avg3, avg4)) in enumerate(data.items())]

    # Calculate totals
    total_time = sum(row[3] for row in table)
    total_emissions = sum(row[4] for row in table)
    
    # Append totals row to the table
    table.append(["Total", 0, 0, total_time, total_emissions])

    print(tabulate(table, headers=headers, floatfmt=".5f", tablefmt="simple"))
    print()
    
    
    print("Clustering Based on Eta")
    more_fit_cluster, more_fit_indices, less_fit_indices, time_cluster, em_cluster = do_clustering(models, skewness_and_mean)
    #display_result(time_cluster, em_cluster,data, models, table, more_fit_cluster, more_fit_indices, less_fit_indices, total_time, total_emissions)
   
 