import pandas as pd
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    random.seed(seed)

def load_and_preprocess_data(csv_file, test_size=0.2):
    try:
        df = pd.read_csv(csv_file)

        if len(df.columns) >= 2:
            query_col = df.columns[0]
            label_col = df.columns[1]

            queries = df[query_col].astype(str).values
            labels = df[label_col].astype(int).values
        else:
            raise ValueError("CSV file must have at least two columns: query and label")

        valid_indices = ~pd.isnull(queries)
        queries = queries[valid_indices]
        labels = labels[valid_indices]

        queries, labels = balance_dataset(queries, labels)

        num_samples = len(queries)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        test_samples = int(num_samples * test_size)
        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:]

        train_queries = queries[train_indices]
        train_labels = labels[train_indices]
        test_queries = queries[test_indices]
        test_labels = labels[test_indices]

        return train_queries, train_labels, test_queries, test_labels

    except Exception as e:
        raise


def balance_dataset(queries, labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) <= 1:
        return queries, labels

    if max(counts) / min(counts) < 1.2:
        return queries, labels

    minority_class = unique_labels[np.argmin(counts)]
    majority_class = unique_labels[np.argmax(counts)]

    minority_indices = np.where(labels == minority_class)[0]
    majority_indices = np.where(labels == majority_class)[0]

    oversampled_indices = np.random.choice(
        minority_indices,
        size=len(majority_indices) - len(minority_indices),
        replace=True
    )

    balanced_indices = np.concatenate([
        minority_indices,
        majority_indices,
        oversampled_indices
    ])

    balanced_queries = queries[balanced_indices]
    balanced_labels = labels[balanced_indices]

    shuffle_indices = np.random.permutation(len(balanced_queries))
    balanced_queries = balanced_queries[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]

    return balanced_queries, balanced_labels