import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split


def read_5putr_from_csv_regression_data(
    file_path,
    num_sequences=None,
    seed=42,
    label_style='rnaseq_log',
    data_style='utr',
    max_length=256,
    clip_direction='right'
):
    random.seed(seed)

    df = pd.read_csv(file_path)

    assert label_style in df.columns, f"CSV file must include '{label_style}' column."
    assert data_style in df.columns, f"CSV file must include '{data_style}' column."

    filtered_df = df[df[data_style].apply(lambda x: isinstance(x, str))].copy()
    filtered_df[data_style] = filtered_df[data_style].apply(lambda x: x.replace("<pad>", ""))

    if num_sequences is not None:
        if len(filtered_df) > num_sequences:
            filtered_df = filtered_df.sample(n=num_sequences, random_state=seed)

    sequences = filtered_df[data_style].tolist()
    if clip_direction == 'right':
        sequences = [seq[:max_length - 1] for seq in sequences]
    elif clip_direction == 'left':
        sequences = [seq[-(max_length - 1):] for seq in sequences]
    else:
        raise ValueError("clip_direction must be 'right' or 'left'")
    labels = filtered_df[label_style].round(4).tolist()

    print(f"Start Loading sequences and labels from {file_path}")

    return sequences, labels


def read_multi_from_csv_regression_data(
    file_path,
    label_style='mean',
    data_style='utr5',
):
    df = pd.read_csv(file_path)

    filtered_df = df[df[data_style].apply(lambda x: isinstance(x, str))].copy()
    filtered_df[data_style] = filtered_df[data_style].apply(lambda x: x.replace("<pad>", ""))
    sequences = filtered_df[data_style].tolist()

    if label_style == 'mean':
        labels = np.array(filtered_df['mean_te'].round(4).tolist())
    elif label_style == 'seperate':
        te_columns = [col for col in filtered_df.columns if col.startswith('TE_')]
        labels = filtered_df[te_columns].astype(float).round(4).to_numpy()
    else:
        raise ValueError("label_style must be 'mean' or 'seperate'")

    print(f"Start Loading sequences and labels from {file_path}")

    return sequences, labels


def split_data(sequences, labels, test_size=0.2, val_size=0.1, random_state=123):
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )
    return train_sequences, test_sequences, train_labels, test_labels
