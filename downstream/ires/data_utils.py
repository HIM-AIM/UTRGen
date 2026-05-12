import pandas as pd
import random
from sklearn.model_selection import train_test_split


def read_5putr_from_csv_regression_data(
    file_path,
    num_sequences=None,
    min_length=100,
    max_length=150,
    seed=42,
    label_style=None
):
    random.seed(seed)

    df = pd.read_csv(file_path)

    assert label_style in df.columns, f"CSV file must include '{label_style}' column."
    if 'utr' not in df.columns or 'te_log' not in df.columns:
        raise ValueError("CSV file must include 'utr' and 'te_log' columns.")

    filtered_df = df[df['utr'].apply(lambda x: isinstance(x, str))].copy()
    filtered_df['utr'] = filtered_df['utr'].apply(lambda x: x.replace("<pad>", ""))

    if num_sequences is not None:
        if len(filtered_df) > num_sequences:
            filtered_df = filtered_df.sample(n=num_sequences, random_state=seed)

    sequences = filtered_df['utr'].tolist()
    print(f'Extracting {label_style} column as labels...')
    labels = filtered_df[label_style].round(4).tolist()
    print(f"Loaded {len(sequences)} sequences and {len(labels)} labels from {file_path}")

    return sequences, labels


def split_data(sequences, labels, test_size=0.2, val_size=0.1, random_state=123):
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {len(train_sequences)} samples")
    print(f"Test set: {len(test_sequences)} samples")
    return train_sequences, test_sequences, train_labels, test_labels
