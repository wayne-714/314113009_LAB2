import numpy as np


def read_bci_data():
    """
    Load BCI Competition Dataset from .npz files.
    
    Returns:
        train_data: Training data (N, 1, T, C) where N=samples, T=time, C=channels
        train_label: Training labels (N,)
        test_data: Test data (N, 1, T, C)
        test_label: Test labels (N,)
    """
    # Load .npz files
    S4b_train = np.load('./data/S4b_train.npz')
    X11b_train = np.load('./data/X11b_train.npz')
    S4b_test = np.load('./data/S4b_test.npz')
    X11b_test = np.load('./data/X11b_test.npz')

    # Concatenate datasets
    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # Convert labels from 1-indexed to 0-indexed
    train_label = train_label - 1
    test_label = test_label - 1
    
    # Reshape: (N, C, T) -> (N, 1, T, C)
    # Add channel dimension and transpose
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    # Handle NaN values
    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(f"Train data shape: {train_data.shape}, Train label shape: {train_label.shape}")
    print(f"Test data shape: {test_data.shape}, Test label shape: {test_label.shape}")

    return train_data, train_label, test_data, test_label


def inspect_npz_file(filepath):
    """
    Inspect contents of .npz file.
    
    Args:
        filepath: Path to .npz file
    """
    data = np.load(filepath)
    print(f"\n{'=' * 60}")
    print(f"File: {filepath}")
    print(f"{'=' * 60}")
    print(f"Keys: {list(data.keys())}")
    
    for key in data.keys():
        print(f"\n{key}:")
        print(f"  Shape: {data[key].shape}")
        print(f"  Dtype: {data[key].dtype}")
        print(f"  Min: {np.min(data[key]):.4f}, Max: {np.max(data[key]):.4f}")
        print(f"  Mean: {np.mean(data[key]):.4f}, Std: {np.std(data[key]):.4f}")


if __name__ == "__main__":
    # Inspect all .npz files
    print("Inspecting .npz files...")
    inspect_npz_file('./data/S4b_train.npz')
    inspect_npz_file('./data/X11b_train.npz')
    inspect_npz_file('./data/S4b_test.npz')
    inspect_npz_file('./data/X11b_test.npz')
    
    # Test data loading
    print("\n\nTesting data loading...")
    train_data, train_label, test_data, test_label = read_bci_data()