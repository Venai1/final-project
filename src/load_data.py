import numpy as np

def load_data(train_path: str, test_path: str):
    """
    Loads USPS digit dataset from plain .txt files.
    
    Each line in the file is formatted as:
    <label> <pixel1> <pixel2> ... <pixel256>
    
    Args:
        train_path (str): Path to training data .txt file
        test_path (str): Path to test data .txt file
    
    Returns:
        X_train (np.ndarray): shape (n_train_samples, 256)
        y_train (np.ndarray): shape (n_train_samples,)
        X_test (np.ndarray): shape (n_test_samples, 256)
        y_test (np.ndarray): shape (n_test_samples,)
    """
    
    def parse_file(file_path):
        features = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                labels.append(int(float(parts[0])))
                features.append([float(p) for p in parts[1:]])
        return np.array(features), np.array(labels)

    X_train, y_train = parse_file(train_path)
    X_test, y_test = parse_file(test_path)
    return X_train, y_train, X_test, y_test
