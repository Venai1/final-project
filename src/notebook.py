from features import projection_features
from load_data import load_data


X_train, y_train, X_test, y_test = load_data(
    "../data/train-data.txt", "../data/test-data.txt"
)

sample = X_train[0]
proj = projection_features(sample)
print("Projection features:", proj)
