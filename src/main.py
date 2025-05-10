from load_data import load_data
from features import extract_features
from classifier import train_svm, train_knn, evaluate_model
from evaluate import plot_confusion_matrix, compute_per_class_accuracy

X_train, y_train, X_test, y_test = load_data(
    "../data/train-data.txt", "../data/test-data.txt"
)

X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

print("Feature shape:", X_train_feat.shape)  # Should be (7291, 6)

# Train SVM
svm_model = train_svm(X_train_feat, y_train)
svm_acc, svm_cm, _ = evaluate_model(svm_model, X_test_feat, y_test)
print(f"SVM Accuracy: {svm_acc:.4f}")

# Optionally train kNN and compare
knn_model = train_knn(X_train_feat, y_train, k=3)
knn_acc, knn_cm, _ = evaluate_model(knn_model, X_test_feat, y_test)
print(f"kNN Accuracy (k=3): {knn_acc:.4f}")

# Plot confusion matrix for SVM
plot_confusion_matrix(svm_cm, class_names=list(range(10)))


# Compute and print per-class accuracy for SVM
per_class_acc = compute_per_class_accuracy(svm_cm)
print("\nPer-Class Accuracy (SVM):")
for digit, acc in per_class_acc.items():
    print(f"Digit {digit}: {acc}%")
