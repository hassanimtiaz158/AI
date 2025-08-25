import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
train_X = np.genfromtxt("trainX.txt", dtype=int, delimiter=' ')
train_Y = pd.read_csv("trainY.txt", header=None).values.flatten()
test_X  = np.genfromtxt("testX.txt", dtype=int, delimiter=' ')
test_Y  = pd.read_csv("testY.txt", header=None).values.flatten()

# --- Split into YES (digit 4) and NO (digit 2) ---
yes_train = train_X[train_Y == 4]  # all images with label 4
no_train  = train_X[train_Y == 2]  # all images with label 2

# --- Calculate probabilities ---
# Laplace smoothing added (+1 and +2)
yes_one_train  = (yes_train.sum(axis=0) + 1) / (yes_train.shape[0] + 2)
yes_zero_train = 1 - yes_one_train

no_one_train   = (no_train.sum(axis=0) + 1) / (no_train.shape[0] + 2)
no_zero_train  = 1 - no_one_train

# Priors
prior_yes = yes_train.shape[0] / train_X.shape[0]
prior_no  = no_train.shape[0] / train_X.shape[0]

# --- Prediction function ---
def predict(X):
    preds = []
    for img in X:
        log_yes = np.log(prior_yes) + np.sum(np.log(yes_one_train[img == 1])) + np.sum(np.log(yes_zero_train[img == 0]))
        log_no  = np.log(prior_no)  + np.sum(np.log(no_one_train[img == 1]))  + np.sum(np.log(no_zero_train[img == 0]))
        preds.append(4 if log_yes > log_no else 2)
    return np.array(preds)

# --- Evaluate ---
train_pred = predict(train_X)
test_pred  = predict(test_X)

train_acc = np.mean(train_pred == train_Y) * 100
test_acc  = np.mean(test_pred == test_Y) * 100

print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Testing Accuracy : {test_acc:.2f}%")

# Per-class accuracy
for digit in [2, 4]:
    train_class_acc = np.mean(train_pred[train_Y == digit] == digit) * 100
    test_class_acc  = np.mean(test_pred[test_Y == digit] == digit) * 100
    print(f"Digit {digit} - Train: {train_class_acc:.2f}%, Test: {test_class_acc:.2f}%")
