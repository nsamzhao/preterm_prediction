from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os
import torch
import numpy as np

if __name__ == "__main__":
    """
    Classifier #1: Control vs. sPTD
    """
    data_dir = "./data/torch/clf_1"
    X = torch.load(os.path.join(data_dir, "X_train_100genes.pt"))
    y = torch.load(os.path.join(data_dir, "y_train.pt"))

    # Convert to numpy arrays
    X = np.float32(X.numpy())
    y = np.float32(y.numpy())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    clf = LogisticRegression(random_state = 42).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Score for Control vs. sPTD is: {}".format(score))
    
    """
    Classifier #2: Control vs. PPROM
    """
    data_dir = "./data/torch/clf_2"
    X = torch.load(os.path.join(data_dir, "X_train_100genes.pt"))
    y = torch.load(os.path.join(data_dir, "y_train.pt"))

    X = np.float32(X.numpy())
    y = np.float32(y.numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    clf = LogisticRegression(random_state = 42).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Score for Control vs. PPROM is: {}".format(score))
