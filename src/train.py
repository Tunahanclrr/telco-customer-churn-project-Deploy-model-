import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_model(X_train, y_train,preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train)
    model = LGBMClassifier(
    subsample=0.8,
    scale_pos_weight=2.768561872909699,
    reg_lambda=0,
    reg_alpha=0.5,
    num_leaves=31,
    n_estimators=400,
    min_child_samples=50,
    max_depth=7,
    learning_rate=0.01,
    colsample_bytree=0.8
    )
    pipe=Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    print("model trained successfully.")
    return pipe
