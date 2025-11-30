from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# ============================================================
# BUILD MODEL
# ============================================================
def build_xgb_model():
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    return model


# ============================================================
# TRAIN MODEL
# ============================================================
def train_xgb(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# ============================================================
# EVALUATE MODEL
# ============================================================
def evaluate_xgb(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


# ============================================================
# SAVE ARTEFACTS FOR XAI
# ============================================================
def save_for_xai(model, X_train, X_test, y_train, y_test, save_dir="src/xgb_model"):
    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "xgb_model.pkl"))
    joblib.dump(X_train, os.path.join(save_dir, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(save_dir, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(save_dir, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(save_dir, "y_test.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(save_dir, "feature_names.pkl"))

    print(f"\nArtifacts saved to folder: {save_dir}")



# ============================================================
# FULL PIPELINE
# ============================================================
def train_xgb_pipeline(X_train, X_test, y_train, y_test):
    model = build_xgb_model()
    model = train_xgb(model, X_train, y_train)
    evaluate_xgb(model, X_test, y_test)
    save_for_xai(model, X_train, X_test, y_train, y_test, save_dir="src/xgb_model")

    return model
