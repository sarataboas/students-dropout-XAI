from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
#from alibi.explainers import AnchorTabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
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



# ============================================================
# EXPLAINABILITY WITH SURROGATE MODEL
# ============================================================
def surrogate_model_from_xgb(model, X_train):

    y_pred_train_xgb = model.predict(X_train)

    surrogate_model = DecisionTreeClassifier(max_depth=4)
    surrogate_model.fit(X_train, y_pred_train_xgb)

    return surrogate_model

def evaluate_surrogate_model(model, surrogate_model, X_test):
    y_pred_xgb = model.predict(X_test)
    y_pred_surrogate = surrogate_model.predict(X_test)

    print("\n=== Surrogate Model Classification Report ===")
    print(classification_report(y_pred_xgb, y_pred_surrogate))

    print("\n=== Surrogate Model Confusion Matrix ===")
    print(confusion_matrix(y_pred_xgb, y_pred_surrogate))

    print("\n=== R2 Score ===")
    r2 = r2_score(y_pred_xgb, y_pred_surrogate)
    print(f"R² (fidelity) of the surrogate model: {r2:.4f}")

    print("\n=== Agreement Rate ===")
    N = len(y_pred_surrogate)
    agreement = np.sum(y_pred_surrogate == y_pred_xgb) / N
    print(f"Agreement Rate: {agreement}")


def plot_surrogate_tree(surrogate_model, feature_names):
    from matplotlib import pyplot as plt
    from sklearn.tree import plot_tree

    plt.figure(figsize=(20, 10))
    plot_tree(surrogate_model, feature_names=feature_names, class_names=['Dropout', 'Graduate'], filled=True)
    plt.title('Surrogate Decision Tree')
    plt.show()



# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def feature_importance(model, feature_names):
    """
    Extracts and prints feature importance from the model.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df)
    return feature_importance_df

def plot_feature_importance(feature_importance_df):
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.show()

# ============================================================
# Partial Dependence Plots (PDP)
# ============================================================

def plot_pdp(model, X_train, features_to_plot):

    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=features_to_plot,
        feature_names=X_train.columns,
        response_method="predict_proba",   # ← IMPORTANT FIX
        ax=ax
    )

    plt.suptitle('Partial Dependence Plots')
    plt.show()


def plot_pdp_interaction(model, X_train, features_to_plot):
    
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=features_to_plot,
        feature_names=X_train.columns,
        grid_resolution=50,
        ax=ax
    )
    plt.suptitle("Partial Dependence Plot for Feature Interaction")
    plt.tight_layout()
    plt.show()



# ============================================================
# ALE
# ============================================================

def plot_ale(model, X, features, n_bins=20, target_class=1, figsize=(18, 5)):
    """
    Computes and plots Accumulated Local Effects (ALE) for a list of features.
    
    Parameters:
    - model: The trained model (must have predict_proba).
    - X: Training data (pandas DataFrame).
    - features: List of feature names (strings) to analyze.
    - n_bins: Number of intervals to divide the feature into (default: 20).
    - target_class: The class index to explain (default: 1 for 'Graduate').
    - figsize: Tuple (width, height) for the plot.
    """
    
    # --- Internal Helper: Calculate ALE for one feature ---
    def calculate_single_ale(feature_name):
        unique_vals = X[feature_name].unique()
        
        # Determine bins: Use unique values if few (categorical/discrete), else quantiles
        if len(unique_vals) <= n_bins:
            bins = np.sort(unique_vals)
        else:
            bins = np.unique(np.percentile(X[feature_name], np.linspace(0, 100, n_bins + 1)))
        
        ale = [0]  # Start with 0 effect
        bin_centers = []
        
        for k in range(len(bins) - 1):
            z_lower, z_upper = bins[k], bins[k+1]
            
            # Filter: Select only students who actually fall in this bin
            if k == 0:
                mask = (X[feature_name] >= z_lower) & (X[feature_name] <= z_upper)
            else:
                mask = (X[feature_name] > z_lower) & (X[feature_name] <= z_upper)
                
            subset = X[mask].copy()
            
            if len(subset) > 0:
                # Create synthetic lower/upper bounds for these specific students
                subset_lower, subset_upper = subset.copy(), subset.copy()
                subset_lower[feature_name] = z_lower
                subset_upper[feature_name] = z_upper
                
                # Predict probability changes
                pred_lower = model.predict_proba(subset_lower)[:, target_class]
                pred_upper = model.predict_proba(subset_upper)[:, target_class]
                
                # Average local effect
                local_effect = np.mean(pred_upper - pred_lower)
            else:
                local_effect = 0
                
            ale.append(ale[-1] + local_effect)
            bin_centers.append((z_lower + z_upper) / 2)
            
        # Center the ALE (so mean effect is 0)
        ale = np.array(ale)
        ale -= ale.mean()
        
        return bins, ale

    # --- Plotting Logic ---
    plt.figure(figsize=figsize)
    
    for i, feature in enumerate(features):
        try:
            ax = plt.subplot(1, len(features), i+1)
            
            # Calculate
            bins, ale_values = calculate_single_ale(feature)
            
            # Plot ALE line
            ax.plot(bins, ale_values, marker='o', markersize=4, label='ALE Effect')
            
            # Add Rug Plot (Data Density)
            # We sample data if it's too large to keep plotting fast
            rug_data = X[feature].sample(min(1000, len(X)), random_state=42)
            ax.plot(rug_data, np.full_like(rug_data, np.min(ale_values)), 
                    '|', color='k', alpha=0.3, label='Student Density')
            
            ax.set_title(f"ALE: {feature}")
            ax.set_xlabel(feature)
            if i == 0:
                ax.set_ylabel(f"Effect on Probability (Class {target_class})")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Could not plot {feature}: {e}")

    plt.suptitle("Accumulated Local Effects (ALE) Plots", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================
# SHAP
# ============================================================
def apply_shap(model, X_train):

    # Create a SHAP explainer for the XGBoost model
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for the subset
    shap_values = explainer.shap_values(X_train)

    return shap_values, explainer


def plot_shap_instance(shap_values, explainer, X_train, y_train, instance_index):

    instance = X_train.loc[instance_index]
    print(instance)
    print("Target = ", y_train.loc[instance_index])
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[instance_index],
            base_values=explainer.expected_value, 
            data=instance.values,  
            feature_names=X_train.columns 
        ),
        max_display=15
    )


# ============================================================
# ANCHOR EXPLANATIONS
# ============================================================
def apply_anchor(model, X_train, X_test, instance_index=0):

    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    
    feature_names = X_train.columns.tolist()
    
    # Create category map for AnchorTabular
    # {col_index: [list of unique values]}
    category_map = {
        i: X_train[col].unique().tolist() 
        for i, col in enumerate(feature_names) 
        if col in categorical_columns
    }

    # Define predictor
    predictor = lambda x: model.predict(x)

    # Initialize explainer
    explainer = AnchorTabular(predictor, feature_names, categorical_names=category_map, seed=42)

    # Fit explainer
    #print("Fitting Anchor Explainer...")
    explainer.fit(X_train.to_numpy(), disc_perc=[25, 50, 75])

    # Select instance
    instance = X_test.iloc[instance_index].to_numpy().reshape(1, -1)

    # Explain
    print(f"Explaining instance {instance_index}...")
    explanation = explainer.explain(instance)

    print(f"\n=== Anchor Explanation for instance {instance_index} ===")
    print("Anchor: %s" % explanation.anchor)
    print("Precision: %.2f" % explanation.precision)
    print("Coverage: %.2f" % explanation.coverage)

    return explanation

# ============================================================
# COUNTERFACTUAL EXPLANATIONS
# ============================================================

def pretty_print_cfs(cf_df, features_varied, label, model, original_instance):
    """
    Prints CFs only when the outcome changed.
    Requires model + original instance to compare predictions.
    """
    if cf_df.empty:
        print(f"\n[{label}] No counterfactuals generated.\n")
        return
    
    shown_df = cf_df[features_varied + ['Target']].copy()

    # Original prediction
    orig_pred = int(model.predict(original_instance)[0])

    # Check whether CF flips the prediction
    unique_targets = shown_df['Target'].unique()
    if len(unique_targets) == 1 and int(unique_targets[0]) == orig_pred:
        print(f"\n[{label}] ⚠️ No actionable change found.")
        print("The counterfactuals did NOT flip the prediction.\n")
        return

    print(f"\n[{label}] Actionable Counterfactuals (prediction flips!)")
    print(shown_df.to_string(index=False))
    print("\n")


# Helper function to align data types
def match_dtypes(query_instance, reference_df):
    """
    Forces the query instance (one row) to match the data types of the training data.
    Crucial for DiCE categorical validation (prevents '1.0' vs '1' errors).
    """
    # Get columns (excluding Target)
    cols = reference_df.drop('Target', axis=1).columns
    
    # Cast the query instance to the exact types of the reference dataframe
    return query_instance.astype(reference_df[cols].dtypes)

print("Helper function 'match_dtypes' defined.")



def select_dropout_candidate(
    X_test, 
    model, 
    df_reference, 
    grade_threshold=11, 
    approved_threshold=4,
    verbose=True
):
    """
    Finds the first student who:
      - Is a debtor
      - Has good academic performance (grade >= threshold, approved >= threshold)
      - Is predicted to Dropout (0)
    
    Automatically matches dtypes and prints the profile.
    Returns the query_instance as a single-row DataFrame.
    """

    # 1. Filter candidates
    candidates = X_test[
        (X_test['Debtor'] == 1) &
        (X_test['Curricular units 1st sem (grade)'] >= grade_threshold) &
        (X_test['Curricular units 1st sem (approved)'] >= approved_threshold)
    ]

    if verbose:
        print(f"Candidates found: {len(candidates)}")

    # 2. Select the first predicted dropout
    query_instance = None
    for idx, row in candidates.iterrows():
        candidate = row.to_frame().T
        if model.predict(candidate)[0] == 0:   # 0 = Dropout
            query_instance = candidate
            if verbose:
                print(f"\nSelected Student #{idx} (Predicted Dropout)")
            break

    if query_instance is None:
        print("\nNo matching dropout candidate found.")
        return None

    # 3. Match dtypes
    query_instance = match_dtypes(query_instance, df_reference)

    # 4. Display profile
    if verbose:
        print("\n--- Student Profile (Fixed Types) ---")
        cols_to_show = [
            'Tuition fees up to date', 
            'Debtor',
            'Marital status',
            'Curricular units 1st sem (grade)'
        ]
        print(query_instance[cols_to_show])

    return query_instance




def select_academic_slide_candidate(
    X_test, 
    model, 
    df_reference,
    max_grade_threshold=10,       # failing or weak
    max_approved_threshold=3,
    verbose=True
):
    """
    Finds a student who:
      - Has tuition fees paid (1)
      - Is a scholarship holder (1)
      - Has bad academic performance (grade <= threshold, approved <= threshold)
      - Is predicted to Dropout (0)
    """

    # 1. Filter candidates
    candidates = X_test[
        (X_test['Tuition fees up to date'] == 1) &
        (X_test['Scholarship holder'] == 1) &
        (X_test['Curricular units 1st sem (grade)'] <= max_grade_threshold) &
        (X_test['Curricular units 1st sem (approved)'] <= max_approved_threshold)
    ]

    if verbose:
        print(f"Candidates found: {len(candidates)}")

    # 2. Select the first predicted dropout
    query_instance = None
    for idx, row in candidates.iterrows():
        candidate = row.to_frame().T
        if model.predict(candidate)[0] == 0:  # 0 = Dropout
            query_instance = candidate
            if verbose:
                print(f"\nSelected Academic-Slide Student #{idx} (Predicted Dropout)")
            break

    if query_instance is None:
        print("\n No suitable Academic Slide dropout candidate found.")
        return None

    # 3. Fix data types
    query_instance = match_dtypes(query_instance, df_reference)

    # 4. Display summary
    if verbose:
        print("\n--- Academic Slide Student Profile (Fixed Types) ---")
        cols_to_show = [
            'Tuition fees up to date',
            'Scholarship holder',
            'Debtor',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (approved)'
        ]
        print(query_instance[cols_to_show])

    return query_instance


import matplotlib.pyplot as plt

def plot_intervention_impact(df_sim, student_id, threshold=0.5):
    """
    Visualize the impact of interventions on dropout risk and print a summary.

    Parameters
    ----------
    df_sim : pd.DataFrame
        Must contain columns ['Scenario', 'Risk']
    student_id : int or str
        Student identifier for the plot title
    threshold : float, default=0.5
        Safety threshold for high/low risk
    """

    # --- Visualization ---
    plt.figure(figsize=(10, 5))

    colors = ['#e74c3c' if x > threshold else '#2ecc71' for x in df_sim['Risk']]
    bars = plt.barh(df_sim['Scenario'], df_sim['Risk'], color=colors)

    plt.axvline(
        threshold,
        color='gray',
        linestyle='--',
        alpha=0.7,
        label=f'Safety Threshold ({int(threshold * 100)}%)'
    )

    for bar, risk in zip(bars, df_sim['Risk']):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{risk:.1%}",
            va='center',
            fontweight='bold'
        )

    plt.xlim(0, 1.1)
    plt.title(f"Impact of Interventions on Student #{student_id}", fontsize=14)
    plt.xlabel("Dropout Probability")
    plt.legend(loc='lower right')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()

    plt.show()

    # --- Text Summary ---
    final_risk = df_sim.iloc[-1]['Risk']
    if final_risk > threshold:
        print(
            f"Even with combined interventions, "
            f"risk remains high ({final_risk:.1%})."
        )
       
    else:
        print(
            f" SUCCESS: Interventions successfully reduce risk "
            f"to {final_risk:.1%}."
        )
