import quantus

import numpy as np
import pandas as pd
import shap
import quantus
import xgboost
import warnings
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

def calculate_faithfulness(model, X, shap_values, num_runs=100, subset_size=5):
    faithfulness_scores = []
    
    # Baseline for masking (mean of the dataset)
    baseline = X.mean(axis=0)
    
    for i in range(len(X)):
        x_instance = X.iloc[i].values
        shap_instance = shap_values[i]
        original_pred = model.predict_proba(x_instance.reshape(1, -1))[0, 1]
        
        feature_indices = np.arange(len(x_instance))
        
        # Run multiple random perturbations for robust correlation
        dropped_preds = []
        attributions_sum = []
        
        for _ in range(num_runs):
            # Select random subset of features to mask
            mask_indices = np.random.choice(feature_indices, size=subset_size, replace=False)
            
            # Create perturbed instance (replace selected features with baseline mean)
            x_perturbed = x_instance.copy()
            x_perturbed[mask_indices] = baseline[mask_indices]
            
            # Predict on perturbed instance
            perturbed_pred = model.predict_proba(x_perturbed.reshape(1, -1))[0, 1]
            
            # Record the drop in probability and the sum of SHAP values for masked features
            pred_drop = original_pred - perturbed_pred
            shap_sum = np.sum(shap_instance[mask_indices])
            
            dropped_preds.append(pred_drop)
            attributions_sum.append(shap_sum)
        
        # Calculate correlation for this instance
        if np.std(dropped_preds) > 0 and np.std(attributions_sum) > 0:
            corr, _ = pearsonr(attributions_sum, dropped_preds)
            faithfulness_scores.append(corr)
            
    return np.mean(faithfulness_scores)


def calculate_stability(model, explainer, X, shap_values, nr_samples=10, noise_scale=0.1):
    stability_scores = []
    
    for i in range(len(X)):
        x_instance = X.iloc[i].values
        original_expl = shap_values[i]
        
        max_sensitivity = 0
        
        for _ in range(nr_samples):
            # Add uniform noise
            noise = np.random.uniform(-noise_scale, noise_scale, size=x_instance.shape)
            x_perturbed = x_instance + noise
            
            # Get explanation for perturbed instance
            # Note: We must re-compute SHAP for the perturbed point
            perturbed_expl = explainer.shap_values(x_perturbed.reshape(1, -1))
            if isinstance(perturbed_expl, list): perturbed_expl = perturbed_expl[1]
            perturbed_expl = perturbed_expl[0] # Unwrap batch
            
            # Calculate distances
            dist_input = np.linalg.norm(x_instance - x_perturbed)
            dist_expl = np.linalg.norm(original_expl - perturbed_expl)
            
            # Avoid division by zero
            if dist_input > 1e-6:
                sensitivity = dist_expl / dist_input
                if sensitivity > max_sensitivity:
                    max_sensitivity = sensitivity
                    
        stability_scores.append(max_sensitivity)
        
    return np.mean(stability_scores)

def calculate_sparseness(shap_values, epsilon=1e-5):
    # Flatten all SHAP values
    all_shap = shap_values.flatten()
    # Count how many are close to zero
    num_near_zero = np.sum(np.abs(all_shap) < epsilon)
    # Calculate ratio
    sparseness = num_near_zero / len(all_shap)
    return sparseness
