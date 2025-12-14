import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---- Function 1: Prepare Candidates ----
def get_high_risk_students(model, file_path='data.csv', risk_threshold=0.85):
    """
    Loads 'Enrolled' students, predicts their dropout risk, 
    and returns a dataframe of students sorted by risk.
    """
    df_full = pd.read_csv(file_path, sep=';')
    
    # Filter Enrolled
    df_enrolled = df_full[df_full['Target'] == 'Enrolled'].copy()
    X_enrolled = df_enrolled.drop('Target', axis=1)
    
    # Predict Risk
    probs = model.predict_proba(X_enrolled)
    X_enrolled['Dropout_Risk'] = probs[:, 0]  # Prob of Class 0 (Dropout)
    
    # Filter High Risk
    high_risk_df = X_enrolled[X_enrolled['Dropout_Risk'] > risk_threshold].sort_values('Dropout_Risk', ascending=False)
    
    print(f"Data Loaded: Found {len(high_risk_df)} high-risk students (Risk > {risk_threshold:.0%})")
    return high_risk_df

# --- Function 2: Run Simulations ---
def simulate_interventions(model, student_row):
    """
    Takes a single student row (DataFrame), runs 3 intervention scenarios,
    and returns:
    1. A DataFrame of results
    2. A boolean (True if any intervention saved the student)
    """
    # Prepare clean feature vector (remove risk column if present)
    features = student_row.drop(['Dropout_Risk'], axis=1, errors='ignore')
    baseline_risk = student_row['Dropout_Risk'].values[0]
    
    results = [{'Scenario': 'Baseline', 'Risk': baseline_risk}]
    
    # Scenario A: Pay Tuition
    sim_a = features.copy()
    sim_a['Tuition fees up to date'] = 1
    risk_a = model.predict_proba(sim_a)[0][0]
    results.append({'Scenario': 'A: Pay Tuition', 'Risk': risk_a})
    
    # Scenario B: Improve Grades (+2)
    sim_b = features.copy()
    grade_col = 'Curricular units 2nd sem (grade)'
    current_grade = sim_b[grade_col].values[0]
    sim_b[grade_col] = current_grade + 2
    risk_b = model.predict_proba(sim_b)[0][0]
    results.append({'Scenario': 'B: Improve Grades (+2)', 'Risk': risk_b})
    
    # Scenario C: Combined
    sim_c = features.copy()
    sim_c['Tuition fees up to date'] = 1
    sim_c[grade_col] = current_grade + 2
    risk_c = model.predict_proba(sim_c)[0][0]
    results.append({'Scenario': 'C: Combined', 'Risk': risk_c})
    
    df_sim = pd.DataFrame(results)
    
    # Check if student is "Savable" (Risk drops below 50%)
    is_savable = df_sim['Risk'].min() < 0.5
    
    return df_sim, is_savable

# --- Function 3: Visualize Impact ---
def plot_intervention_impact(df_sim, student_id):
    """Plots the bar chart for the simulation results."""
    plt.figure(figsize=(10, 5))
    
    # Colors: Red if > 50%, Green if < 50%
    colors = ['#e74c3c' if x > 0.5 else '#2ecc71' for x in df_sim['Risk']]
    
    bars = plt.barh(df_sim['Scenario'], df_sim['Risk'], color=colors)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Safety Threshold (50%)')
    
    # Add text labels
    for bar, risk in zip(bars, df_sim['Risk']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{risk:.1%}", va='center', fontweight='bold')
        
    plt.xlim(0, 1.1)
    plt.gca().invert_yaxis()
    plt.title(f"Intervention Simulation: Student #{student_id}")
    plt.xlabel("Dropout Probability")
    plt.legend(loc='lower right')
    plt.grid(axis='x', alpha=0.3)
    plt.show()

# --- Function 4: The Main Controller ---
def find_and_explain_savable_student(model, candidates_df):
    """
    Iterates through candidates until it finds one that becomes 'Graduate' 
    after intervention.
    """
    print("Searching for a savable student...")
    
    found = False
    count = 0
    
    for student_id, row in candidates_df.iterrows():
        count += 1
        # Convert Series to DataFrame to keep format compatible with model
        student_data = candidates_df.loc[[student_id]]
        
        # Run Simulation
        df_sim, is_savable = simulate_interventions(model, student_data)
        
        # Logic: If savable, STOP and show.
        if is_savable:
            print(f"\nSUCCESS: Found a savable student after checking {count} candidates!")
            print(f"Student ID: {student_id}")
            print(f"Baseline Risk: {df_sim.iloc[0]['Risk']:.1%}")
            
            # 1. SHAP Waterfall
            print("\n### 1. Root Cause Analysis (SHAP)")
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            # Drop risk column for SHAP
            X_shap = student_data.drop(['Dropout_Risk'], axis=1)
            shap_values = explainer(X_shap)
            shap.plots.waterfall(shap_values[0])
            
            # 2. Intervention Plot
            print("\n### 2. Intervention Simulation")
            plot_intervention_impact(df_sim, student_id)
            
            found = True
            break
            
    if not found:
        print("No savable students found in this batch (Risk remained > 50% for all scenarios).")


def find_and_explain_unsavable_student(model, candidates_df):
    """
    Finds a student who remains at High Risk (>50%) despite all interventions.
    """
    print("\n" + "="*60)
    print("SEARCHING FOR A 'HARD CASE' (UNSAVABLE STUDENT)...")
    print("="*60)
    
    found = False
    
    for student_id, row in candidates_df.iterrows():
        # Convert to DataFrame
        student_data = candidates_df.loc[[student_id]]
        
        # Run Simulation
        df_sim, is_savable = simulate_interventions(model, student_data)
        
        # Logic: If NOT savable (and risk is still high), show this case
        if not is_savable:
            final_risk = df_sim.iloc[-1]['Risk']
            print(f"FOUND: Student #{student_id} is a Hard Case.")
            print(f"Baseline Risk: {df_sim.iloc[0]['Risk']:.1%}")
            print(f"Risk after ALL interventions: {final_risk:.1%}")
            
            # 1. SHAP Waterfall (Why is the risk so stubborn?)
            print("\n### 1. Root Cause Analysis (SHAP)")
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            X_shap = student_data.drop(['Dropout_Risk'], axis=1)
            shap_values = explainer(X_shap)
            shap.plots.waterfall(shap_values[0])
            
            # 2. Intervention Plot
            print("\n### 2. Intervention Simulation")
            plot_intervention_impact(df_sim, student_id)
            
            # 3. Text Summary
            print(f"\n CONCLUSION: Interventions failed. Risk remains {final_risk:.1%}.")
            print("Recommendation: Standard automated interventions are insufficient.")
            
            
            found = True
            break
            
    if not found:
        print("Note: No completely unsavable students found in this batch.")