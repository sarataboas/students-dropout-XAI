from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import scipy.stats as ss
# from io import BytesIO
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", palette="pastel")

####################### Dataset Overview and Quality Assessment ###########################


def plot_target_distribution(df):
    target_counts = df['Target'].value_counts()
    target_pct = target_counts / target_counts.sum() * 100
    plt.figure()
    target_pct.plot(kind='bar')
    plt.title("Target Variable Distribution (%)")
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.show()


####################### Feature Assessment and Visualization ###########################


# ------- Univariate Analysis ------- #

# 1. Numerical Features Statistics
def numerical_feature_stats(feature_name, df):
    '''
    compute statistics for a given feature in the dataframe: total count, distinct count (%), missing count (%), memory size, mean, std, coefficient of variation, min, 25%, median(50%), 75%, max.
    args:
        feature_name (str)
        df (pd.DataFrame)
    returns:
        dict: dictionary containing various statistics about the feature.
    '''
    feature = df[feature_name]
    stats = {}
    stats['total_count'] = len(feature)
    distinct_count = feature.nunique(dropna=True)
    stats['distinct_count'] = distinct_count / len(feature)  
    missing_count = feature.isnull().sum()
    stats['missing_count(%)'] = (missing_count / len(feature)) * 100
    memory_size = feature.memory_usage(deep=True)
    stats['memory_size(bytes)'] = memory_size

    # Compute descriptive statistics
    stats['mean'] = feature.mean()
    stats['std'] = feature.std()
    if feature.mean() != 0:
        stats['coefficient of variation'] = stats['std'] / stats['mean']
    stats['min'] = feature.min()
    stats['25%'] = feature.quantile(0.25)
    stats['median(50%)'] = feature.median()
    stats['75%'] = feature.quantile(0.75)
    stats['max'] = feature.max()
    if feature.mean() != 0:
        stats['coefficient of variation'] = stats['std'] / stats['mean']
    
    return stats

def detect_outliers(feature_name, df, multiplier=1.5):
    '''
    detect outliers in numerical features using the IQR rule --> every point outside the Q1 - 1.5*IQR and Q3 + 1.5*IQR is considered an outlier.
    args:
        feature_name (str)
        df (pd.DataFrame)
        multiplier (float): multiplier for the IQR to define outlier boundaries (1.5 from documentation)
    returns:
        numerical_outliers (pd.Series): series containing the outlier values.
        outliers_perc (float): percentage of outliers in the feature.
        lower_bound (float): lower bound for outlier detection. (for plotting the boxplot)
        upper_bound (float): upper bound for outlier detection.(for plotting the boxplot)
    '''
    feature_data = df[feature_name].dropna()
    Q1, Q3 = np.percentile(feature_data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    numerical_outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    outliers_perc = (len(numerical_outliers) / len(feature_data)) * 100
    return numerical_outliers, outliers_perc, lower_bound, upper_bound


# 2. Categorical Features Description
def categorical_feature_desc(feature_name, df):
    '''
    describes a given categorical feature: total count, distinct categories, missing count (%), most frequent category, memory size.
    args:
        feature_name (str)
        df (pd.DataFrame)
    returns:
        dict: dictionary containing various descriptions about the categorical feature.
    '''
    feature = df[feature_name]
    descriptions = {}
    descriptions['total_count'] = len(feature)
    distinct_count = feature.nunique(dropna=True)
    descriptions['distinct_categories'] = distinct_count
    missing_count = feature.isnull().sum()
    descriptions['missing_count(%)'] = (missing_count / len(feature)) * 100
    memory_size = feature.memory_usage(deep=True)
    most_frequent = feature.mode().iloc[0] if not feature.mode().empty else np.nan
    descriptions['most_frequent_category'] = most_frequent
    descriptions['memory_size(bytes)'] = memory_size

    return descriptions

def rare_categories(feature_name, df, threshold=0.01):
    '''
    identify rare categories in a categorical feature based on a frequency threshold.
    args:
        feature_name (str)
        df (pd.DataFrame)
        threshold (float): frequency threshold to consider a category as rare (set to 1%)
    returns:
        rare_categories (pd.Series): series containing the rare categories and their frequencies.
        rare_categories_perc (float): percentage of rare categories among all distinct categories.
    '''
    feature_data = df[feature_name].dropna()
    value_counts = feature_data.value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold]
    rare_categories_perc = len(rare_categories) / len(value_counts) * 100
    # print(rare_categories)
    # print(len(rare_categories))
    return rare_categories, rare_categories_perc
    


# 3. Rendering Functions
def render_numerical_feature(feature_name, df, target_col, bins=30, bar_width=3):
    '''
    render visualizations for numerical features. numerical_feature_card includes:
        - summary statistics table
        - histogram of feature distribution
        - boxplot with outliers indicated
        - boxplot of feature vs target variable
    '''
    stats = numerical_feature_stats(feature_name, df)
    numerical_outliers, outliers_perc, lower_bound, upper_bound = detect_outliers(feature_name, df)
    feature = df[feature_name].dropna()
    stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)

    target_filtered = df[df[target_col].isin(["Dropout", "Graduate"])]


    fig, (ax_table, ax_hist, ax_box, ax_target) = plt.subplots(1, 4, figsize=(20, 4))


    # Feature vs Taregt Boxplot
    sns.boxplot(
        data=target_filtered,
        x=target_col,
        y=feature_name,
        ax=ax_target,
        hue=target_col, 
        palette=["#efa3a0", '#CAE08DB4']
    )

    ax_target.set_title(f"{feature_name} vs. Target", fontsize=13, pad=10)
    ax_target.set_xlabel("Target", fontsize=11)
    ax_target.set_ylabel(feature_name, fontsize=11)

    # Outliers Boxplot
    sns.boxplot(y=feature, ax=ax_box, color="#93C5D4", orient="v")
    ax_box.axhline(lower_bound, color='red', linestyle='--')
    ax_box.axhline(upper_bound, color='red', linestyle='--')
    ax_box.set_title(f"{feature_name} Boxplot (IQR Outliers)")
    ax_box.set_ylabel(feature_name)
    ax_box.set_xlabel("")


    # Histogram -- Feature Distribution
    sns.histplot(feature, bins=bins, kde=True, ax=ax_hist, shrink=bar_width, color="#93C5D4")
    ax_hist.set_title(f"{feature_name} Distribution")
    ax_hist.set_xlabel(feature_name)
    ax_hist.set_ylabel("Count")

    # stats table
    ax_table.axis("off")
    ax_table.text(0.02, 1.05, "Summary statistics", fontsize=11, weight="bold", family="DejaVu Sans")
    y_start = 0.95
    y_step = 1.0 / (len(stats_df) + 2)
    font_family = "DejaVu Sans"
    ax_table.text(0.02, y_start, "Metric", weight="bold", fontsize=10, family=font_family)
    ax_table.text(0.98, y_start, "Value", weight="bold", fontsize=10, family=font_family, ha="right")

    y = y_start - y_step
    for i, (metric, value) in enumerate(stats_df.values):
        y_mid = y - (y_step / 2)
        ax_table.text(0.02, y_mid, metric, fontsize=9.5, family=font_family, color="#333", va="center")
        ax_table.text(0.98, y_mid, value, fontsize=9.5, family=font_family, color="#333", ha="right", va="center")
        ax_table.plot([0.02, 0.98], [y - y_step, y - y_step], color="#D3D3D3", lw=0.6)
        y -= y_step

    plt.tight_layout()
    plt.show()

def render_categorical_feature(feature_name, df, target_col):
    '''
    render visualizations for categorical features. categorical_feature_card includes:
        - summary descriptions table
        - frequency bar chart
        - undirected barplot of feature vs target variable
    '''
    descriptions = categorical_feature_desc(feature_name, df)
    feature = df[feature_name].dropna()

    vc = feature.value_counts(dropna=True)
    if not vc.empty:
        descriptions['mode_freq(%)'] = (vc.iloc[0] / len(feature)) * 100
    stats_df = pd.DataFrame(list(descriptions.items()), columns=["Metric", "Value"])
    stats_df["Value"] = stats_df["Value"].apply(
        lambda x: f"{x:,.4f}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
    )


    target_filtered = df[df[target_col].isin(["Dropout", "Graduate"])]

    fig, (ax_table, ax_freq, ax_target) = plt.subplots(1, 3, figsize=(20, 4))

    # Feature vs Target Undirected Barplot
    top_n = 10
    if target_filtered[feature_name].nunique() > top_n:
        top_categories = target_filtered[feature_name].value_counts().index[:top_n].tolist()
        plot_data = target_filtered[target_filtered[feature_name].isin(top_categories)].copy()
        order = top_categories
        title = f"Dropout/Graduate Proportion by {feature_name} (Top {top_n})"
    else:
        plot_data = target_filtered.copy()
        order = list(plot_data[feature_name].value_counts().index)
        title = f"Dropout/Graduate Proportion by {feature_name}"

    # Ensure categorical order is respected
    plot_data[feature_name] = pd.Categorical(plot_data[feature_name], categories=order, ordered=True)

    sns.histplot(
        data=plot_data,
        y=feature_name,
        hue=target_col,
        multiple="fill",            # 100% stacked bars
        palette=['#efa3a0', '#CAE08DB4'],
        shrink=0.8,
        ax=ax_target
    )
    ax_target.set_title(title, fontsize=13, pad=10)
    ax_target.set_xlabel("Proportion", fontsize=11)
    ax_target.set_ylabel(feature_name, fontsize=11)
    ax_target.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc="upper left")


    # Frequency bar chart
    freq_data = feature.value_counts(normalize=False).sort_values(ascending=False)
    sns.barplot(
        x=freq_data.index,
        y=freq_data.values,
        ax=ax_freq,
        color="#DDBFE2"
    )
    ax_freq.set_title(f"{feature_name} Frequency", fontsize=13, pad=10)
    ax_freq.set_xlabel(feature_name, fontsize=11)
    ax_freq.set_ylabel("Count", fontsize=11)
    ax_freq.set_xticklabels(ax_freq.get_xticklabels(), rotation=45, ha="right")

    # Descriptions table
    ax_table.axis("off")
    ax_table.text(0.02, 1.05, "Summary statistics", fontsize=11, weight="bold", family="DejaVu Sans")

    y_start = 0.95
    y_step = 1.0 / (len(stats_df) + 2)
    font_family = "DejaVu Sans"
    ax_table.text(0.02, y_start, "Metric", weight="bold", fontsize=10, family=font_family)
    ax_table.text(0.98, y_start, "Value",  weight="bold", fontsize=10, family=font_family, ha="right")

    y = y_start - y_step
    for metric, value in stats_df.values:
        y_mid = y - (y_step / 2)
        ax_table.text(0.02, y_mid, metric, fontsize=9.5, family=font_family, color="#333", va="center")
        ax_table.text(0.98, y_mid, value,  fontsize=9.5, family=font_family, color="#333", va="center", ha="right")
        ax_table.plot([0.02, 0.98], [y - y_step, y - y_step], color="#D3D3D3", lw=0.6)
        y -= y_step

    plt.tight_layout()
    plt.show()

def visualize_features_cards(df, numerical_features_list, categorical_features_list, target_col='Target'):
    '''
    visualize all features in the dataframe using their respective rendering functions.
    '''
    for feature in numerical_features_list:
        render_numerical_feature(feature, df, target_col)
    for feature in categorical_features_list:
        render_categorical_feature(feature, df, target_col)


 
# ------- Multivariate Analysis ------- #

# 1. Correlation Matrix for Numerical Features
def plot_numeric_correlation_matrix(df, numerical_features_list):
    '''
    plot correlation matrix for numerical features in the dataframe.
    args:
        df (pd.DataFrame)
        numerical_features_list (list): list of numerical feature names.
    '''
    spearman_corr = df[numerical_features_list].corr(method='spearman')
    plt.figure(figsize=(14, 10))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Spearman Correlation (Numeric vs Numeric)", fontsize=14, pad=12)
    plt.show()


# 2. Correlation between Categorical Features

def cramers_v(x, y):
    """Compute Cramér’s V correlation between two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def plot_categorical_correlation_matrix(df, categorical_features):
    """
    Compute and visualize a Cramér’s V correlation matrix
    for a given list of categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing categorical columns.
    categorical_features : list
        List of column names (categorical features).

    Returns
    -------
    cramers_v_matrix : pd.DataFrame
        Symmetric DataFrame of pairwise Cramér’s V correlations.
    """

    # Filter only provided categorical columns
    cat_df = df[categorical_features].copy()

    # Initialize empty matrix
    cramers_v_matrix = pd.DataFrame(
        np.zeros((len(cat_df.columns), len(cat_df.columns))),
        columns=cat_df.columns,
        index=cat_df.columns
    )

    # Compute pairwise Cramér's V
    for col1 in cat_df.columns:
        for col2 in cat_df.columns:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0
            else:
                cramers_v_matrix.loc[col1, col2] = cramers_v(cat_df[col1], cat_df[col2])

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cramers_v_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cramér's V"}
    )
    plt.title("Cramér's V Correlation Matrix (Categorical Features)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()

def plot_numeric_categorical_correlation_matrix(df, numeric_features, categorical_features, bins=5):
    """
    Compute and visualize the Cramér’s V correlation matrix between
    numerical and categorical features (numericals are discretized first).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numeric and categorical variables.
    numeric_features : list
        Names of numeric features.
    categorical_features : list
        Names of categorical features.
    bins : int, optional (default=5)
        Number of bins to discretize numeric features.

    Returns
    -------
    corr_matrix : pd.DataFrame
        DataFrame with Cramér’s V correlations (numeric vs categorical).
    """

    # Copy selected columns
    num_df = df[numeric_features].copy()
    cat_df = df[categorical_features].copy()

    # Discretize numeric variables into equal-frequency bins (quantiles)
    discretized_num_df = pd.DataFrame()
    for col in num_df.columns:
        discretized_num_df[col] = pd.qcut(num_df[col], q=bins, duplicates='drop')

    # Initialize empty matrix
    corr_matrix = pd.DataFrame(
        np.zeros((len(discretized_num_df.columns), len(cat_df.columns))),
        index=discretized_num_df.columns,
        columns=cat_df.columns
    )

    # Compute pairwise Cramér's V
    for num_col in discretized_num_df.columns:
        for cat_col in cat_df.columns:
            corr_matrix.loc[num_col, cat_col] = cramers_v(discretized_num_df[num_col], cat_df[cat_col])


    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Cramér's V"},
        linewidths=0.5,
        square=False
    )
    plt.title("Cramér's V Correlation Matrix (Numeric vs Categorical Features)", fontsize=14, pad=12)
    plt.xlabel("Categorical Features")
    plt.ylabel("Numeric Features (Discretized)")
    plt.tight_layout()
    plt.show()