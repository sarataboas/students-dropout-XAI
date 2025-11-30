from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from interpret import show


import matplotlib.pyplot as plt

# Data Splitting Function
def split_data(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


######################## Decision Tree Classifier ###########################

def decision_tree_classifier(X_train, y_train, X_test, y_test, max_depth=None, random_state=42):
    '''
    Trains a Decision Tree Classifier on the provided training data.
    Evaluates the model.
    Plots the decision tree.
    '''
    # training 
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree_clf.fit(X_train, y_train)

    # evaluation
    y_pred =  tree_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Visualization
    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, feature_names=X_train.columns, class_names=['Dropout', 'Graduate'], filled=True)
    plt.title(f'Decision Tree (max_depth={max_depth})')
    plt.show()

    return tree_clf


######################## Explainable Boosting Machine ###########################
def explainable_boosting_classifier(X_train, y_train, X_test, y_test):
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_train, y_train)


    y_pred = ebm.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return ebm