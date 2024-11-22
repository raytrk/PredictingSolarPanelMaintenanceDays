from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import randint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
import settings

# Set precision to 2
pd.set_option("display.precision", 2)


def prepare_data(df):
    # Split and scale the data

    train, test = train_test_split(
        df, test_size=0.3, random_state=21, shuffle=True
    )

    X_train = StandardScaler().fit_transform(
        train.drop(columns='Daily Solar Panel Efficiency')
    )
    y_train = train['Daily Solar Panel Efficiency'].values

    X_test = StandardScaler().fit_transform(
        test.drop(columns='Daily Solar Panel Efficiency')
    )
    y_test = test['Daily Solar Panel Efficiency'].values

    return X_train, X_test, y_train, y_test


def define_models():
    # Define the models and parameter spaces for hypertuning
    model_list = []

    # Neural Net Classifier
    mlp_clf = MLPClassifier(batch_size=1000,
                            max_iter=100,
                            random_state=100
                            )
    mlp_space = {
        'hidden_layer_sizes': [(50, 50), (50, 100, 50)],
        'activation': ['tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp_name = "Neural Net Classifier"
    model_list.append([mlp_clf, mlp_space, mlp_name])

    # Random Forest
    rf_clf = RandomForestClassifier(random_state=100)
    rf_space = {'max_depth': list(np.arange(10, 101, step=10)),
                'n_estimators': np.arange(100, 301, step=100),
                'max_features': randint(3, 7),
                'min_samples_leaf': randint(1, 4),
                'min_samples_split': np.arange(2, 5, step=2)
                }

    rf_name = "RandomForestClassifier"
    model_list.append([rf_clf, rf_space, rf_name])

    # SVC
    svc_clf = SVC(gamma="auto", random_state=100)
    svc_space = {'C': list(np.logspace(-5, 4, num=10, base=10)),
                 'gamma': list(np.logspace(-5, 0, num=10, base=10)),
                 'kernel': ['rbf']}
    svc_name = "SVC"
    model_list.append([svc_clf, svc_space, svc_name])

    return model_list


def class_specific_metrics(model_name, y_test, y_pred):

    cm = confusion_matrix(
        y_test, y_pred, labels=settings.DSPE_in_numbers, normalize=None)

    # True positives
    tp = np.diag(cm)

    # True negatives
    tn = np.sum(cm) - np.sum(cm, axis=0) - \
        np.sum(cm, axis=1) + tp

    # Metrics
    accuracy = (tp+tn) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = (2 * precision*recall) / (precision + recall)

    # Compilation
    daily_solar_panel_efficiency_classes = [k for k in settings.DSPE_in_words]

    class_specific_metrics_df = pd.DataFrame(
        {'Accuracy': accuracy, 'F1': f1, 'Precision': precision}, index=daily_solar_panel_efficiency_classes)
    flattened_csm = class_specific_metrics_df.to_numpy().flatten(order='C')
    flattened_names = [
        'DSPE_' + x + ' ' + y for x in daily_solar_panel_efficiency_classes for y in ['Accuracy', 'F1', 'Precision']]
    class_specific_metrics_dict = dict(zip(flattened_names, flattened_csm))

    # Printing
    print(settings.B +
          f"Class specific metrics of {model_name}" + settings.W)
    print(class_specific_metrics_df)

    return class_specific_metrics_dict, cm


def store_results(results, model_name, y_test, y_pred):
    # Store results for each model

    # Store Generic results
    results[model_name]["Overall Accuracy"] = f"{accuracy_score(y_test, y_pred)*100:.2f}%"
    results[model_name]["Overall F1 Score"] = f"{f1_score(y_test, y_pred, average = 'macro'):.2f}"
    results[model_name]["Overall Precision"] = f"{precision_score(y_test, y_pred,average = 'macro'):.2f}"

    # Obtain class specific results and confusion matrix
    class_specific_metrics_dict, cm = class_specific_metrics(
        model_name, y_test, y_pred)

    # Store Custom results - Weighted cost
    '''
    Tenets:
    1. Greater cost to incorrect predictions of actual Low compared to actual High Solar Panel Effiency. 
    - I.e. worse to miss out on storing energy than miss out on maintenance days
    2. Lower cost to incorrectly predicting Medium Solar Panel Efficiency compared to the extremes.
    3. When we have the actual costs of false predictions, this cost_matrix can be updated accordingly
    '''
    # Actually High, but predicted incorrectly
    Pred_Medium_Actual_High = 2
    Pred_Low_Actual_High = 4
    # Actually Medium, but predicted incorrectly
    Pred_High_Actual_Medium = 1
    Pred_Low_Actual_Medium = 2
    # Actually Low, but predicted incorrectly
    Pred_High_Actual_Low = 3
    Pred_Medium_Actual_Low = 1

    cost_matrix = np.array([[0, Pred_Medium_Actual_High, Pred_Low_Actual_High],
                            [Pred_High_Actual_Medium, 0, Pred_Low_Actual_Medium],
                            [Pred_High_Actual_Low, Pred_Medium_Actual_Low, 0]])

    results[model_name]['Weighted Cost'] = np.sum(cost_matrix * cm)

    # Store class specific results
    for k, v in class_specific_metrics_dict.items():
        results[model_name][k] = v

    return results


def run_models(df):

    results = defaultdict(dict)

    X_train, X_test, y_train, y_test = prepare_data(df)
    model_list = define_models()

    # Fit and predict using each model
    for model_record in model_list:

        model, param_space, model_name = model_record

        print(settings.G +
              f"\nRunning {model_name}" + settings.W)

        # Run each model with hypertuning of parameters
        rs_model = RandomizedSearchCV(
            model, param_space, n_iter=100, scoring='f1_macro', n_jobs=-1, cv=3, random_state=100, verbose=1)
        classifier = rs_model.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Print Best hyperparameters
        print(settings.G + f"\nResults for {model_name}" + settings.W)
        print(f'Best random search hyperparameters for {model_name} are: ' +
              str(rs_model.best_params_))

        # Store results
        results = store_results(
            results, model_name, y_test, y_pred)

    # Create result tables
    overall = pd.DataFrame(results.items())[1].apply(pd.Series)
    model_names = [model_record[2] for model_record in model_list]
    overall.index = model_names
    overall.sort_values(by=['Weighted Cost'],
                        ascending=True, inplace=True)

    class_specific = overall.iloc[:, 3:]
    overall = overall.iloc[:, :3]

    print(settings.G + f"\nResult tables" + settings.W)
    print(settings.B + f"Overall results" + settings.W)
    print(overall)
    print(settings.B + f"Class specific results" + settings.W)
    print(class_specific)
