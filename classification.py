import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import probplot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import project_globals


#   stratified split. Allows a repetitive division,
#   considers the stratas (groups) that should be equaly represented in testing
def split_by_target_classes(data, test_ratio=0.2, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    #   here we are considering stratas for a more representative test sample
    #   we stratify by the target

    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)

    for train_index, test_index in split.split(data, data["target"],):
        strat_train_set = data.iloc[train_index]
        strat_test_set = data.iloc[test_index]

    return strat_train_set, strat_test_set


def get_sets(data: pd.DataFrame):
    train_set, test_set = split_by_target_classes(data)
    x_train, y_train = train_set.drop('target', axis=1), train_set['target'].copy()
    x_test, y_test = test_set.drop('target', axis=1), test_set['target'].copy()
    return x_train, y_train, x_test, y_test


def get_data():
    return pd.read_csv(project_globals.DATA_FRAME_PATH+'2024-05-21_09-09-30', index_col='song_name')


def visualise_data(data):
    for cat, group in data.groupby('target'):
        for column in group.columns:
            if column != 'target':
                plt.figure()
                probplot(group[column], dist="norm", plot=plt)
                plt.title(f'Q-Q Plot for {cat} - {column}')
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel('Ordered Values')
                plt.grid(True)
                plt.show()


if __name__ == '__main__':
    data = get_data()
    data = data[data['target'] != 'Queens of the Stone Age']
    data = data[data['target'] != 'Rap']
    to_drop = ['repetitiveness', 'max_spectral_centroid', 'median_spectral_rolloff_high_pitch',
               'accented_Hzs_median', 'loudness_variation', 'note_above_threshold_set']
    data.drop(to_drop, axis=1, inplace=True)
    print(data.columns)
    # visualise_data(data)

    X_train, y_train, X_test, y_test = get_sets(data)

    # Initialize the model
    # model = SVC()  # works 0.55
    # model = DecisionTreeClassifier()  # works 0.6
    # model = RandomForestClassifier()  # works 0.65
    model = GradientBoostingClassifier()  # works 0.7

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)




    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
