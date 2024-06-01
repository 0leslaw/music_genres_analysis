import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
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





def visualise_conf_matrix():
    # Plot the confusion matrix
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=y_test.unique())
    disp.plot(cmap=plt.cm.Blues)

    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.show()


def visualise_feature_distribution_in_genres(data: pd.DataFrame):
    # Group DataFrame by target variable
    groups = data.groupby('target')

    # Iterate over groups and compute histograms for each feature
    for target, group_df in groups:
        # Drop the target column
        group_df = group_df.drop('target', axis=1)
        # Iterate over features
        for col in group_df.columns:
            # if col != 'max_spectral_centroid':
            #     continue
            feature_values = group_df[col]  # Get values of the current feature

            # Compute the histogram
            min_val = feature_values.min()
            max_val = feature_values.max()

            # Automatically divide the range into 100 bins
            bins = np.linspace(min_val, max_val, num=11)
            hist, bin_edges = np.histogram(feature_values, bins=bins)

            # Plot the histogram
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')

            # Customize plot
            plt.title(f"Distribution of '{col}' in class '{target}'")
            plt.xlabel("Feature Values")
            plt.ylabel("Count")

            # Reduce the number of ticks on the x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(10))  # Set a maximum of 10 ticks on the x-axis

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plt.savefig('./plots/features_distribution_in_genres/'+col+target.__str__())
            plt.close()


if __name__ == '__main__':
    data = get_data()
    data = data[data['target'] != 'Queens of the Stone Age']
    # data = data[data['target'] != 'Blues']
    # data = data[data['target'] != 'Rap']
    # to_drop = ['repetitiveness', 'median_spectral_rolloff_high_pitch',
    #            'accented_Hzs_median', 'loudness_variation', 'note_above_threshold_set']
    # data.drop(to_drop, axis=1, inplace=True)
    # data = data[data['max_spectral_centroid'] < 0.9]
    # data = data[data['max_spectral_centroid'] > 0.1]
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

    visualise_conf_matrix()

    visualise_feature_distribution_in_genres(data)
