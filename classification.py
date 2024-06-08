import heapq
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import probplot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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


def get_sets(data: pd.DataFrame, seed=0):
    train_set, test_set = split_by_target_classes(data, seed=seed)
    x_train, y_train = train_set.drop('target', axis=1), train_set['target'].copy()
    x_test, y_test = test_set.drop('target', axis=1), test_set['target'].copy()
    return x_train, y_train, x_test, y_test


def get_data_as_df():
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


def visualise_conf_matrix(y_test, y_pred, model):
    # Plot the confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
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

            if col != 'max_spectral_centroid':
                continue
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
            # plt.savefig('./plots/features_distribution_in_genres/'+col+target.__str__())
            # plt.savefig('./plots/imputed_spctr_centroid_distribution/' + col + target.__str__())

            plt.close()
            plt.show()


def remove_noise_from_feature(data: pd.DataFrame, imputed='max_spectral_centroid', leave_predicate=lambda x: x > 0.1 and x < 0.9):
    if imputed not in data.columns:
        return

    groups = data.groupby('target')
    per_target_medians = {}
    for target, group_df in groups:
        set_for_median = group_df[imputed].tolist()
        set_for_median = filter(leave_predicate, set_for_median)
        per_target_medians[target] = np.median(list(set_for_median))
        for index, row in group_df.iterrows():
            if not leave_predicate(row[imputed]):
                data.at[index, imputed] = per_target_medians[target]


def compare_models(data: pd.DataFrame):
    # Initialize the models
    modelSVC = SVC()  # works 0.55
    modelRFC = RandomForestClassifier()  # works 0.65
    modelGBC = GradientBoostingClassifier()  # works 0.7

    accuracies = [[], [], []]
    for i in range(10):
        X_train, y_train, X_test, y_test = get_sets(data, seed=i)

        modelSVC.fit(X_train, y_train)
        y_pred = modelSVC.predict(X_test)
        accuracies[0].append(accuracy_score(y_test, y_pred))

        modelRFC.fit(X_train, y_train)
        y_pred = modelRFC.predict(X_test)
        accuracies[1].append(accuracy_score(y_test, y_pred))

        modelGBC.fit(X_train, y_train)
        y_pred = modelGBC.predict(X_test)
        accuracies[2].append(accuracy_score(y_test, y_pred))

    print('Accuracies:')
    print('SVC', np.mean(accuracies[0]))
    print('RFC', np.mean(accuracies[1]))
    print('GBC', np.mean(accuracies[2]))


def compare_imputed(data: pd.DataFrame):
    X_train, y_train, X_test, y_test = get_sets(data)

    # Initialize the model

    model = GradientBoostingClassifier()  # works 0.7

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy non-imputed: {accuracy}')


    remove_noise_from_feature(data)
    X_train, y_train, X_test, y_test = get_sets(data)
    model.fit(X_train, y_train)

    #Predict
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy imputed: {accuracy}')


def make_predictions_on_data(data: pd.DataFrame, visualise=False, seed=0):
    remove_noise_from_feature(data)
    X_train, y_train, X_test, y_test = get_sets(data, seed=seed)

    # Initialize the model
    model = GradientBoostingClassifier()  # works 0.7

    # Fit the model
    model.fit(X_train, y_train)

    # Tested Prediction
    y_pred = model.predict(X_test)

    if visualise:
        visualise_conf_matrix(y_test, y_pred, model)

    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy}')

    return accuracy


def make_model_on_full_set(data: pd.DataFrame):
    remove_noise_from_feature(data)
    X, y = data.drop(['target'], axis=1, inplace=False).copy(), data['target']

    model = GradientBoostingClassifier()  # works 0.7

    # Fit the model
    model.fit(X, y)
    return model


def compare_features_drop_mean_deltas_powerset(data: pd.DataFrame, seed_range=1):
    feature_labels = project_globals.FEATURE_LABELS
    accuracy_deltas = {}
    for seed in range(seed_range):
        entire_set_accuracy = make_predictions_on_data(data, seed=seed)

        for i in range(len(feature_labels)):
            print(i)
            if i > len(feature_labels) - 3:
                continue
            for label in itertools.combinations(feature_labels, i+1):
                data_dropped = data.drop([*label], axis=1, inplace=False)
                accuracy = make_predictions_on_data(data_dropped, seed=seed)
                if label in accuracy_deltas:
                    accuracy_deltas[label].append(accuracy - entire_set_accuracy)
                else:
                    accuracy_deltas[label] = [accuracy - entire_set_accuracy]

    accuracy_deltas_mean = {k: np.mean(v) for k, v in accuracy_deltas.items()}

    largest_items = dict(heapq.nlargest(5, accuracy_deltas_mean.items(), key=lambda item: item[1]))
    smallest_items = dict(heapq.nsmallest(5, accuracy_deltas_mean.items(), key=lambda item: item[1]))
    print(largest_items)
    print(smallest_items)

def compare_features_drop_mean_deltas_sub_powerset(data: pd.DataFrame, seed_range=1, up_to_how_many_features_to_test=1):
    feature_labels = project_globals.FEATURE_LABELS
    accuracy_deltas = {}
    for seed in range(seed_range):
        entire_set_accuracy = make_predictions_on_data(data, seed=seed)

        for i in range(len(feature_labels)):
            print(i)
            if i > len(feature_labels) - 2 or i >= up_to_how_many_features_to_test:
                continue
            for label in itertools.combinations(feature_labels, i+1):
                data_dropped = data.drop([*label], axis=1, inplace=False)
                accuracy = make_predictions_on_data(data_dropped, seed=seed)
                if label in accuracy_deltas:
                    accuracy_deltas[label].append(accuracy - entire_set_accuracy)
                else:
                    accuracy_deltas[label] = [accuracy - entire_set_accuracy]

    accuracy_deltas_mean = {k: np.mean(v) for k, v in accuracy_deltas.items()}

    largest_items = dict(heapq.nlargest(5, accuracy_deltas_mean.items(), key=lambda item: item[1]))
    smallest_items = dict(heapq.nsmallest(5, accuracy_deltas_mean.items(), key=lambda item: item[1]))
    print(largest_items)
    print(smallest_items)


def compare_features_drop_mean_deltas_specified_set(data: pd.DataFrame, dropped_sets: list[tuple[str, ...]], seed_range=30):
    accuracy_deltas = {}
    for seed in range(seed_range):
        entire_set_accuracy = make_predictions_on_data(data, seed=seed)

        for labels in dropped_sets:
            data_dropped = data.drop([*labels], axis=1, inplace=False)
            accuracy = make_predictions_on_data(data_dropped, seed=seed)
            if labels in accuracy_deltas:
                accuracy_deltas[labels].append(accuracy - entire_set_accuracy)
            else:
                accuracy_deltas[labels] = [accuracy - entire_set_accuracy]

    accuracy_deltas_mean = {k: np.mean(v) for k, v in accuracy_deltas.items()}

    print(accuracy_deltas_mean)


def check_for_similarities_in_genres(seed=0):
    remove_noise_from_feature(data)
    X_train, y_train, X_test, y_test = get_sets(data, seed=seed)

    # Initialize the model
    model = GradientBoostingClassifier()  # works 0.7

    # Fit the model
    model.fit(X_train, y_train)

    # Tested Prediction
    y_pred = model.predict(X_test)
    # Determine the unique labels and their corresponding indices used by the classifier
    labels = model.classes_
    print(labels)

    # Compute the confusion matrix with the correct order of labels
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Finding the count of the mistaken genres
    # Extract the mistaken genres counts
    mistaken_pairs = Counter()
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                actual_genre = labels[i]
                predicted_genre = labels[j]
                # Use frozenset to treat (actual_genre, predicted_genre) and (predicted_genre, actual_genre) as the same
                pair = frozenset([actual_genre, predicted_genre])
                mistaken_pairs[pair] += cm[i, j]

    # Sort mistaken pairs by the count in descending order
    sorted_mistaken_pairs = sorted(mistaken_pairs.items(), key=lambda item: item[1], reverse=True)

    for p in sorted_mistaken_pairs:
        print(list(p[0])[0]+' and '+list(p[0])[1], p[1])


def examine_qotsa():
    model = make_model_on_full_set(data)
    qotsa.drop(['target'], axis=1, inplace=True)
    y_pred = model.predict(qotsa)

    # Count the occurrences of each string
    counts = Counter(y_pred)

    # Extract the keys and values
    labels, values = zip(*counts.items())

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Predicated genres occurrences for QOTSA')
    plt.xticks(rotation=90)  # Rotate x labels if they are long
    plt.tight_layout()  # Adjust layout to make room for x labels

    # Show the histogram
    plt.show()

    for song, predicated in zip(qotsa.index, y_pred):
        print(song, predicated)


if __name__ == '__main__':
    # !!! HERE ARE ALL THE FUNCTIONS NECESSARY FOR CARRYING OUT THE STEPS TAKEN IN THE PROJECT AFTER EXTRACTION!!! #


    # PREPARING THE DATA (most of the next ones don't work without this one)
    data = get_data_as_df()
    qotsa = data[data['target'] == 'Queens of the Stone Age']
    data = data[data['target'] != 'Queens of the Stone Age']
    data = data[data['target'] != 'Blues']

    # MAKING THE PLOTS
    # visualise_feature_distribution_in_genres(data)

    # IMPROVING THE MODEL
    # compare_models(data)
    # compare_imputed(data)

    # CHECKING HOW THE MODEL IS CLASSIFYING
    # make_predictions_on_data(data, visualise=True)

    # CHECKING WHICH FEATURE DROP CAUSE ACCURACY DECREASE
    # compare_features_drop_mean_deltas_powerset(data)

    # CHECKING WHICH FEATURES SET DROP IN A POWERSET ON LABELS SET CAUSE ACCURACY DECREASE
    # compare_features_drop_mean_deltas_powerset(data)

    # CHECKING WETHER THE ABOVE PILOT WORKS ON DIFERENT SEEDS WITH MEAN
    # compare_features_drop_mean_deltas_specified_set(data,
    #                                                 list(project_globals.SET_OF_REMOVED_FEATURES_THAT_IMPROVE_MODEL.keys()))

    # CHECKING FOR UNDERLAYING SIMILARITIES BETWEEN GENRES BASED ON THE MODEL
    # check_for_similarities_in_genres()

    # CHECKING WHAT THE MODEL THINKS ABOUT QOTSA
    examine_qotsa()
