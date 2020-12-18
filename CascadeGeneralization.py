import numpy as np
import pandas as pd
import itertools

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# run the single specified model with passed data
def run_model(X_train, y_train, X_test, model_name):

    # create specified model
    if model_name == "linear discriminant":
        this_classifier = LinearDiscriminantAnalysis()
    elif model_name == "naive bayes":
        this_classifier = GaussianNB()
    elif model_name == "decision tree":
        this_classifier = tree.DecisionTreeClassifier()
    elif model_name == "NN":
        this_classifier = MLPClassifier(solver='adam', learning_rate_init=0.01, max_iter=1000, alpha=1e-5,
                                        hidden_layer_sizes=(5, 2), random_state=1)

    # train selected model and make predictions
    this_classifier.fit(X_train, y_train)
    predictions = this_classifier.predict(X_test)

    # append class probability distributions to training & test sets
    X_train_new = np.append(X_train, this_classifier.predict_proba(X_train), 1)
    X_test_new = np.append(X_test, this_classifier.predict_proba(X_test), 1)

    return X_train_new, X_test_new, predictions


# run Cascade Generalization with specified model configuration
def get_stack_accuracy(model_combination, xTrain, xTest, yTrain, yTest):

    # generalize data sets with passed model stack
    for model in model_combination:
        xTrain, xTest, predictions = run_model(xTrain, yTrain, xTest, model)

    # return accuracy of this configuration
    return accuracy_score(yTest, predictions)


# convert categorical attributes to numerical data
def encode_categorical_features(this_dataset):

    for column in this_dataset:
        if this_dataset[column].dtype == 'object':
            this_dataset[column] = this_dataset[column].astype('category')
            this_dataset[column] = this_dataset[column].cat.codes
    return this_dataset


# write results to text file w/ formatting
def write_results_to_file():
    with open('results.txt', 'w') as file:

        # write win/accuracy results for each dataset
        for item in dataset_results.items():
            file.write(
                "-------------------------------------------------------\n" + str(item[0]).upper() + " DATASET:\n\n")
            file.write("\tWins\t\tAvg. Acc.\tConfiguration\n")
            for key, value in item[1].items():
                file.write("\t" + str(value[0]) + "\t\t" + str(value[1]) + "%\t\t" + str(key) + "\n")

        # write the overall win/rank results for each model configuration
        file.write("-------------------------------------------------------\n")
        file.write("OVERALL WINS (INCLUDING TIES) & AVG. RANK FOR EACH CONFIGURATION:\n\n")
        file.write("\tWins\t\tAvg. Rank\tConfiguration\n")
        num_trials = len(datasets) * iterations
        for key, value in combo_total_wins_and_rank.items():
            file.write(
                "\t" + str(value[0]) + "/" + str(num_trials) + "\t\t" + str(round(value[1] / num_trials, 2)) + "\t\t"
                + str(key) + "\n")


# load & split desired dataset
def load_data(desired_dataset):

    # Adult Dataset (48842 Ex.)
    if desired_dataset == "Adult":
        cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                "salary"]
        data = pd.read_csv('adult.data', names=cols, sep=',')
        data = encode_categorical_features(data)
        features = data.drop('salary', axis=1)
        target = data['salary']

    # Bank Dataset (45211 Ex.)
    elif desired_dataset == "Bank":
        data = pd.read_csv('bank-additional-full.csv', sep=';')
        data = encode_categorical_features(data)
        features = data.drop('y', axis=1)
        target = data['y']

    # Sat-Image Dataset (6435 Ex.)
    elif desired_dataset == "Satimage":
        cols = list(range(37))
        data_training = pd.read_csv('sat.trn', names=cols, sep=' ')
        data_testing = pd.read_csv('sat.tst', names=cols, sep=' ')
        data = data_training.append(data_testing)
        features = data.drop(36, axis=1)
        target = data[36]

    # Digits Dataset (1797 Ex.)
    elif desired_dataset == "Digits":
        digits_data = load_digits()
        return train_test_split(digits_data.data, digits_data.target, stratify=digits_data.target)

    # Credit Dataset (690 Ex.)
    elif desired_dataset == "Credit":
        cols = list(range(16))
        data = pd.read_csv('credit.data', names=cols, sep=',')
        data = encode_categorical_features(data)
        features = data.drop(15, axis=1)
        target = data[15]

    # Breast Cancer Dataset (569 Ex.)
    elif desired_dataset == "Cancer":
        cancer_data = load_breast_cancer()
        return train_test_split(cancer_data.data, cancer_data.target, stratify=cancer_data.target)

    # Banding Dataset (512 Ex.)
    elif desired_dataset == "Band":
        cols = list(range(40))
        data = pd.read_csv('bands.data', names=cols, sep=',')
        data = encode_categorical_features(data)
        features = data.drop(39, axis=1)
        target = data[39]

    # Voting Records Dataset (435 Ex.)
    elif desired_dataset == "Votes":
        cols = list(range(17))
        data = pd.read_csv('house-votes-84.data', names=cols, sep=',')
        data = encode_categorical_features(data)
        features = data.drop(16, axis=1)
        target = data[16]

    # Glass Dataset (214 Ex.)
    elif desired_dataset == "Glass":
        cols = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Glass Type']
        data = pd.read_csv('glass.data', names=cols, sep=',').drop('ID', axis=1)
        features = data.drop('Glass Type', axis=1)
        target = data['Glass Type']

    # Wine Dataset (178 Ex.)
    elif desired_dataset == "Wine":
        wine_data = load_wine()
        return train_test_split(wine_data.data, wine_data.target, stratify=wine_data.target)

    return train_test_split(features, target, test_size=0.3, random_state=42)


# sort a given dict by the given value index
def sort_dictionary(dictionary, index):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda val: val[1][index], reverse=True)}


# run all model stack permutations on this dataset 'iterations'-times
def run_cascade_generalization_variants(current_dataset):

    stack_results = {}      # results for each model configuration

    # run entire CG 'iterations'-times
    for i in range(0, iterations):

        highest_accuracy = 0.0                                      # highest accuracy for this iteration
        best_combo = []                                             # model combo with highest accuracy
        tied_winners = []                                           # keep track of current ties in highest accuracy
        xTrain, xTest, yTrain, yTest = load_data(current_dataset)   # load data from current specified dataset

        # for each model permutation
        for combination in model_combos:

            # get accuracy of this model combination on current dataset
            accuracy = get_stack_accuracy(combination, xTrain, xTest, yTrain, yTest)

            # if new highest accuracy, update
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_combo = combination
                tied_winners = [best_combo]

            # else if a tie, add to tie array
            elif accuracy == highest_accuracy:
                tied_winners.append(combination)

            # accumulate accuracy score for this combination (to be averaged later)
            if combination in stack_results:
                stack_results[combination][1] += accuracy
            else:
                stack_results[combination] = [0, accuracy]

        # increment win count of winning combo of this iteration
        stack_results[best_combo][0] += 1

        # sort model combinations by current accuracy
        stack_results = sort_dictionary(stack_results, 1)

        # add to rank of all combos (to be averaged later)
        rank = 1
        for this_key in stack_results.keys():
            combo_total_wins_and_rank[this_key][1] += rank
            rank += 1

        # increment wins including ties
        for tied_combo in tied_winners:
            combo_total_wins_and_rank[tied_combo][0] += 1

    # calculate average accuracy for each combination
    for result in stack_results.values():
        result[1] = round(((result[1] / iterations) * 100), 2)

    return stack_results


iterations = 50                     # number of times CG permutations will be run for each dataset
model_combos = []                   # array for all classifier permutations
dataset_results = {}                # dict to accumulate results for each dataset
combo_total_wins_and_rank = {}      # dict to accrue results across all datasets

# names of datasets to be analyzed
datasets = ["Bank", "Adult", "Glass", "Cancer", "Wine", "Credit", "Band", "Satimage", "Votes"]

# create all permutations of model stacks (size 1-4)
for r in range(1, 5):
    model_combos.extend(list(itertools.permutations(["linear discriminant", "naive bayes", "decision tree", "NN"], r)))

# initialize total wins/rank dict to 0s
for combo in model_combos:
    combo_total_wins_and_rank[combo] = [0, 0]

# run CG with all permutations on all datasets, 'iterations'-times
for dataset in datasets:
    dataset_results[dataset] = run_cascade_generalization_variants(dataset)

# sort overall wins dict by number of wins, write all results to file
combo_total_wins_and_rank = sort_dictionary(combo_total_wins_and_rank, 0)
write_results_to_file()

