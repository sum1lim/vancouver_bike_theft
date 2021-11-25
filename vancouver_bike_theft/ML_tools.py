import sys
import csv
import yaml
import smogn
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler


def config_parser(ml_config):
    """
    Parse the parameters defined in the configuration part.
    """
    num_epochs, hidden_size, verbosity, K = 100, None, 2, 1

    if ml_config:
        stream = open(ml_config, "r")
        config_dict = yaml.safe_load(stream)
    else:
        return num_epochs, hidden_size, verbosity, K

    num_epochs, hidden_size, verbosity, K, kernel_size = None, None, None, None, None
    if "configuration" in config_dict.keys():
        params = config_dict["configuration"]
        train_data = params["train_data"]
        test_data = params["test_data"]
        if "epochs" in params.keys():
            num_epochs = params["epochs"]
        if "hidden_size" in params.keys():
            hidden_size = params["hidden_size"]
        if "verbosity" in params.keys():
            verbosity = params["verbosity"]
        if "K-fold" in params.keys():
            K = params["K-fold"]
        if "kernel_size" in params.keys():
            kernel_size = params["kernel_size"]
        if "min_num_points" in params.keys():
            min_num_points = params["min_num_points"]
        else:
            min_num_points = 0
        if "smote" in params.keys():
            smote = params["smote"]
        else:
            smote = False

    return (
        train_data,
        test_data,
        num_epochs,
        hidden_size,
        verbosity,
        K,
        kernel_size,
        min_num_points,
        smote,
    )


def calculate_hidden_layer_size(input_layer_size, output_layer_size, user_defined=None):
    """
    Calculate the hidden layer size if user did not define the size
    """
    if user_defined == None:
        hidden_layer_size = ((input_layer_size + output_layer_size) * 2) // 3
    else:
        hidden_layer_size = user_defined

    if hidden_layer_size > 2 * input_layer_size:
        hidden_layer_size = 2 * input_layer_size

    if hidden_layer_size < 2:
        hidden_layer_size = 2

    return hidden_layer_size


def process_data(
    data_file,
    ml_config=None,
    regression=True,
    min_num_points=0,
    smote=False,
):
    """
    Merge labels and/or select feautres for learning
    based on the user definition in the configuration file
    """
    if ml_config:
        stream = open(ml_config, "r")
        config_dict = yaml.safe_load(stream)
    else:
        config_dict = None

    dataframe = pandas.read_csv(data_file, header=0, index_col=False)

    dataframe.drop(
        columns=["id", "left", "top", "right", "bottom", "NUM_Thefts"], inplace=True
    )

    if config_dict:
        if "NUM_Thefts" in config_dict.keys():
            try:
                for idx, data_range in config_dict["NUM_Thefts"].items():
                    min, max = data_range
                    dataframe["NUM_Thefts"] = np.where(
                        dataframe["NUM_Thefts"].between(min, max),
                        -idx,
                        dataframe["NUM_Thefts"],
                    )
                dataframe["NUM_Thefts"] = -dataframe["NUM_Thefts"]
            except KeyError:
                print("Error in configuration format", file=sys.stderr)
                sys.exit(1)

        if "features" in config_dict.keys():
            try:
                dataframe.drop(
                    dataframe.columns.difference(
                        config_dict["features"] + ["NUM_Thefts"]
                    ),
                    1,
                    inplace=True,
                )
            except KeyError:
                print("Error in configuration format", file=sys.stderr)
                sys.exit(1)

    # SMOTE over/under-sampling
    if smote:
        print(
            f"Before SMOTE\n Box Stats: {smogn.box_plot_stats(dataframe['NUM_Thefts'])['stats']}",
            file=sys.stdout,
        )
        print(f" Number of samples: {dataframe.shape[0]}\n", file=sys.stdout)
        dataframe = dataframe.dropna()
        dataframe.reset_index(drop=True, inplace=True)
        while True:
            try:
                dataframe = smogn.smoter(
                    data=dataframe, y="NUM_Thefts", samp_method="extreme"
                )
                break
            except ValueError:
                continue
        dataframe = dataframe.dropna()
        dataframe.reset_index(drop=True, inplace=True)
        print(
            f"After SMOTE\n Box Stats: {smogn.box_plot_stats(dataframe['NUM_Thefts'])['stats']}",
            file=sys.stdout,
        )
        print(f" Number of samples: {dataframe.shape[0]}\n", file=sys.stdout)

    dataset = dataframe.values
    print(f"Size of dataset: {dataset.shape}", file=sys.stderr)
    X = dataset[:, 1:].astype(float)
    Y = dataset[:, 0]

    if regression:
        return X, Y
    else:
        print(f"Before oversampling: {Counter(Y)}", file=sys.stdout)
        oversample = RandomOverSampler(sampling_strategy="not majority")
        X, Y = oversample.fit_resample(X, Y)
        print(f"After oversampling: {Counter(Y)}", file=sys.stdout)

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        print(f"Classes: {encoder.classes_}", file=sys.stdout)

        return X, encoded_Y, encoder.classes_


def learning_curve(model_hist, result_dir, iter):
    """
    This chunk of code is sourced and modified from Machin Learning Mastery [1].

    [1] J. Brownlee, “Display Deep Learning Model Training History in Keras,” Machine Learning Mastery, 03-Oct-2019. [Online].
    Available: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/. [Accessed: 16-Jul-2021].
    """
    # summarize history for accuracy
    if "accuracy" in model_hist.keys() and "val_accuracy" in model_hist.keys():
        plt.plot(model_hist["accuracy"])
        plt.plot(model_hist["val_accuracy"])
        plt.title(f"Learning Curve (Fold #: {iter+1})")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")
        plt.savefig(f"{result_dir}/learning_curve_{iter+1}.png")
        plt.clf()
    # summarize history for loss
    plt.plot(model_hist["loss"])
    plt.plot(model_hist["val_loss"])
    plt.title(f"Loss Curve (Fold #: {iter+1})")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.savefig(f"{result_dir}/loss_curve_{iter+1}.png")
    plt.clf()


def tr_val_split(K, X_tr, Y_tr):
    if K < 2:
        K = 2
    kfold = KFold(n_splits=K, shuffle=True)
    tr_val_pairs = kfold.split(X_tr, Y_tr)

    return tr_val_pairs


def regression_plots(Y_expert, Y_pred, abs_error, result_dir, iter):
    maximum_val = max([max(Y_expert), max(Y_pred)])

    # pred vs expert
    pred_df = {"expert": Y_expert, "pred": Y_pred}
    plt.scatter(Y_expert, Y_pred)
    plt.title(f"Prediction VS Expert (Fold #: {iter+1})")
    plt.ylabel("Predcited Value")
    plt.xlabel("Expert Value")
    sns.kdeplot(data=pred_df, x="expert", y="pred", color="red")
    xpoints = ypoints = plt.xlim(0, maximum_val + maximum_val / 10)
    plt.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=3, scalex=False, scaley=False
    )
    plt.axis("square")
    plt.savefig(f"{result_dir}/pred_expert_{iter+1}.png")
    plt.clf()

    # absolute errors
    plt.scatter(Y_expert, abs_error)
    plt.title(f"Absolute Errors (Fold #: {iter+1})")
    plt.ylabel("Absolute Error")
    plt.xlabel("Expert Value")
    err_df = {"expert": Y_expert, "error": abs_error}
    sns.kdeplot(data=err_df, x="expert", y="error", color="red")
    plt.savefig(f"{result_dir}/abs_errs_{iter+1}.png")
    plt.clf()
