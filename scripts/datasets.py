import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import torch
import logging
from defer.datasetsdefer.basedataset import BaseDataset


class Dataset(BaseDataset):
    """Base Dataset for L2D"""

    def __init__(
        self,
        X,
        y,
        human_predictions,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
        random_seed=42,
    ):
        """

        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        expert: expert module used to generate human predictions
        """
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.random_seed = random_seed
        self.generate_data(X, y, human_predictions)

    def generate_data(self, X, y, human_predictions):
        """
        Load and process data for training, validation and test sets
        """
        # Convert to PyTorch tensors
        human_predictions = torch.tensor(human_predictions)
        y = torch.tensor(y)
        X = torch.from_numpy(X).float()

        self.total_samples = len(X)
        self.d = len(X[0])

        # Split into train/val/test sets with consistent random seed
        train_size = int(self.train_split * self.total_samples)
        val_size = int(self.val_split * self.total_samples)
        test_size = self.total_samples - train_size - val_size

        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(
            X,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        self.train_y, self.val_y, self.test_y = torch.utils.data.random_split(
            y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        self.train_h, self.val_h, self.test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        logging.info(f"Train size: {len(self.train_x)}")
        logging.info(f"Val size: {len(self.val_x)}")
        logging.info(f"Test size: {len(self.test_x)}")

        # Create TensorDatasets
        self.data_train = torch.utils.data.TensorDataset(
            self.train_x.dataset[self.train_x.indices],
            self.train_y.dataset[self.train_y.indices],
            self.train_h.dataset[self.train_h.indices],
        )
        self.data_val = torch.utils.data.TensorDataset(
            self.val_x.dataset[self.val_x.indices],
            self.val_y.dataset[self.val_y.indices],
            self.val_h.dataset[self.val_h.indices],
        )
        self.data_test = torch.utils.data.TensorDataset(
            self.test_x.dataset[self.test_x.indices],
            self.test_y.dataset[self.test_y.indices],
            self.test_h.dataset[self.test_h.indices],
        )

        # Create DataLoaders
        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=True
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=True
        )


def adult_load_and_preprocess():
    # The files adult.data and adult.test were extracted from the file 'adult.zip',
    # available at the UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/2/adult

    data_split = pd.read_csv("data/adult/adult.data", header=None)

    with open("data/adult/adult.test", "r") as file:
        first_line = file.readline()
        print("Line to drop: ", first_line)

    test_split = pd.read_csv("data/adult/adult.test", header=None, skiprows=1)

    # Column names as shown in https://archive.ics.uci.edu/dataset/2/adult
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    data_split.columns = column_names
    test_split.columns = column_names

    # And, for simplicity, we drop the missing values identified by the ' ?' string.
    data_split.replace(" ?", np.nan, inplace=True)
    data_split.dropna(inplace=True)

    test_split.replace(" ?", np.nan, inplace=True)
    test_split.dropna(inplace=True)

    print("Training instances last index:", len(data_split) - 1)
    # Instances up to the index 30162 (not including) will be training instances
    # Instances from the index 30162 will be training instances

    combined_data = pd.concat([data_split, test_split], ignore_index=True)
    combined_data.reset_index(drop=True, inplace=True)

    # Now we convert the dataset's label to a binary one
    combined_data["income"] = combined_data["income"].apply(
        lambda x: 1 if (x == " >50K") or (x == " >50K.") else 0
    )
    # We will also create a new column which binarizes the "race" column into "White" and "Non-White" citizens.
    combined_data["binarized_race"] = combined_data["race"].apply(
        lambda x: "White" if x == " White" else "Non-White"
    )
    # and we reorder the columns so that binarized_race appears next to the race column
    cols = column_names[:9] + ["binarized_race"] + column_names[9:]
    combined_data = combined_data[cols]

    # We also drop the column "education" as it is redundant due to the "education_num" column, as well as the column "race", which we have binarized
    combined_data.drop(columns=["education", "race"], inplace=True)
    # Furthermore, if we treat the education as a numerical value, we can induce bias, for example, more false positives for people with a higher education level.
    # We will alter this column so it does not serve as a proxy for sex
    combined_data["relationship"] = combined_data["relationship"].apply(
        lambda x: " Married" if (x == " Wife") or (x == " Husband") else x
    )
    # Let us start by first defining which columns correspond to the categorical variables
    categorical_cols = [
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "binarized_race",
        "sex",
        "native_country",
    ]
    # Fit the encoder to the data and transform the specified columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_data[categorical_cols] = encoder.fit_transform(
        combined_data[categorical_cols]
    )
    # Data has to be scaled
    y = combined_data["income"].values
    combined_data = combined_data.drop(["income"], axis=1)
    feature_names = combined_data.columns
    # Normalize data
    scaler = StandardScaler()
    combined_data[feature_names] = scaler.fit_transform(combined_data[feature_names])
    return combined_data, y, feature_names


def plant_load_and_preprocess():
    df = pd.read_csv("data/plant_disease.csv")
    # Data has to be scaled
    y = df["disease_present"].values
    df = df.drop(["disease_present"], axis=1)
    feature_names = df.columns
    # Normalize data
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    return df, y, feature_names


def wine_load_and_preprocess():
    """
    https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
    """
    df = pd.read_csv("data/wine.csv")
    df["quality"] = df["quality"].apply(lambda x: 1 if (x > 5) else 0)
    df = df.drop("Id", axis=1)
    # Data has to be scaled
    y = df["quality"].values
    df = df.drop(["quality"], axis=1)
    feature_names = df.columns
    # Normalize data
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    return df, y, feature_names


def diabetes_load_and_preprocess():
    df = pd.read_csv("data/diabetes.csv")
    # Data has to be scaled
    y = df["Outcome"].values
    df = df.drop(["Outcome"], axis=1)
    feature_names = df.columns
    # Normalize data
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    return df, y, feature_names


def cancer_load_and_preprocess():
    df = pd.read_csv("data/breast_cancer.csv")
    df["diagnosis"] = df["diagnosis"].apply(
        lambda x: 1 if (x == "M") or (x == " M") else 0
    )
    df = df.drop(["id", "Unnamed: 32"], axis=1)
    const = df.nunique()[df.nunique() <= 1].index
    df.drop(columns=const, inplace=True)

    y = df["diagnosis"].values
    df = df.drop(["diagnosis"], axis=1)
    feature_names = df.columns
    # Normalize data
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])

    return df, y, feature_names


def star_load_and_preprocess(filepath="data/star_classification.csv"):
    """
    Loads and preprocesses the star classification dataset, returning a scaled DataFrame.

    - The internal logic is made more robust to prevent NaN values during scaling.
    - The function signature and return types are identical to the original:
      (pandas.DataFrame, numpy.ndarray, list)

    Returns:
        df_scaled (pd.DataFrame): The scaled feature data in a DataFrame.
        y (np.ndarray): The binarized target labels.
        feature_names (list): The names of the feature columns.
    """
    # 1. Load Data
    df = pd.read_csv(filepath)

    # 2. Filter out 'QSO' class and create a copy to avoid warnings
    df = df.loc[df["class"] != "QSO"].copy()

    # 3. Binarize target and separate features (X) from target (y)
    df["class"] = df["class"].map({"STAR": 1, "GALAXY": 0})
    y = df["class"].values
    X = df.drop("class", axis=1)

    # 4. Drop non-feature columns from the features DataFrame (X)
    ids_to_drop = [
        "obj_ID",
        "run_ID",
        "rerun_ID",
        "cam_col",
        "field_ID",
        "spec_obj_ID",
        "fiber_ID",
        "MJD",
    ]
    # Drop only the columns that actually exist in the DataFrame
    cols_that_exist = [col for col in ids_to_drop if col in X.columns]
    X.drop(columns=cols_that_exist, inplace=True)

    # Ensure all data is numeric, coercing errors. This step is fine on X.
    X = X.apply(pd.to_numeric, errors="coerce")

    # IMPORTANT: Find and remove constant columns (the source of NaNs)
    # This must be done *before* scaling.
    const_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if const_cols:
        print(f"Removing constant columns: {const_cols}")
        X.drop(columns=const_cols, inplace=True)

    # Store the final feature names after all dropping has occurred
    feature_names = X.columns.tolist()

    # 5. Scale the features
    scaler = StandardScaler()
    # scaler.fit_transform returns a NumPy array
    X_scaled_np = scaler.fit_transform(X)

    # --- This is the key change to match your return type ---
    # 6. Re-create the DataFrame from the scaled NumPy array
    # This puts the scaled data back into the DataFrame format you expect.
    df_scaled = pd.DataFrame(X_scaled_np, columns=feature_names)

    # 7. Final Sanity Check
    if df_scaled.isnull().sum().sum() > 0:
        print("\nWARNING: NaN values were found in the final scaled DataFrame!")
    else:
        print("\nPreprocessing complete. No NaN values found.")

    # 8. Return the exact types you need for your pipeline
    return df_scaled, y, feature_names


def telescope_load_and_preprocess():
    # https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
    column_names = [
        "fLength",  # major axis of ellipse [mm],
        "fWidth",  # minor axis of ellipse [mm]
        "fSize",  # 10-log of sum of content of all pixels [in #phot]
        "fConc",  # ratio of sum of two highest pixels over fSize  [ratio]
        "fConc1",  # ratio of highest pixel over fSize  [ratio]
        "fAsym",  # distance from highest pixel to center, projected onto major axis [mm]
        "fM3Long",  # 3rd root of third moment along major axis  [mm]
        "fM3Trans",  # 3rd root of third moment along minor axis  [mm]
        "fAlpha",  # angle of major axis with vector to origin [deg]
        "fDist",  # distance from origin to center of ellipse [mm]
        "class",  # gamma (signal), hadron (background)
    ]
    df = pd.read_csv("data/telescope/telescope.data", header=None, names=column_names)
    df["class"] = df["class"].apply(lambda x: 1 if (x == "g") else 0)
    y = df["class"].values
    df = df.drop(["class"], axis=1)
    feature_names = df.columns
    # Normalize data
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    return df, y, feature_names
