import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        self.constant_features = []
        self.skewed_features = []
        self.transformed_features = []

    # -----------------------------------------
    # 1. Identify binary and continuous features
    # -----------------------------------------

    def classify_features(self, X):

        binary_features = []
        continuous_features = []
        categorical_features = []

        for column in X.columns:

            series = X[column].dropna()

            if series.empty:
                continue

            unique_values = set(series.unique())
            unique_count = len(unique_values)

            # Boolean features
            if pd.api.types.is_bool_dtype(series):
                binary_features.append(column)

            # Binary features containing only 0 and 1
            elif (
                unique_count == 2
                and unique_values.issubset({0, 1})
            ):
                binary_features.append(column)

            # Text and categorical features
            elif (
                pd.api.types.is_object_dtype(series)
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype
                )
            ):
                categorical_features.append(column)

            # Remaining numerical features
            elif pd.api.types.is_numeric_dtype(series):
                continuous_features.append(column)

            else:
                categorical_features.append(column)

        return {
            "binary": binary_features,
            "continuous": continuous_features,
            "categorical": categorical_features
        }

    # -----------------------------------------
    # 2. Detect skewed continuous features
    # -----------------------------------------

    def detect_skewed_features(self, X):

        feature_types = self.classify_features(X)

        numerical_columns = feature_types["continuous"]

        skewed_features = []

        print("\n" + "=" * 50)
        print("SKEWED FEATURE DETECTION")
        print("=" * 50)

        for column in numerical_columns:

            series = X[column].dropna()

            # Skip constant features
            if series.nunique() <= 1:
                continue

            skewness = series.skew()

            if pd.notna(skewness) and abs(skewness) > 1:

                skewed_features.append(column)

                print(
                    f"{column}: skewness = "
                    f"{skewness:.3f}"
                )

        if not skewed_features:
            print(
                "No highly skewed continuous features found."
            )

        self.skewed_features = skewed_features

        return skewed_features

    # -----------------------------------------
    # 3. Apply log transformation
    # -----------------------------------------

    def log_transform(self, X_train, X_test):

        X_train = X_train.copy()
        X_test = X_test.copy()

        if list(X_train.columns) != list(X_test.columns):
            raise ValueError(
                "Training and testing columns must match."
            )

        skewed_features = self.detect_skewed_features(
            X_train
        )

        transformed_features = []

        for column in skewed_features:

            train_values = X_train[column].dropna()

            # Log1p requires values greater than or equal to zero.
            if (train_values >= 0).all():

                X_train[column] = np.log1p(
                    X_train[column]
                )

                # Apply the same transformation to test data.
                # Do not refit or redetect skewness on test data.
                X_test[column] = np.log1p(
                    X_test[column]
                )

                transformed_features.append(column)

                print(
                    f"Log transformation applied: {column}"
                )

            else:

                print(
                    f"Log transformation skipped for {column}: "
                    "negative values detected."
                )

        self.transformed_features = transformed_features

        return X_train, X_test

    # -----------------------------------------
    # 4. Remove constant features
    # -----------------------------------------

    def remove_constant_features(
        self,
        X_train,
        X_test
    ):

        X_train = X_train.copy()
        X_test = X_test.copy()

        if list(X_train.columns) != list(X_test.columns):
            raise ValueError(
                "Training and testing columns must match."
            )

        constant_columns = [
            column
            for column in X_train.columns
            if X_train[column].nunique(dropna=False) <= 1
        ]

        print("\n" + "=" * 50)
        print("CONSTANT FEATURE REMOVAL")
        print("=" * 50)

        if constant_columns:

            print(f"Removing: {constant_columns}")

            X_train = X_train.drop(
                columns=constant_columns
            )

            X_test = X_test.drop(
                columns=constant_columns
            )

        else:

            print("No constant features found.")

        self.constant_features = constant_columns

        return X_train, X_test

    # -----------------------------------------
    # 5. Run feature engineering
    # -----------------------------------------

    def run(self, X_train, X_test):

        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)

        # Remove constant features first.
        X_train, X_test = self.remove_constant_features(
            X_train,
            X_test
        )

        # Detect skewness only in appropriate features.
        X_train, X_test = self.log_transform(
            X_train,
            X_test
        )

        print("\nFeature engineering completed.")

        print(
            f"Training features: {X_train.shape[1]}"
        )

        print(
            f"Testing features: {X_test.shape[1]}"
        )

        return X_train, X_test