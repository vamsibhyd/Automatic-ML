import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    # -----------------------------------------
    # 1. Detect skewed numerical features
    # -----------------------------------------

    def detect_skewed_features(self, X):

        numerical_columns = X.select_dtypes(
            include=["number"]
        ).columns

        skewed_features = []

        for column in numerical_columns:

            skewness = X[column].skew()

            if abs(skewness) > 1:

                skewed_features.append(column)

        print("\n" + "=" * 50)
        print("SKEWED FEATURE DETECTION")
        print("=" * 50)

        if skewed_features:

            print("\nHighly skewed features:")

            for column in skewed_features:

                print(
                    f"{column}: "
                    f"skewness = "
                    f"{X[column].skew():.3f}"
                )

        else:

            print("\nNo highly skewed features found.")

        return skewed_features

    # -----------------------------------------
    # 2. Apply log transformation
    # -----------------------------------------

    def log_transform(self, X_train, X_test):

        skewed_features = self.detect_skewed_features(
            X_train
        )

        for column in skewed_features:

            # Only apply log transformation
            # when values are non-negative

            if (X_train[column] >= 0).all():

                X_train[column] = np.log1p(
                    X_train[column]
                )

                X_test[column] = np.log1p(
                    X_test[column]
                )

                print(
                    f"Log transformation applied: "
                    f"{column}"
                )

        return X_train, X_test

    # -----------------------------------------
    # 3. Remove constant features
    # -----------------------------------------

    def remove_constant_features(
        self,
        X_train,
        X_test
    ):

        constant_columns = []

        for column in X_train.columns:

            if X_train[column].nunique() <= 1:

                constant_columns.append(column)

        print("\n" + "=" * 50)
        print("CONSTANT FEATURE REMOVAL")
        print("=" * 50)

        if constant_columns:

            print(
                f"\nRemoving: "
                f"{constant_columns}"
            )

            X_train = X_train.drop(
                columns=constant_columns
            )

            X_test = X_test.drop(
                columns=constant_columns
            )

        else:

            print("\nNo constant features found.")

        return X_train, X_test

    # -----------------------------------------
    # 4. Run feature engineering
    # -----------------------------------------

    def run(self, X_train, X_test):

        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)

        # Remove constant features
        X_train, X_test = (
            self.remove_constant_features(
                X_train,
                X_test
            )
        )

        # Detect and transform skewed features
        X_train, X_test = (
            self.log_transform(
                X_train,
                X_test
            )
        )

        print("\nFeature engineering completed.")

        print(
            f"Training features: "
            f"{X_train.shape[1]}"
        )

        print(
            f"Testing features: "
            f"{X_test.shape[1]}"
        )

        return X_train, X_test