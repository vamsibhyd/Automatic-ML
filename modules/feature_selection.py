import pandas as pd

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)


class FeatureSelector:

    def __init__(self):
        self.scores = None
        self.selected_features = None

    # ==========================================
    # DETECT DISCRETE FEATURES
    # ==========================================

    def detect_discrete_features(self, X_train):
        """
        Detect discrete features automatically.

        Binary and categorical columns are treated as discrete.
        Continuous numerical columns are treated as continuous.
        """

        discrete_features = []

        for column in X_train.columns:

            values = X_train[column].dropna()
            unique_values = values.nunique()

            # Boolean and categorical columns are discrete
            if (
                pd.api.types.is_bool_dtype(X_train[column])
                or isinstance(
                    X_train[column].dtype,
                    pd.CategoricalDtype
                )
                or pd.api.types.is_object_dtype(X_train[column])
                or pd.api.types.is_string_dtype(X_train[column])
            ):
                discrete_features.append(True)

            # Binary columns containing only 0 and 1
            elif set(values.unique()).issubset({0, 1}):
                discrete_features.append(True)

            # Low-cardinality integer columns are discrete
            elif (
                pd.api.types.is_integer_dtype(X_train[column])
                and unique_values <= 10
            ):
                discrete_features.append(True)

            # Other numerical columns are continuous
            else:
                discrete_features.append(False)

        return discrete_features

    # ==========================================
    # CALCULATE FEATURE IMPORTANCE
    # ==========================================

    def calculate_scores(
        self,
        X_train,
        y_train,
        problem_type="classification",
        discrete_features=None
    ):

        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        # Validate input
        if X_train.shape[1] == 0:
            raise ValueError(
                "No features available for selection."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train have different lengths."
            )

        if y_train.isna().any():
            raise ValueError(
                "Target contains missing values."
            )

        if X_train.isna().any().any():
            raise ValueError(
                "Features contain missing values. "
                "Handle missing values before feature selection."
            )

        # Automatically detect feature types
        if discrete_features is None:
            discrete_features = self.detect_discrete_features(
                X_train
            )

        # Select the appropriate scoring method
        if problem_type in (
            "classification",
            "Binary Classification",
            "Multiclass Classification"
        ):

            score_function = mutual_info_classif

        elif problem_type in (
            "regression",
            "Regression"
        ):

            score_function = mutual_info_regression

        else:
            raise ValueError(
                f"Unsupported problem type: {problem_type}"
            )

        # Calculate mutual information scores
        scores = score_function(
            X_train,
            y_train,
            discrete_features=discrete_features,
            random_state=42
        )

        self.scores = pd.Series(
            scores,
            index=X_train.columns,
            name="mutual_information"
        ).sort_values(ascending=False)

        print("\nFeature scores:")
        print(self.scores)

        return self.scores

    # ==========================================
    # SELECT TOP FEATURES
    # ==========================================

    def select_features(
        self,
        X_train,
        X_test,
        y_train,
        number_of_features=20,
        problem_type="classification",
        discrete_features=None
    ):

        if number_of_features < 1:
            raise ValueError(
                "number_of_features must be at least 1."
            )

        if list(X_train.columns) != list(X_test.columns):
            raise ValueError(
                "Training and testing features do not match."
            )

        if X_train.shape[1] == 0:
            raise ValueError(
                "No features available for selection."
            )

        # Calculate feature scores
        self.calculate_scores(
            X_train,
            y_train,
            problem_type=problem_type,
            discrete_features=discrete_features
        )

        # Select the requested number of features
        number_of_features = min(
            number_of_features,
            X_train.shape[1]
        )

        self.selected_features = (
            self.scores
            .head(number_of_features)
            .index
            .tolist()
        )

        print("\n" + "=" * 50)
        print("FEATURE SELECTION")
        print("=" * 50)

        print(
            f"\nOriginal number of features: "
            f"{X_train.shape[1]}"
        )

        print(
            f"Selected number of features: "
            f"{len(self.selected_features)}"
        )

        print("\nSelected features:")

        for feature in self.selected_features:
            print(
                f"{feature}: "
                f"{self.scores[feature]:.4f}"
            )

        # Apply the same selected columns to both datasets
        X_train_selected = X_train[
            self.selected_features
        ].copy()

        X_test_selected = X_test[
            self.selected_features
        ].copy()

        return (
            X_train_selected,
            X_test_selected
        )