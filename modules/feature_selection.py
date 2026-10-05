import pandas as pd

from sklearn.feature_selection import mutual_info_classif


class FeatureSelector:

    def __init__(self):
        self.scores = None
        self.selected_features = None

    def calculate_scores(self, X_train, y_train):

        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        scores = mutual_info_classif(
            X_train,
            y_train,
            random_state=42
        )

        self.scores = pd.Series(
            scores,
            index=X_train.columns
        )

        self.scores = self.scores.sort_values(
            ascending=False
        )

        print("\nFeature scores:")

        print(self.scores)

        return self.scores

    def select_features(
        self,
        X_train,
        X_test,
        y_train,
        number_of_features=20
    ):

        self.calculate_scores(
            X_train,
            y_train
        )

        # Select top features
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

        X_train_selected = X_train[
            self.selected_features
        ]

        X_test_selected = X_test[
            self.selected_features
        ]

        return (
            X_train_selected,
            X_test_selected
        )