import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:

    def __init__(self, df):

        self.df = df.copy()

    # -----------------------------------------
    # 1. Separate features and target
    # -----------------------------------------

    def separate_features_target(self, target_column):

        X = self.df.drop(columns=[target_column])

        y = self.df[target_column]

        print("\n" + "=" * 50)
        print("FEATURE / TARGET SEPARATION")
        print("=" * 50)

        print(f"\nTarget column: {target_column}")

        print(f"Number of features: {X.shape[1]}")

        print(f"Number of samples: {X.shape[0]}")

        return X, y


    # -----------------------------------------
    # 2. Handle missing values
    # -----------------------------------------

    def handle_missing_values(self, X):

        print("\n" + "=" * 50)
        print("MISSING VALUE HANDLING")
        print("=" * 50)

        missing_values = X.isnull().sum()

        total_missing = missing_values.sum()

        if total_missing == 0:

            print("No missing values found.")

        else:

            print(f"Total missing values: {total_missing}")

            for column in X.columns:

                missing = X[column].isnull().sum()

                if missing > 0:

                    if pd.api.types.is_numeric_dtype(X[column]):

                        median_value = X[column].median()

                        X[column] = X[column].fillna(
                            median_value
                        )

                        print(
                            f"{column}: "
                            f"filled using median"
                        )

                    else:

                        mode_value = X[column].mode()[0]

                        X[column] = X[column].fillna(
                            mode_value
                        )

                        print(
                            f"{column}: "
                            f"filled using mode"
                        )

        return X


    # -----------------------------------------
    # 3. Encode categorical features
    # -----------------------------------------

    def encode_categorical_features(self, X):

        categorical_columns = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numerical_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        print("\n" + "=" * 50)
        print("CATEGORICAL ENCODING")
        print("=" * 50)

        print(f"\nCategorical columns: {categorical_columns}")

        if len(categorical_columns) == 0:

            print("No categorical features require encoding.")

            return X

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        encoded_data = encoder.fit_transform(
            X[categorical_columns]
        )

        encoded_columns = encoder.get_feature_names_out(
            categorical_columns
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_columns,
            index=X.index
        )

        X = X.drop(columns=categorical_columns)

        X = pd.concat(
            [X, encoded_df],
            axis=1
        )

        print("Categorical features encoded.")

        return X


    # -----------------------------------------
    # 4. Scale numerical features
    # -----------------------------------------

    def scale_features(self, X):

        numerical_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        print("\n" + "=" * 50)
        print("FEATURE SCALING")
        print("=" * 50)

        print(
            f"\nNumber of numerical features: "
            f"{len(numerical_columns)}"
        )

        if len(numerical_columns) == 0:

            print("No numerical features to scale.")

            return X

        scaler = StandardScaler()

        X[numerical_columns] = scaler.fit_transform(
            X[numerical_columns]
        )

        print("Numerical features standardized.")

        return X


    # -----------------------------------------
    # 5. Train / Test split
    # -----------------------------------------

    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("\n" + "=" * 50)
        print("TRAIN / TEST SPLIT")
        print("=" * 50)

        print(f"\nTraining samples: {len(X_train)}")

        print(f"Testing samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test