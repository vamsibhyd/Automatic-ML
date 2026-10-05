import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

        self.scaler = None
        self.encoder = None

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
    # 2. Train / Test split
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

    # -----------------------------------------
    # 3. Handle missing values
    # -----------------------------------------

    def handle_missing_values(self, X_train, X_test):

        print("\n" + "=" * 50)
        print("MISSING VALUE HANDLING")
        print("=" * 50)

        for column in X_train.columns:

            if X_train[column].isnull().sum() > 0:

                if pd.api.types.is_numeric_dtype(
                    X_train[column]
                ):

                    value = X_train[column].median()

                else:

                    value = X_train[column].mode()[0]

                X_train[column] = X_train[column].fillna(value)

                X_test[column] = X_test[column].fillna(value)

                print(
                    f"{column}: missing values filled"
                )

        print("Missing value handling completed.")

        return X_train, X_test

    # -----------------------------------------
    # 4. Encode categorical features
    # -----------------------------------------

    def encode_categorical_features(
        self,
        X_train,
        X_test
    ):

        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        print("\n" + "=" * 50)
        print("CATEGORICAL ENCODING")
        print("=" * 50)

        print(
            f"\nCategorical columns: "
            f"{categorical_columns}"
        )

        if len(categorical_columns) == 0:

            print(
                "No categorical features require encoding."
            )

            return X_train, X_test

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        train_encoded = self.encoder.fit_transform(
            X_train[categorical_columns]
        )

        test_encoded = self.encoder.transform(
            X_test[categorical_columns]
        )

        encoded_columns = (
            self.encoder
            .get_feature_names_out(
                categorical_columns
            )
        )

        train_encoded = pd.DataFrame(
            train_encoded,
            columns=encoded_columns,
            index=X_train.index
        )

        test_encoded = pd.DataFrame(
            test_encoded,
            columns=encoded_columns,
            index=X_test.index
        )

        X_train = X_train.drop(
            columns=categorical_columns
        )

        X_test = X_test.drop(
            columns=categorical_columns
        )

        X_train = pd.concat(
            [X_train, train_encoded],
            axis=1
        )

        X_test = pd.concat(
            [X_test, test_encoded],
            axis=1
        )

        return X_train, X_test

    # -----------------------------------------
    # 5. Feature scaling
    # -----------------------------------------

    def scale_features(self, X_train, X_test):

        numerical_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        print("\n" + "=" * 50)
        print("FEATURE SCALING")
        print("=" * 50)

        print(
            f"\nNumerical features: "
            f"{len(numerical_columns)}"
        )

        if len(numerical_columns) == 0:

            print("No numerical features to scale.")

            return X_train, X_test

        self.scaler = StandardScaler()

        # Learn scaling parameters ONLY from training data
        X_train[numerical_columns] = (
            self.scaler.fit_transform(
                X_train[numerical_columns]
            )
        )

        # Apply the same transformation to test data
        X_test[numerical_columns] = (
            self.scaler.transform(
                X_test[numerical_columns]
            )
        )

        print("Features standardized.")

        return X_train, X_test