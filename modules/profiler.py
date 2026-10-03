import pandas as pd


class DatasetProfiler:

    def __init__(self, df):
        self.df = df

    def profile(self):

        print("\n" + "=" * 50)
        print("DATASET PROFILE")
        print("=" * 50)

        # 1. Dataset size
        rows, columns = self.df.shape

        print(f"\nNumber of rows: {rows}")
        print(f"Number of columns: {columns}")

        # 2. Numerical columns
        numerical_columns = self.df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        # 3. Categorical columns
        categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        print("\nNumerical columns:")
        print(numerical_columns)

        print(f"\nNumber of numerical columns: {len(numerical_columns)}")

        print("\nCategorical columns:")
        print(categorical_columns)

        print(f"\nNumber of categorical columns: {len(categorical_columns)}")

        # 4. Missing values
        missing_values = self.df.isnull().sum()

        print("\nMissing values:")
        print(missing_values[missing_values > 0])

        total_missing = missing_values.sum()

        print(f"\nTotal missing values: {total_missing}")

        # 5. Duplicate rows
        duplicates = self.df.duplicated().sum()

        print(f"\nDuplicate rows: {duplicates}")

        # 6. Unique values
        print("\nUnique values per column:")

        unique_values = self.df.nunique()

        print(unique_values)

        # 7. Constant columns
        constant_columns = unique_values[
            unique_values <= 1
        ].index.tolist()

        print("\nConstant columns:")

        if constant_columns:
            print(constant_columns)
        else:
            print("None")

        # 8. Data types
        print("\nData types:")

        print(self.df.dtypes)