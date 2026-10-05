import pandas as pd


class DatasetProfiler:

    def __init__(self, df):

        self.df = df

    def profile(self):

        print("\n" + "=" * 50)
        print("DATASET PROFILE")
        print("=" * 50)

        # ==========================================
        # DATASET SIZE
        # ==========================================

        rows, columns = self.df.shape

        print(f"\nRows: {rows}")
        print(f"Columns: {columns}")

        # ==========================================
        # COLUMN TYPES
        # ==========================================

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        print("\nNumerical columns:")
        print(numerical_columns)

        print(
            f"Number of numerical columns: "
            f"{len(numerical_columns)}"
        )

        print("\nCategorical columns:")
        print(categorical_columns)

        print(
            f"Number of categorical columns: "
            f"{len(categorical_columns)}"
        )

        # ==========================================
        # MISSING VALUES
        # ==========================================

        missing_values = self.df.isnull().sum()

        total_missing = missing_values.sum()

        print("\nMissing values:")

        if total_missing == 0:

            print("None")

        else:

            print(
                missing_values[
                    missing_values > 0
                ]
            )

        print(
            f"Total missing values: "
            f"{total_missing}"
        )

        # ==========================================
        # DUPLICATES
        # ==========================================

        duplicates = self.df.duplicated().sum()

        print(
            f"\nDuplicate rows: {duplicates}"
        )

        # ==========================================
        # UNIQUE VALUES
        # ==========================================

        unique_values = self.df.nunique()

        print("\nUnique values per column:")

        print(unique_values)

        # ==========================================
        # CONSTANT COLUMNS
        # ==========================================

        constant_columns = (
            unique_values[
                unique_values <= 1
            ]
            .index
            .tolist()
        )

        print("\nConstant columns:")

        if constant_columns:

            print(constant_columns)

        else:

            print("None")

        # ==========================================
        # DATA TYPES
        # ==========================================

        print("\nData types:")

        print(self.df.dtypes)