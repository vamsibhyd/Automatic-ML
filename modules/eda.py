import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAEngine:

    def __init__(self, df):
        self.df = df.copy()

    # ==========================================
    # DATA QUALITY ANALYSIS
    # ==========================================

    def data_quality_analysis(self):

        print("\n" + "=" * 50)
        print("DATA QUALITY ANALYSIS")
        print("=" * 50)

        rows, columns = self.df.shape

        print(f"\nRows: {rows}")
        print(f"Columns: {columns}")

        # Missing values
        missing_counts = self.df.isnull().sum()
        missing_counts = missing_counts[
            missing_counts > 0
        ].sort_values(ascending=False)

        print("\nMissing values:")

        if missing_counts.empty:
            print("No missing values found.")
        else:
            print(missing_counts)

        total_missing = int(
            self.df.isnull().sum().sum()
        )

        print(f"Total missing values: {total_missing}")

        # Duplicate rows
        duplicate_count = int(
            self.df.duplicated().sum()
        )

        duplicate_percentage = (
            duplicate_count / rows * 100
            if rows > 0
            else 0
        )

        print(f"\nDuplicate rows: {duplicate_count}")
        print(
            f"Duplicate percentage: "
            f"{duplicate_percentage:.2f}%"
        )

        # Data types
        print("\nData types:")
        print(self.df.dtypes.value_counts())

        # Constant columns
        constant_columns = [
            column
            for column in self.df.columns
            if self.df[column].nunique(dropna=False) <= 1
        ]

        print("\nConstant columns:")

        if constant_columns:
            print(constant_columns)
        else:
            print("No constant columns found.")

    # ==========================================
    # NUMERICAL SUMMARY
    # ==========================================

    def numerical_summary(self):

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        print("\n" + "=" * 50)
        print("NUMERICAL SUMMARY")
        print("=" * 50)

        if len(numerical_columns) == 0:
            print("No numerical columns found.")
            return

        print(
            self.df[numerical_columns].describe()
        )

    # ==========================================
    # CATEGORICAL SUMMARY
    # ==========================================

    def categorical_summary(self):

        categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        print("\n" + "=" * 50)
        print("CATEGORICAL SUMMARY")
        print("=" * 50)

        if len(categorical_columns) == 0:
            print("No categorical columns found.")
            return

        for column in categorical_columns:

            print(f"\n{column}:")

            print(
                self.df[column].value_counts(
                    dropna=False
                ).head(20)
            )

    # ==========================================
    # BINARY FEATURE SUMMARY
    # ==========================================

    def binary_summary(self):

        binary_columns = []

        for column in self.df.columns:

            series = self.df[column].dropna()

            if series.empty:
                continue

            unique_values = set(series.unique())

            if (
                len(unique_values) == 2
                and unique_values.issubset({0, 1})
            ):
                binary_columns.append(column)

        print("\n" + "=" * 50)
        print("BINARY FEATURE SUMMARY")
        print("=" * 50)

        if not binary_columns:
            print("No binary 0/1 features found.")
            return

        summary = pd.DataFrame({
            "Zero_Count": (
                self.df[binary_columns] == 0
            ).sum(),

            "One_Count": (
                self.df[binary_columns] == 1
            ).sum(),

            "One_Percentage": (
                self.df[binary_columns].mean() * 100
            ).round(2)
        })

        print(summary.to_string())

    # ==========================================
    # DISTRIBUTION ANALYSIS
    # ==========================================

    def plot_distributions(self, max_plots=10):

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        columns_to_plot = list(
            numerical_columns[:max_plots]
        )

        print("\n" + "=" * 50)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 50)

        print(
            f"\nGenerating {len(columns_to_plot)} "
            "distribution plots..."
        )

        for column in columns_to_plot:

            series = self.df[column].dropna()

            if series.empty:
                continue

            plt.figure(figsize=(6, 4))

            unique_values = set(series.unique())

            # Binary 0/1 features
            if (
                len(unique_values) == 2
                and unique_values.issubset({0, 1})
            ):

                sns.countplot(
                    data=self.df,
                    x=column
                )

                plt.ylabel("Count")

            # Other numerical features
            else:

                sns.histplot(
                    data=self.df,
                    x=column,
                    kde=(
                        series.nunique() > 10
                        and len(series) > 1
                    )
                )

                plt.ylabel("Frequency")

            plt.title(
                f"Distribution of {column}"
            )

            plt.xlabel(column)

            plt.tight_layout()
            plt.show()

    # ==========================================
    # CORRELATION ANALYSIS
    # ==========================================

    def correlation_analysis(
        self,
        target_column=None,
        top_n=20
    ):

        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        # Optionally exclude the target
        if target_column is not None:

            if target_column not in self.df.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    "not found."
                )

            numerical_columns = [
                column
                for column in numerical_columns
                if column != target_column
            ]

        if len(numerical_columns) < 2:

            print(
                "Not enough numerical features "
                "for correlation analysis."
            )

            return None, None

        correlation_matrix = (
            self.df[numerical_columns].corr()
        )

        # Extract unique feature pairs
        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool)
        )

        pairs = (
            upper_triangle.stack()
            .rename("correlation")
            .reset_index()
        )

        pairs.columns = [
            "Feature_1",
            "Feature_2",
            "Correlation"
        ]

        pairs["Absolute_Correlation"] = (
            pairs["Correlation"].abs()
        )

        strongest_pairs = pairs.sort_values(
            "Absolute_Correlation",
            ascending=False
        ).head(top_n)

        print(
            f"\nTop {len(strongest_pairs)} "
            "feature correlations:"
        )

        if strongest_pairs.empty:
            print("No valid feature pairs found.")
        else:
            print(
                strongest_pairs.to_string(
                    index=False
                )
            )

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Feature Correlation Matrix"
        )

        plt.tight_layout()
        plt.show()

        return correlation_matrix, strongest_pairs

    # ==========================================
    # OUTLIER ANALYSIS
    # ==========================================

    def outlier_analysis(self, target_column=None):

        print("\n" + "=" * 50)
        print("OUTLIER ANALYSIS")
        print("=" * 50)

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        # Do not treat the target as a predictor feature
        if target_column is not None:

            if target_column not in self.df.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    "not found."
                )

            numerical_columns = [
                column
                for column in numerical_columns
                if column != target_column
            ]

        continuous_columns = []

        for column in numerical_columns:

            series = self.df[column].dropna()

            if series.empty:
                continue

            unique_values = set(series.unique())

            # Exclude binary 0/1 features
            if (
                len(unique_values) == 2
                and unique_values.issubset({0, 1})
            ):
                continue

            # Exclude constant columns
            if series.nunique() <= 1:
                continue

            continuous_columns.append(column)

        if not continuous_columns:

            print(
                "No suitable continuous numerical "
                "features found."
            )

            print(
                "IQR-based outlier analysis skipped."
            )

            return {}

        results = {}

        for column in continuous_columns:

            series = self.df[column].dropna()

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)

            IQR = Q3 - Q1

            if IQR == 0:

                print(
                    f"{column}: skipped because IQR = 0"
                )

                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (
                (series < lower_bound)
                |
                (series > upper_bound)
            )

            outlier_count = int(
                outlier_mask.sum()
            )

            percentage = (
                outlier_count / len(series)
            ) * 100

            results[column] = {
                "count": outlier_count,
                "percentage": round(
                    percentage, 2
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

            print(
                f"{column}: {outlier_count} potential "
                f"outliers ({percentage:.2f}%)"
            )

        return results

    # ==========================================
    # RUN COMPLETE EDA
    # ==========================================

    def run(
        self,
        target_column=None,
        max_plots=10,
        top_n=20
    ):

        self.data_quality_analysis()

        self.numerical_summary()

        self.categorical_summary()

        self.binary_summary()

        self.outlier_analysis(
            target_column=target_column
        )

        self.correlation_analysis(
            target_column=target_column,
            top_n=top_n
        )

        self.plot_distributions(
            max_plots=max_plots
        )