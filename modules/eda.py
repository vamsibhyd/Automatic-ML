import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAEngine:

    def __init__(self, df):
        self.df = df

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
                self.df[column].value_counts()
            )

    # ==========================================
    # DISTRIBUTION ANALYSIS
    # ==========================================

    def plot_distributions(self, max_plots=10):

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        columns_to_plot = numerical_columns[:max_plots]

        print("\n" + "=" * 50)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 50)

        print(
            f"\nGenerating "
            f"{len(columns_to_plot)} "
            f"distribution plots..."
        )

        for column in columns_to_plot:

            plt.figure(figsize=(6, 4))

            sns.histplot(
                self.df[column],
                kde=True
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.xlabel(column)
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()

    # ==========================================
    # CORRELATION ANALYSIS
    # ==========================================

    def correlation_analysis(self):

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        if len(numerical_columns) < 2:

            print(
                "Not enough numerical columns "
                "for correlation analysis."
            )

            return

        correlation_matrix = (
            self.df[numerical_columns].corr()
        )

        print("\nCorrelation matrix:")

        print(correlation_matrix)

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

    # ==========================================
    # OUTLIER ANALYSIS
    # ==========================================

    def outlier_analysis(self):

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        print("\n" + "=" * 50)
        print("OUTLIER ANALYSIS")
        print("=" * 50)

        if len(numerical_columns) == 0:

            print("No numerical columns found.")

            return

        for column in numerical_columns:

            Q1 = self.df[column].quantile(0.25)

            Q3 = self.df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR

            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[column] < lower_bound)
                |
                (self.df[column] > upper_bound)
            ]

            print(
                f"{column}: "
                f"{len(outliers)} outliers"
            )

    # ==========================================
    # RUN COMPLETE EDA
    # ==========================================

    def run(self):

        self.numerical_summary()

        self.categorical_summary()

        self.outlier_analysis()

        self.correlation_analysis()

        self.plot_distributions(
            max_plots=10
        )