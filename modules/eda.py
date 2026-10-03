import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAEngine:

    def __init__(self, df):
        self.df = df

    def numerical_summary(self):

        numerical_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        print("\n" + "=" * 50)
        print("NUMERICAL SUMMARY")
        print("=" * 50)

        print(self.df[numerical_columns].describe())

    def categorical_summary(self):

        categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        print("\n" + "=" * 50)
        print("CATEGORICAL SUMMARY")
        print("=" * 50)

        for column in categorical_columns:

            print(f"\nColumn: {column}")

            print(
                self.df[column].value_counts()
            )

    def plot_distributions(self):

        numerical_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        print("\nGenerating distribution plots...")

        for column in numerical_columns:

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

    def correlation_analysis(self):

        numerical_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        correlation_matrix = self.df[
            numerical_columns
        ].corr()

        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        print(correlation_matrix)

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0
        )

        plt.title("Feature Correlation Matrix")

        plt.tight_layout()
        plt.show()

    def outlier_analysis(self):

        numerical_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        print("\n" + "=" * 50)
        print("OUTLIER ANALYSIS")
        print("=" * 50)

        for column in numerical_columns:

            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)

            IQR = Q3 - Q1                #Interquartile Range(outlier is nothing but the far away values from the feature)

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[column] < lower_bound)
                |
                (self.df[column] > upper_bound)
            ]

            print(
                f"{column}: {len(outliers)} outliers"
            )

    def run(self):

        self.numerical_summary()

        self.categorical_summary()

        self.outlier_analysis()

        self.correlation_analysis()

        self.plot_distributions()