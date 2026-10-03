import pandas as pd


class ProblemDetector:

    def __init__(self, df):
        self.df = df

    def find_target_candidates(self):

        candidates = []

        for column in self.df.columns:

            unique_values = self.df[column].nunique()
            data_type = self.df[column].dtype

            # Categorical column
            if data_type == "object" or str(data_type) == "category":

                candidates.append({
                    "column": column,
                    "type": "categorical",
                    "unique_values": unique_values
                })

            # Numerical column with relatively few unique values
            elif unique_values <= 10:

                candidates.append({
                    "column": column,
                    "type": "numerical",
                    "unique_values": unique_values
                })

        return candidates

    def detect_problem(self, target_column):

        target = self.df[target_column]

        unique_values = target.nunique()

        if pd.api.types.is_numeric_dtype(target):

            if unique_values <= 10:

                problem_type = "Classification"

            else:

                problem_type = "Regression"

        else:

            if unique_values == 2:

                problem_type = "Binary Classification"

            else:

                problem_type = "Multiclass Classification"

        return problem_type

    def analyze(self):

        print("\n" + "=" * 50)
        print("PROBLEM DETECTION")
        print("=" * 50)

        candidates = self.find_target_candidates()

        print("\nPossible target columns:")

        for candidate in candidates:

            print(
                f"{candidate['column']} | "
                f"Type: {candidate['type']} | "
                f"Unique values: {candidate['unique_values']}"
            )

        return candidates