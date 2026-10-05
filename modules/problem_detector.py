import pandas as pd


class ProblemDetector:

    def __init__(self, df):

        self.df = df

    # ==========================================
    # FIND POSSIBLE TARGET COLUMNS
    # ==========================================

    def find_target_candidates(self):

        candidates = []

        for column in self.df.columns:

            unique_values = self.df[column].nunique()
            data_type = self.df[column].dtype

            # Categorical columns
            if (
                data_type == "object"
                or str(data_type) == "category"
            ):

                candidates.append({
                    "column": column,
                    "type": "categorical",
                    "unique_values": unique_values
                })

            # Numerical columns with few unique values
            elif unique_values <= 10:

                candidates.append({
                    "column": column,
                    "type": "numerical",
                    "unique_values": unique_values
                })

        return candidates

    # ==========================================
    # DETECT MACHINE LEARNING PROBLEM
    # ==========================================

    def detect_problem(self, target_column):

        target = self.df[target_column]

        unique_values = target.nunique()

        # Numerical target
        if pd.api.types.is_numeric_dtype(target):

            if unique_values <= 10:

                return "Classification"

            return "Regression"

        # Categorical target
        if unique_values == 2:

            return "Binary Classification"

        return "Multiclass Classification"

    # ==========================================
    # RUN PROBLEM DETECTION
    # ==========================================

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
                f"Unique values: "
                f"{candidate['unique_values']}"
            )

        # Check whether a candidate was found
        if not candidates:

            raise ValueError(
                "No possible target column found."
            )

        # Select the first candidate
        target_column = candidates[0]["column"]

        problem_type = self.detect_problem(
            target_column
        )

        print(
            f"\nDetected target: "
            f"{target_column}"
        )

        print(
            f"Detected problem type: "
            f"{problem_type}"
        )

        return target_column, problem_type