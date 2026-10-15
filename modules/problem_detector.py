import pandas as pd


class ProblemDetector:

    def __init__(self, df):
        self.df = df.copy()

    # ==========================================
    # FIND POSSIBLE TARGET COLUMNS
    # ==========================================

    def find_target_candidates(self):

        candidates = []

        for column in self.df.columns:

            series = self.df[column]
            unique_values = series.nunique()
            data_type = series.dtype

            # Skip columns containing only one unique value
            if unique_values <= 1:
                continue

            # Categorical columns
            if (
                pd.api.types.is_object_dtype(series)
                or isinstance(
                    data_type,
                    pd.CategoricalDtype
                )
                or pd.api.types.is_bool_dtype(series)
            ):

                candidates.append({
                    "column": column,
                    "type": "categorical",
                    "unique_values": unique_values
                })

            # Numerical columns with relatively few unique values
            elif unique_values <= 10:

                candidates.append({
                    "column": column,
                    "type": "numerical",
                    "unique_values": unique_values
                })

            # Numerical columns with many unique values
            elif pd.api.types.is_numeric_dtype(series):

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

        if target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )

        target = self.df[target_column].dropna()

        if target.empty:
            raise ValueError(
                "The target column contains no valid values."
            )

        unique_values = target.nunique()

        # A constant target cannot support meaningful prediction
        if unique_values < 2:
            raise ValueError(
                "The target must contain at least "
                "two distinct values."
            )

        # Categorical targets indicate classification
        if (
            pd.api.types.is_object_dtype(target)
            or isinstance(
                target.dtype,
                pd.CategoricalDtype
            )
            or pd.api.types.is_bool_dtype(target)
        ):

            if unique_values == 2:
                return "Binary Classification"

            return "Multiclass Classification"

        # Numeric targets with two distinct values
        # are treated as binary classification by default.
        if unique_values == 2:
            return "Binary Classification"

        # Numeric targets with a small number of distinct values
        # are treated as classification candidates.
        if unique_values <= 10:
            return "Multiclass Classification"

        # Numeric targets with many distinct values
        # are treated as regression candidates.
        if pd.api.types.is_numeric_dtype(target):
            return "Regression"

        raise ValueError(
            "Unable to determine the problem type. "
            "Please specify it manually."
        )

    # ==========================================
    # RUN PROBLEM DETECTION
    # ==========================================

    def analyze(self, target_column=None):

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

        if not candidates:
            raise ValueError(
                "No possible target column found."
            )

        # Use the explicitly selected target when provided.
        if target_column is not None:

            if target_column not in self.df.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    "not found."
                )

        else:

            # Prefer a conventional target name as a suggestion.
            # This is a heuristic, not proof that the column
            # is the correct target.
            common_target_names = [
                "target",
                "label",
                "class",
                "outcome",
                "y"
            ]

            column_lookup = {
                str(column).strip().lower(): column
                for column in self.df.columns
            }

            target_column = None

            for name in common_target_names:

                if name in column_lookup:

                    candidate_column = column_lookup[name]

                    if (
                        self.df[candidate_column].nunique() > 1
                    ):
                        target_column = candidate_column
                        break

            if target_column is None:

                raise ValueError(
                    "Target column could not be determined "
                    "reliably. Please select the target "
                    "column explicitly."
                )

            print(
                "\nSuggested target based on its name: "
                f"{target_column}"
            )

            print(
                "Please confirm that this is the intended "
                "prediction target."
            )

        problem_type = self.detect_problem(target_column)

        print(f"\nDetected target: {target_column}")
        print(f"Detected problem type: {problem_type}")

        return target_column, problem_type