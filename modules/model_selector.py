from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class ModelSelector:

    def __init__(self, problem_type):

        self.problem_type = problem_type

    def get_models(self):

        print("\n" + "=" * 50)
        print("MODEL SELECTION")
        print("=" * 50)

        if self.problem_type == "Binary Classification":

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        random_state=42
                    ),

                "K-Nearest Neighbors":
                    KNeighborsClassifier(),

                "Support Vector Machine":
                    SVC(),

                "Naive Bayes":
                    GaussianNB()
            }

        elif self.problem_type == "Multiclass Classification":

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        random_state=42
                    ),

                "K-Nearest Neighbors":
                    KNeighborsClassifier(),

                "Support Vector Machine":
                    SVC(),

                "Naive Bayes":
                    GaussianNB()
            }

        else:

            raise ValueError(
                "Unsupported problem type"
            )

        print("\nSuitable algorithms:")

        for model_name in models:

            print(f"- {model_name}")

        return models