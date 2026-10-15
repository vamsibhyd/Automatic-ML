from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelTrainer:

    def __init__(self, models):

        self.models = models
        self.results = {}

    # ==========================================
    # TRAIN AND EVALUATE MODELS
    # ==========================================

    def train_and_evaluate(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        print("\n" + "=" * 50)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 50)

        for name, model in self.models.items():

            print(f"\nTraining: {name}")

            # ==========================================
            # TRAIN MODEL
            # ==========================================

            model.fit(
                X_train,
                y_train
            )

            # ==========================================
            # MAKE PREDICTIONS
            # ==========================================

            y_pred = model.predict(
                X_test
            )

            # ==========================================
            # CALCULATE METRICS
            # ==========================================

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            # ==========================================
            # STORE RESULTS
            # ==========================================

            self.results[name] = {

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1 Score": f1
            }

            # ==========================================
            # DISPLAY RESULTS
            # ==========================================

            print(
                f"Accuracy: {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall: {recall:.4f}"
            )

            print(
                f"F1 Score: {f1:.4f}"
            )

        return self.results

    # ==========================================
    # SELECT BEST MODEL
    # ==========================================

    def select_best_model(self):

        print("\n" + "=" * 50)
        print("BEST MODEL SELECTION")
        print("=" * 50)

        best_model_name = max(
            self.results,
            key=lambda model:
            self.results[model]["F1 Score"]
        )

        best_score = self.results[
            best_model_name
        ]["F1 Score"]

        print(
            f"\nBest model: {best_model_name}"
        )

        print(
            f"Best F1 Score: {best_score:.4f}"
        )

        return best_model_name