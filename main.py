import pandas as pd

from modules.profiler import DatasetProfiler
from modules.eda import EDAEngine
from modules.problem_detector import ProblemDetector
from modules.preprocessing import DataPreprocessor
from modules.feature_engineering import FeatureEngineer
from modules.feature_selection import FeatureSelector
from modules.model_selector import ModelSelector
from modules.model_trainer import ModelTrainer


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Sonar.csv")

print("\nDataset loaded successfully!")


# ==========================================
# 2. DATASET PROFILING
# ==========================================

profiler = DatasetProfiler(df)

profiler.profile()


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

eda = EDAEngine(df)

eda.run()


# ==========================================
# 4. PROBLEM DETECTION
# ==========================================

detector = ProblemDetector(df)

target_column, problem_type = detector.analyze()


# ==========================================
# 5. DATA PREPARATION
# ==========================================

preprocessor = DataPreprocessor(df)


# Separate features and target
X, y = preprocessor.separate_features_target(
    target_column
)


# Split dataset
X_train, X_test, y_train, y_test = (
    preprocessor.split_data(
        X,
        y
    )
)


# Handle missing values
X_train, X_test = (
    preprocessor.handle_missing_values(
        X_train,
        X_test
    )
)


# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

feature_engineer = FeatureEngineer()

X_train, X_test = feature_engineer.run(
    X_train,
    X_test
)


# ==========================================
# 7. CATEGORICAL ENCODING
# ==========================================

X_train, X_test = (
    preprocessor.encode_categorical_features(
        X_train,
        X_test
    )
)


# ==========================================
# 8. FEATURE SCALING
# ==========================================

X_train, X_test = (
    preprocessor.scale_features(
        X_train,
        X_test
    )
)


# ==========================================
# 9. FEATURE SELECTION
# ==========================================

selector = FeatureSelector()

X_train, X_test = selector.select_features(
    X_train,
    X_test,
    y_train,
    number_of_features=20
)


# ==========================================
# 10. MODEL SELECTION
# ==========================================

model_selector = ModelSelector(
    problem_type
)

models = model_selector.get_models()


# ==========================================
# 11. MODEL TRAINING & EVALUATION
# ==========================================

trainer = ModelTrainer(models)

results = trainer.train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test
)


# ==========================================
# 12. BEST MODEL
# ==========================================

best_model = trainer.select_best_model()