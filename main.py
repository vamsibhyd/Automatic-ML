import pandas as pd

from modules.profiler import DatasetProfiler
from modules.eda import EDAEngine
from modules.problem_detector import ProblemDetector
from modules.preprocessing import DataPreprocessor


# Load dataset
df = pd.read_csv("Sonar.csv")

print("Dataset loaded successfully!")


# Dataset profiling
profiler = DatasetProfiler(df)
profiler.profile()


# EDA
eda = EDAEngine(df)
eda.run()



# Problem detection
detector = ProblemDetector(df)

candidates = detector.analyze()


# Data processing

preprocessor = DataPreprocessor(df)

# Step 1: Separate X and y
X, y = preprocessor.separate_features_target("Label")

# Step 2: Handle missing values
X = preprocessor.handle_missing_values(X)

# Step 3: Encode categorical features
X = preprocessor.encode_categorical_features(X)

# Step 4: Scale numerical features
X = preprocessor.scale_features(X)

# Step 5: Split into training and testing data
X_train, X_test, y_train, y_test = preprocessor.split_data(
    X,
    y
)