import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from modules.model_trainer import ModelTrainer


def test_fit_selects_using_training_cv_and_fits_preprocessor_per_fold():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X)
    trainer = ModelTrainer(
        {'logistic': LogisticRegression(), 'baseline': DummyClassifier()},
        task='classification', cv=3, random_state=42,
    )
    trainer.fit(X, pd.Series(y), preprocessor=StandardScaler())
    assert trainer.best_model_name_ in {'logistic', 'baseline'}
    assert set(trainer.results) == {'logistic', 'baseline'}
    assert trainer.scoring_ == 'f1_macro'
    assert trainer.best_pipeline_.predict(X).shape == (100,)
    # Fitted scalers from each CV fold must not have seen the entire training set.
    for fitted in trainer.cv_estimators_['logistic']:
        assert fitted.named_steps['preprocessing'].n_samples_seen_ < len(X)
    np.testing.assert_allclose(
        trainer.best_pipeline_.named_steps['preprocessing'].mean_, X.mean()
    )
