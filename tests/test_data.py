import numpy as np
import pandas as pd
import pytest

from modules.problem_detector import ProblemDetector


def test_target_must_be_explicit_not_first_binary_feature():
    df = pd.DataFrame({'A0': [0, 1, 0, 1], 'class': [1, 1, 0, 0]})
    with pytest.raises(ValueError, match='explicit.*target'):
        ProblemDetector(df).analyze()
