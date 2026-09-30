import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

import evaluate


def _fitted_stack(processed, passthrough):
    df, _ = processed
    X = df.drop(columns=["price"]).astype(float)
    y = np.log1p(df["price"])
    stack = StackingRegressor(
        estimators=[("RidgeCV", RidgeCV()),
                    ("ExtraTrees", ExtraTreesRegressor(n_estimators=5, random_state=0))],
        final_estimator=RidgeCV(), cv=2, passthrough=passthrough,
    )
    return stack.fit(X, y), list(X.columns)


def test_stacking_weights_one_per_base_model(processed):
    stack, cols = _fitted_stack(processed, passthrough=False)
    w = evaluate.stacking_weights(stack, cols)

    assert list(w.index) == ["RidgeCV", "ExtraTrees"]
    assert w.notna().all()


def test_stacking_weights_with_passthrough_names_features(processed):
    stack, cols = _fitted_stack(processed, passthrough=True)
    w = evaluate.stacking_weights(stack, cols)

    assert list(w.index) == ["RidgeCV", "ExtraTrees"] + cols


def test_stacking_weights_passthrough_without_feature_names_raises(processed):
    stack, _ = _fitted_stack(processed, passthrough=True)
    with pytest.raises(ValueError, match="passthrough"):
        evaluate.stacking_weights(stack)
