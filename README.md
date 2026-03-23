# Custom Conformal Prediction with MIA

This project is complementary to the paper

*Enhancing Conformal Prediction Efficiency via Multiscale Impurity Analysis: A Complexity-Aware Non-Conformity Function* 

and implements a novel method for conformal prediction called **Multiscale Impurity Analysis (MIA)**. It can be used for both classification and regression tasks to generate statistically rigorous prediction sets or intervals.

The core idea of MIA is to adapt the conformal prediction procedure to the local complexity of the data. Regions of the feature space with high complexity (i.e., where the label is hard to predict) will produce larger prediction sets (for classification) or wider prediction intervals (for regression).

## How it Works

The MIA method estimates the local complexity of a new instance by analyzing the impurity of its neighbors in the training data at multiple scales (hence "multiscale").

1.  For a new instance `x`, a set of nested hyperspheres with increasing radii are centered at `x`.
2.  Within each hypersphere, the impurity of the labels of the training instances is calculated. For classification, this can be Gini impurity and for regression, this can be the variance.
3.  The average impurity over all radii is used as a measure of the local complexity at `x`.
4.  This complexity measure is then used to modulate the conformity scores in the conformal prediction framework. For example, by weighting the non-conformity scores.

This repository provides the `ConformalClassifierMIA` and `ConformalRegressorMIA` classes that implement this logic, extending the `crepes` library.

## Installation

You can install the required packages using pip:

```bash
pip install -r code/requirements.txt
```

## Usage

The following sections demonstrate how to use the `ConformalClassifierMIA` and `ConformalRegressorMIA` for classification and regression tasks.

### Classification

```python
from base_conformal import ConformalClassifierMIA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from crepes import ConformalClassifier
from crepes.extras import hinge
import numpy as np

RANDOM_SEED = 0
epsilon = 0.1

X, y = load_iris(return_X_y=True)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
model = RandomForestClassifier(n_estimators=10,random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

model.fit(X_train, y_train)
probs_calib = model.predict_proba(X_calib)
probs_test = model.predict_proba(X_test)

# Standard Conformal Prediction
cp = ConformalClassifier()
alphas_cal = hinge(probs_calib,model.classes_, y_calib)
cp.fit(alphas_cal)
alphas_test = hinge(probs_test)

# Conformal Prediction with MIA
cp_mia = ConformalClassifierMIA()
cp_mia.fit_mia(X_train, y_train)
alphas_calib_mica = cp_mia.alpha_weighting(X_calib, probs_calib,classes=model.classes_, y=y_calib,hinge_margin="complex_hinge")
cp_mia.fit(alphas_calib_mica)
alphas_test_mica = cp_mia.alpha_weighting(X_test, probs_test, hinge_margin="complex_hinge")

print(f"CP: {cp.evaluate(alphas_test, model.classes_, y_test,confidence=1-epsilon)}")
print(f"CP-MIA: {cp_mia.evaluate(alphas_test_mica, model.classes_, y_test,confidence=1-epsilon)}")
```

#### UCEM Evaluation Metric

To evaluate the performance of the conformal classifiers, we use the **Uncertainty-aware Conformal Evaluation Metric (UCEM)**. This metric provides a more holistic evaluation of the prediction sets by considering not just the error rate, but also the size and correctness of the sets for each class.

The `UCEM` function calculates this metric. It constructs a confusion matrix for each class to evaluate the performance of the prediction sets and then computes a final score. This allows for a more granular analysis of the conformal predictor's performance.

### Regression

```python
from base_conformal import ConformalRegressorMIA
from sklearn.model_selection import train_test_split
from crepes import ConformalRegressor
from crepes.extras import DifficultyEstimator
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

RANDOM_SEED = 0
epsilon = 0.05

X,y = fetch_california_housing(return_X_y=True)
X = StandardScaler().fit_transform(X)
model = GradientBoostingRegressor(random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

model.fit(X_train, y_train)
y_hat_calib = model.predict(X_calib)
residuals_calib = y_calib - y_hat_calib
y_hat_test = model.predict(X_test)

# Standard Conformal Prediction
cr_std = ConformalRegressor()
cr_std.fit(residuals_calib)

# Conformal Prediction with MIA
mia_estimator = ConformalRegressorMIA()
mia_estimator.fit_mia(X_train, y_train)
mia_estimator.fit(residuals_calib,sigmas=mia_estimator.mia.predict(X_calib))
sigmas_test_mia = mia_estimator.mia.predict(X_test)

print(f"CP: {cr_std.evaluate(y_hat_test, y_test,y_min=0.0,y_max=1.0,confidence=1-epsilon)}")
print(f"CP MIA: {mia_estimator.evaluate(y_hat_test, y_test,sigmas = sigmas_test_mia,y_min=0.0,y_max=1.0,confidence=1-epsilon)}")
```

## Paper Results

The `results_classification` and `results_regression` directories contain the additional plots from the experiments mentioned in the paper.

### Classification Results (`results_classification/`)

This directory contains the results for the classification experiments.

-   `fsc_ssc/`: Plots for FSC and SSC metrics for each dataset, each classifier at different confidence levels.
-   `per_dataset/`: Standard Conformal Prediction metric plots for each dataset, each classifier for the different methods at different confidence levels.
-   `per_dataset_autorank/`: Autorank plots for each dataset, comparing the different methods on the standard conformal prediction metric at different confidence levels.
-   `performance_vector/`: Detailed performance metrics per dataset for the use of Acc/Acc or F1/Acc metrics on each dataset for each classifier at different confidence levels.

### Regression Results (`results_regression/`)

This directory contains the results for the regression experiments.

-   `per_dataset/`: Performance plots for each individual dataset on standard metrics, at different confidence levels.
-   `per_dataset_autorank/`: Autorank plots for each dataset, comparing the different methods, at different confidence levels.
-   `winkler_score/`: Plots of the Winkler score for each dataset for each regressor at different confidence levels.

## Authors

-   [Vasco Costa](https://github.com/vascovc)
-   [Diogo Costa](https://github.com/DiogoFDCosta)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

