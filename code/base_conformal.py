from mia.mia import MIA
from time import time
import numpy as np
from scipy.stats import kstest
import pandas as pd
import warnings
from crepes.base import p_values_batch, ConformalPredictor


class ConformalPredictorMIA(ConformalPredictor):
    """
    Uses MIA implementation to provide conformity scores for conformal prediction.
    """

    def __init__(
        self, n_radii: int = 10, impurity_fn=None, task: str = "classification", raddi_spacing='log'
    ):
        super().__init__()
        self.task = task
        self.fn_impurity = impurity_fn
        self.mia = MIA(n_radii=n_radii, impurity_fn=impurity_fn,
                         task=task, raddi_spacing=raddi_spacing)
        self.mia_fitted = False
        self.mia_time_fit = None

    def fit_mia(self, X, y):
        """
        Fits the MIA model to the data.
        Parameters:
        -----------
        X: np.ndarray
            Feature matrix.
        y: np.ndarray
            Target vector.
        """
        tic = time()
        self.mia.fit(X, y)
        self.mia_fitted = True
        self.mia_time_fit = time() - tic
        return self


class ConformalClassifierMIA(ConformalPredictorMIA):
    """
    Conformal predictor for classification tasks using MIA.
    """

    def __init__(self, n_radii: int = 10, impurity_fn=None, raddi_spacing='log'):
        super().__init__(
            n_radii=n_radii, impurity_fn=impurity_fn, task="classification", raddi_spacing=raddi_spacing
        )

    def __repr__(self):
        return f"ConformalClassifier(n_radii={self.mia.n_radii}, impurity_fn={self.fn_impurity})"

    def fit(self, alphas, bins=None, seed=None):
        tic = time()  # place safetyguard for type
        self.alphas = alphas
        if bins is None:
            self.bins = None
            self.mondrian = False
        else:
            self.bins = bins
            self.mondrian = True
        self.seed = seed
        self.fitted = True
        self.fitted_ = True
        toc = time()
        self.time_fit = (toc - tic) + \
            self.mia_time_fit if self.mia_time_fit else toc - tic
        return self

    def alpha_weighting(
        self,
        X_values,
        X_prob,
        classes=None,
        y=None,
        hinge_margin="hinge",
        T=1.0,
    ):
        result = None
        if hinge_margin == "hinge":
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y.values
                class_indexes = np.array(
                    [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
                )
                result = 1 - X_prob[np.arange(len(y)), class_indexes]
            else:
                result = 1 - X_prob

        elif hinge_margin == "margin":  # margin
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y.values
                class_indexes = np.array(
                    [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
                )
                result = np.array(
                    [
                        (
                            np.max(
                                X_prob[
                                    i,
                                    [
                                        j != class_indexes[i]
                                        for j in range(X_prob.shape[1])
                                    ],
                                ]
                            )
                            - X_prob[i, class_indexes[i]]
                        )
                        for i in range(len(X_prob))
                    ]
                )
            else:
                result = np.array(
                    [
                        [
                            (
                                np.max(
                                    X_prob[i, [j != c for j in range(
                                        X_prob.shape[1])]]
                                )
                                - X_prob[i, c]
                            )
                            for c in range(X_prob.shape[1])
                        ]
                        for i in range(len(X_prob))
                    ]
                )

        elif hinge_margin == "complex_hinge":
            complexity = self.mia.predict(X_values)
            X_prob = (complexity**T)*X_prob
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y.values
                class_indexes = np.array(
                    [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
                )
                result = 1 - X_prob[np.arange(len(y)), class_indexes]
            else:
                result = 1 - X_prob

        elif hinge_margin == "complex_margin":  # margin
            complexity = self.mia.predict(X_values)
            X_prob = (complexity**T)*X_prob
            if y is not None:
                if isinstance(y, pd.Series):
                    y = y.values
                class_indexes = np.array(
                    [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
                )
                result = np.array(
                    [
                        (
                            np.max(
                                X_prob[
                                    i,
                                    [
                                        j != class_indexes[i]
                                        for j in range(X_prob.shape[1])
                                    ],
                                ]
                            )
                            - X_prob[i, class_indexes[i]]
                        )
                        for i in range(len(X_prob))
                    ]
                )
            else:
                result = np.array(
                    [
                        [
                            (
                                np.max(
                                    X_prob[i, [j != c for j in range(
                                        X_prob.shape[1])]]
                                )
                                - X_prob[i, c]
                            )
                            for c in range(X_prob.shape[1])
                        ]
                        for i in range(len(X_prob))
                    ]
                )

        else:
            raise ValueError(
                "hinge_margin must be 'hinge', 'margin', 'mia','mia_hinge' or 'mia_margin'"
            )
        return result

    def predict_p(
        self,
        alphas,
        bins=None,
        all_classes=True,
        classes=None,
        y=None,
        smoothing=True,
        seed=None,
    ):
        tic = time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        p_values = p_values_batch(
            self.alphas, alphas, self.bins, bins, smoothing, seed)
        if not all_classes:
            class_indexes = np.array(
                [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
            )
            p_values = p_values[np.arange(len(y)), class_indexes]
        toc = time()
        self.time_predict = toc - tic
        return p_values

    def predict_set(
        self, alphas, bins=None, confidence=0.95, smoothing=True, seed=None
    ):
        tic = time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        # if type(bins) == list:
        #    bins = np.array(bins)
        if seed is None:
            seed = self.seed
        p_values = p_values_batch(
            self.alphas, alphas, self.bins, bins, smoothing, seed)
        prediction_sets = (p_values >= 1 - confidence).astype(int)
        toc = time()
        self.time_predict = toc - tic
        return prediction_sets

    def evaluate(
        self,
        alphas,
        classes,
        y,
        confidence=0.95,
        smoothing=True,
        metrics=None,
        seed=None,
        online=False,
        warm_start=True,
        bins=None,
    ):
        if metrics is None:
            metrics = [
                "error",
                "avg_c",
                "one_c",
                "empty",
                "ks_test",
                "time_fit",
                "time_evaluate",
            ]
        tic = time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if type(classes) == list:
            classes = np.array(classes)
        if seed is None:
            seed = self.seed
        if not online:
            p_values = self.predict_p(
                alphas, bins, True, classes, y, smoothing, seed)
            prediction_sets = (p_values >= 1 - confidence).astype(int)
        # else:
        #    p_values = self.predict_p_online(alphas, classes, y, bins, True,
        #                                     smoothing, seed, warm_start)
        #    prediction_sets = (p_values >= 1-confidence).astype(int)
        test_results = get_classification_results(
            prediction_sets, p_values, classes, y)
        toc = time()
        self.time_evaluate = toc - tic
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results


def get_classification_results(prediction_sets, p_values, classes, y):
    test_results = {}
    class_indexes = np.array(
        [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
    )

    test_results["error"] = 1 - np.sum(
        prediction_sets[np.arange(len(y)), class_indexes]
    ) / len(y)

    test_results["avg_c"] = np.sum(prediction_sets) / len(y)

    test_results["one_c"] = np.sum(
        [np.sum(p) == 1 for p in prediction_sets]) / len(y)

    test_results["empty"] = np.sum(
        [np.sum(p) == 0 for p in prediction_sets]) / len(y)

    test_results["ks_test"] = kstest(
        p_values[np.arange(len(y)), class_indexes], "uniform"
    ).pvalue
    return test_results


class ConformalRegressorMIA(ConformalPredictorMIA):
    """
    Conformal predictor for regression tasks using MIA.
    """

    def __init__(self, n_radii: int = 10, impurity_fn=None, raddi_spacing='log'):
        super().__init__(n_radii=n_radii, impurity_fn=impurity_fn,
                         task="regression", raddi_spacing=raddi_spacing)

    def __repr__(self):
        return f"ConformalRegressor(n_radii={self.mia.n_radii}, impurity_fn={self.fn_impurity})"

    def fit(self, residuals, sigmas=None, bins=None):
        tic = time()
        if type(residuals) == list:
            residuals = np.array(residuals)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        abs_residuals = np.abs(residuals)
        if bins is None:
            self.mondrian = False
            self.bins = None
            if sigmas is None:
                self.normalized = False
                self.alphas = np.sort(abs_residuals)[::-1]
            else:
                self.normalized = True
                self.alphas = np.sort(abs_residuals/sigmas)[::-1]
        else:
            self.mondrian = True
            self.bins = bins
            if sigmas is None:
                self.alphas = abs_residuals
            else:
                self.alphas = abs_residuals/sigmas
            bin_values = np.unique(bins)
            if sigmas is None:
                self.normalized = False
                self.binned_alphas = (bin_values, [np.sort(
                    abs_residuals[bins == b])[::-1] for b in bin_values])
            else:
                self.normalized = True
                self.binned_alphas = (bin_values, [np.sort(
                    abs_residuals[bins == b]/sigmas[bins == b])[::-1]
                    for b in bin_values])
        self.fitted = True
        self.fitted_ = True
        toc = time()
        self.time_fit = (toc-tic)+self.mia_time_fit if self.mia_time_fit else toc - tic
        return self

    def predict_p(self, y_hat, y, sigmas=None, bins=None, smoothing=True,
                  seed=None):
        if not self.fitted:
            raise RuntimeError(("Batch predictions requires a fitted "
                                "conformal regressor"))
        tic = time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if sigmas is None:
            alphas = np.abs(y - y_hat)
        else:
            alphas = np.abs(y - y_hat)/sigmas
        p_values = p_values_batch(self.alphas, alphas, self.bins, bins,
                                  smoothing, seed)
        toc = time()
        self.time_predict = toc-tic
        return p_values

    def predict_int(self, y_hat, sigmas=None, bins=None, confidence=0.95,
                    y_min=-np.inf, y_max=np.inf):
        if not self.fitted:
            raise RuntimeError(("Batch predictions requires a fitted "
                                "conformal regressor"))
        tic = time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        intervals = np.zeros((len(y_hat), 2))
        if not self.mondrian:
            alpha_index = int((1-confidence)*(len(self.alphas)+1))-1
            if alpha_index >= 0:
                alpha = self.alphas[alpha_index]
                if self.normalized:
                    intervals[:, 0] = y_hat - alpha*sigmas
                    intervals[:, 1] = y_hat + alpha*sigmas
                else:
                    intervals[:, 0] = y_hat - alpha
                    intervals[:, 1] = y_hat + alpha
            else:
                intervals[:, 0] = -np.inf
                intervals[:, 1] = np.inf
                warnings.warn("the no. of calibration examples is too small"
                              "for the chosen confidence level; the "
                              "intervals will be of maximum size")
        else:
            bin_values, bin_alphas = self.binned_alphas
            bin_indexes = [np.argwhere(bins == b).T[0]
                           for b in bin_values]
            alpha_indexes = np.array(
                [int((1-confidence)*(len(bin_alphas[b])+1))-1
                 for b in range(len(bin_values))])
            too_small_bins = np.argwhere(alpha_indexes < 0)
            if len(too_small_bins) > 0:
                if len(too_small_bins[:, 0]) < 11:
                    bins_to_show = " ".join([str(bin_values[i]) for i in
                                             too_small_bins[:, 0]])
                else:
                    bins_to_show = " ".join([str(bin_values[i]) for i in
                                             too_small_bins[:10, 0]]+['...'])
                warnings.warn("the no. of calibration examples is too "
                              "small for the chosen confidence level "
                              f"in the following bins: {bins_to_show}; "
                              "the corresponding intervals will be of "
                              "maximum size")
            bin_alpha = np.array([bin_alphas[b][alpha_indexes[b]]
                                  if alpha_indexes[b] >= 0 else np.inf
                                  for b in range(len(bin_values))])
            if self.normalized:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b], 0] = y_hat[bin_indexes[b]] \
                        - bin_alpha[b]*sigmas[bin_indexes[b]]
                    intervals[bin_indexes[b], 1] = y_hat[bin_indexes[b]] \
                        + bin_alpha[b]*sigmas[bin_indexes[b]]
            else:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b], 0] = y_hat[bin_indexes[b]] \
                        - bin_alpha[b]
                    intervals[bin_indexes[b], 1] = y_hat[bin_indexes[b]] \
                        + bin_alpha[b]
        if y_min > -np.inf:
            intervals[intervals < y_min] = y_min
        if y_max < np.inf:
            intervals[intervals > y_max] = y_max
        toc = time()
        self.time_predict = toc-tic
        return intervals

    def evaluate(self, y_hat, y, sigmas=None, bins=None, confidence=0.95,
                 y_min=-np.inf, y_max=np.inf, metrics=None, smoothing=True,
                 seed=None, online=False, warm_start=True):
        if not self.fitted and not online:
            raise RuntimeError(("Batch evaluation requires a fitted "
                                "conformal regressor"))
        tic = time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if not online and not self.normalized:
            sigmas = None
        if not online and not self.mondrian:
            bins = None
        if metrics is None:
            metrics = ["error", "eff_mean", "eff_med", "ks_test",
                       "time_fit", "time_evaluate"]
        test_results = {}
        if not online:
            intervals = self.predict_int(y_hat, sigmas, bins, confidence,
                                         y_min, y_max)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(
                np.logical_and(intervals[:, 0] <= y, y <= intervals[:, 1]))
        if "eff_mean" in metrics:
            test_results["eff_mean"] = np.mean(
                intervals[:, 1] - intervals[:, 0])
        if "eff_med" in metrics:
            test_results["eff_med"] = np.median(
                intervals[:, 1] - intervals[:, 0])
        if "ks_test" in metrics:
            if not online:
                p_values = self.predict_p(y_hat, y, sigmas, bins, smoothing,
                                          seed)
            test_results["ks_test"] = kstest(p_values, "uniform").pvalue
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        toc = time()
        self.time_evaluate = toc-tic
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results