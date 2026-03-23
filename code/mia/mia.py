# project/mica/mica.py
import numpy as np
from .distance_matrix import pairwise_euclidean
from .hypersphere_analysis import impurities_at_radius, radii_sequence
from .impurity_measures import regression_impurity_variance, gini_impurity
from typing import Dict, Any, Callable, Optional

class MIA:
    """
    Implementation of the Multiscale Impurity Complexity Analysis (MICA).
    This version supports a fit/predict API.
    """

    def __init__(self, n_radii: int = 10, impurity_fn: Optional[Callable] = None, task: str = "classification",raddi_spacing: str = "log"):
        self.n_radii = n_radii
        if impurity_fn:
            self.impurity_fn = impurity_fn
        else:
            self.impurity_fn = regression_impurity_variance if task == "regression" else gini_impurity
        
        # Fitted attributes with trailing underscore
        self.X_train_ = None
        self.y_train_ = None
        self.radii_ = None
        self.max_impurity_ = None
        self.train_impurities_ = None
        self.is_fitted_ = False
        self.task = task
        self.raddi_spacing = raddi_spacing

        # Scores from the last fit_predict call
        self.local_scores_ = None
        self.CS_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the MICA model to the training data.

        This method learns the sequence of radii and the normalization constant
        (max_impurity_) from the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_train_ = X
        self.y_train_ = y

        self.y_unique_ = np.unique(y)
        D = pairwise_euclidean(X)
        max_dist = float(np.max(D))
        #print(f"Max pairwise distance in training data: {max_dist:.4f}")
        #max_dist=max_dist/2 # changed now to be a radius
        if max_dist <= 0:
            self.radii_ = np.ones(self.n_radii) * 0.001
        else:
            self.radii_ = radii_sequence(max_dist, self.n_radii,self.raddi_spacing)
        
        if self.task == "classification":
            self.max_impurity_by_class_ = np.ones(len(self.y_unique_))

            for class_index, class_label in enumerate(self.y_unique_):
                mask = y == class_label
                y_local = y.copy()
                y_local[mask] = 0
                y_local[~mask] = 1

                train_impurities = np.zeros((self.n_radii, X.shape[0]))
                for idx, r in enumerate(self.radii_):
                    train_impurities[idx, :] = impurities_at_radius(
                        X, X, y_local, r, impurity_fn=self.impurity_fn
                    )
                local_avg_impurity = np.mean(train_impurities, axis=0)
                self.max_impurity_by_class_[class_index] = np.max(local_avg_impurity)
        
        if self.task == "regression": # if for std still to implement with the training impurities
            if self.impurity_fn == regression_impurity_variance and y.size > 1:
                #Use Popoviciu's inequality on variance for a robust upper bound.
                # This prevents dependence on a single outlier's max local impurity. 
                #max_impurity = ((np.max(y) - np.min(y))**2) / 4.0
                max_impurity = 0.25 # If the values are normalized between 0 and 1, the max variance is 0.25
        
            if max_impurity > 1e-9:
                self.max_impurity_ = max_impurity
            else:
                self.max_impurity_ = 1.0  # Avoid division by zero

        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute local complexity scores for new data X.

        Parameters
        ----------
        X : np.ndarray
            New data of shape (n_samples, n_features).

        Returns
        -------
        local_complexity : np.ndarray
            Array of local complexity scores of shape (n_samples,).
        """
        if not self.is_fitted_:
            raise RuntimeError("This MICA instance is not fitted yet. Call 'fit' with appropriate data.")

        if self.task == "classification":
            local_complexity = np.zeros(shape=(X.shape[0], self.y_unique_.shape[0]))
            for class_index, class_label in enumerate(self.y_unique_):
                impurities = np.zeros((self.n_radii, X.shape[0]))
                y_local = self.y_train_.copy()
                y_local[self.y_train_ == class_label] = 0
                y_local[self.y_train_ != class_label] = 1

                for idx, r in enumerate(self.radii_):
                    impurities[idx, :] = impurities_at_radius(
                        X, self.X_train_, y_local, r, impurity_fn=self.impurity_fn
                    )
                
                local_avg_impurity = np.mean(impurities, axis=0)
                
                local_complexity_class = local_avg_impurity / self.max_impurity_by_class_[class_index]
                local_complexity_class = np.clip(local_complexity_class, 1e-8, 1)

                local_complexity[:, class_index] = local_complexity_class

            return local_complexity

        if self.task == "regression":
            impurities = np.zeros((self.n_radii, X.shape[0]))
            for idx, r in enumerate(self.radii_):
                impurities[idx, :] = impurities_at_radius(
                    X, self.X_train_, self.y_train_, r, impurity_fn=self.impurity_fn
                )
            
            local_avg_impurity = np.mean(impurities, axis=0)
            
            local_complexity = local_avg_impurity / self.max_impurity_
            local_complexity = np.clip(local_complexity, 1e-8, 1)

        return local_complexity

    def fit_predict(self, X: np.ndarray, y: np.ndarray, return_all: bool = False) -> Dict[str, Any]:
        """
        Fit the model and return complexity scores for the training data.

        This is a convenience method that combines fit and predict.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).
        return_all : bool, optional
            If True, return a dict with detailed scores, by default False.

        Returns
        -------
        Dict[str, Any]
            Dictionary with local scores and global complexity score (CS).
        """
        self.fit(X, y)
        
        local_avg_impurity = np.mean(self.train_impurities_, axis=0)
        local_complexity = local_avg_impurity / self.max_impurity_
        local_complexity = np.clip(local_complexity, 0, 1)

        self.local_scores_ = local_complexity
        self.CS_ = float(np.mean(local_complexity))

        if return_all:
            return {
                "local_scores": self.local_scores_,
                "CS": self.CS_,
                "impurities_by_radius": self.train_impurities_,
                "radii": self.radii_
            }

        return {"local_scores": self.local_scores_, "CS": self.CS_}