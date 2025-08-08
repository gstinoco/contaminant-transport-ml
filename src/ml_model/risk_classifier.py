"""
Ecological Risk Classifier Module
Machine Learning Models for Environmental Risk Assessment

Description:
    This module implements comprehensive machine learning models for classifying
    environmental zones according to their ecological risk level based on
    contaminant concentration patterns. It provides a complete framework for
    training, evaluating, and deploying classification models in environmental
    risk assessment applications.
    
    Key functionalities:
    - Multi-algorithm classification (Random Forest, SVM, Gradient Boosting, Logistic Regression)
    - Hyperparameter optimization using GridSearchCV
    - Cross-validation model evaluation
    - Feature importance analysis
    - Performance metrics calculation and visualization
    - Model persistence and deployment capabilities
    - Risk level prediction for new scenarios

All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    Secretary of Science, Humanities, Technology and Innovation, SECIHTI (Secretaria de Ciencia, Humanidades, Tecnología e Innovación). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
    SIIIA-MATH: Soluciones de Ingeniería. México

Date:
    February, 2025.

Last Modification:
    August, 2025.
"""

# Standard libraries
import os
from typing import Dict, List, Tuple, Any

# Third-party libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, precision_recall_fscore_support,
                           precision_score, recall_score)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

class RiskClassifier:
    """
    Classifier for ecological risk levels.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 
                                     '../../config/parameters.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.ml_config = self.config['ml_parameters']
        self.models = {}
        self.best_model = None
        self.feature_names = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize machine learning models based on configuration.
        
        This method creates instances of the machine learning algorithms
        specified in the configuration file, including RandomForest, SVM,
        LogisticRegression, and GradientBoosting classifiers with their
        respective hyperparameters.
        
        Note:
            Models are stored in self.models dictionary with algorithm
            names as keys. Parameters are loaded from the ml_parameters
            section of the configuration file.
        """
        algorithms = self.ml_config['algorithms']
        
        if 'RandomForest' in algorithms:
            rf_params = self.ml_config.get('random_forest', {})
            self.models['RandomForest'] = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', 10),
                min_samples_split=rf_params.get('min_samples_split', 5),
                min_samples_leaf=rf_params.get('min_samples_leaf', 2),
                random_state=self.ml_config['random_state']
            )
        
        if 'SVM' in algorithms:
            self.models['SVM'] = SVC(
                kernel='rbf',
                random_state=self.ml_config['random_state'],
                probability=True
            )
        
        if 'GradientBoosting' in algorithms:
            self.models['GradientBoosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.ml_config['random_state']
            )
        
        if 'LogisticRegression' in algorithms:
            self.models['LogisticRegression'] = LogisticRegression(
                random_state=self.ml_config['random_state'],
                max_iter=1000
            )
    

    
    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Evaluate all initialized models using k-fold cross-validation.
        
        This method performs cross-validation on all available models to assess
        their performance before final training. It provides mean accuracy scores
        and standard deviations to help identify the best performing algorithm.
        
        Args:
            X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features)
            y_train (np.ndarray): Training target vector of shape (n_samples,)
            
        Returns:
            Dict[str, float]: Dictionary containing cross-validation results for each model:
                - Keys: Model names (e.g., 'RandomForest', 'SVM')
                - Values: Dict with 'mean', 'std', and 'scores' keys
                
        Note:
            The number of cross-validation folds is specified in the configuration
            file under ml_parameters.cross_validation_folds.
        """
        cv_scores = {}
        cv_folds = self.ml_config['cross_validation_folds']
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                scores = cross_val_score(model, X_train, y_train, 
                                        cv=cv_folds, scoring='accuracy')
                cv_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                cv_scores[name] = {'mean': 0, 'std': 0, 'scores': []}
        
        return cv_scores
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             model_name: str = 'RandomForest') -> Dict[str, Any]:
        """
        Optimize hyperparameters using GridSearchCV for the specified model.
        
        This method performs exhaustive search over specified parameter values
        to find the optimal hyperparameter combination that maximizes model
        performance using cross-validation.
        
        Args:
            X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features)
            y_train (np.ndarray): Training target vector of shape (n_samples,)
            model_name (str, optional): Name of the model to optimize. 
                                      Supported: 'RandomForest', 'SVM', 'LogisticRegression'.
                                      Defaults to 'RandomForest'.
            
        Returns:
            Dict[str, Any]: Dictionary containing optimization results:
                - 'best_params': Best hyperparameter combination found
                - 'best_score': Best cross-validation score achieved
                - 'best_estimator': Trained model with optimal parameters
                
        Raises:
            ValueError: If model_name is not supported
        """

    
        if model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=self.ml_config['random_state'])
        
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            base_model = SVC(random_state=self.ml_config['random_state'], probability=True)
        
        elif model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
            base_model = GradientBoostingClassifier(random_state=self.ml_config['random_state'])
        
        else:
            print(f"Optimization not implemented for {model_name}")
            return {}
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=self.ml_config['cross_validation_folds'],
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return results
    
    def train_best_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        cv_scores: Dict[str, float]) -> str:
        """
        Select and train the best performing model based on cross-validation results.
        
        This method identifies the model with the highest mean cross-validation
        score and trains it on the full training dataset. The trained model
        is stored as the best_model for future predictions.
        
        Args:
            X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features)
            y_train (np.ndarray): Training target vector of shape (n_samples,)
            cv_scores (Dict[str, float]): Cross-validation scores from evaluate_models()
                                        containing mean scores for each model
            
        Returns:
            str: Name of the best performing model that was selected and trained
            
        Note:
            The trained model is accessible via self.best_model attribute after
            calling this method.
        """
        # Find the best model
        best_model_name = max(cv_scores.keys(), 
                             key=lambda x: cv_scores[x]['mean'])
        
        # Train the best model
        self.best_model = self.models[best_model_name]
        self.best_model.fit(X_train, y_train)
        
        return best_model_name
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the best trained model on the test dataset.
        
        This method performs comprehensive evaluation of the best model using
        the test set, computing various classification metrics including accuracy,
        precision, recall, F1-score, confusion matrix, and detailed per-class metrics.
        
        Args:
            X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features)
            y_test (np.ndarray): Test target vector of shape (n_samples,)
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation metrics including:
                - 'accuracy': Overall classification accuracy
                - 'precision': Weighted precision score
                - 'recall': Weighted recall score
                - 'f1': Weighted F1-score
                - 'classification_report': Detailed per-class metrics
                - 'confusion_matrix': Confusion matrix as numpy array
                - 'predictions': Model predictions on test set
                - 'probabilities': Prediction probabilities for each class
                
        Raises:
            ValueError: If no trained model is available (call train_best_model first)
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['Low', 'Medium', 'High'],
                                           output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        return results
    
    def _calculate_model_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics for a model.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # General metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Complete classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['Low', 'Medium', 'High'],
                                           output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }

    def evaluate_all_models_on_test(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all available models on the test set for comparison.
        
        This method trains each initialized model on the training data and
        evaluates them on the test set, providing comprehensive metrics for
        model comparison and selection. It generates a complete performance
        report for all algorithms.
        
        Args:
            X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features)
            y_train (np.ndarray): Training target vector of shape (n_samples,)
            X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features)
            y_test (np.ndarray): Test target vector of shape (n_samples,)
            
        Returns:
            Dict[str, Dict[str, Any]]: Nested dictionary with model names as keys
                and evaluation metrics as values. Each model's metrics include:
                - 'accuracy': Overall classification accuracy
                - 'precision_macro/weighted': Macro and weighted precision scores
                - 'recall_macro/weighted': Macro and weighted recall scores
                - 'f1_macro/weighted': Macro and weighted F1-scores
                - 'precision_per_class': Per-class precision scores
                - 'recall_per_class': Per-class recall scores
                - 'f1_per_class': Per-class F1-scores
                - 'classification_report': Detailed sklearn classification report
                - 'confusion_matrix': Confusion matrix
                - 'predictions': Model predictions on test set
                
        Note:
            This method is useful for comprehensive model comparison before
            selecting the final model for deployment.
        """
        all_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate all metrics using auxiliary function
                all_results[model_name] = self._calculate_model_metrics(y_test, y_pred, y_pred_proba)
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                all_results[model_name] = {
                    'error': str(e),
                    'accuracy': 0,
                    'precision_macro': 0,
                    'recall_macro': 0,
                    'f1_macro': 0
                }
        
        return all_results
    
    def generate_metrics_report(self, all_results: Dict[str, Dict[str, Any]], 
                               save_path: str = None) -> pd.DataFrame:
        """
        Generate a tabulated report with all metrics for all models.
        
        Args:
            all_results: Evaluation results for all models
            save_path: Path to save the report
            
        Returns:
            DataFrame with metrics report
        """
        report_data = []
        
        for model_name, results in all_results.items():
            if 'error' not in results:
                row = {
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision (Macro)': results['precision_macro'],
                    'Recall (Macro)': results['recall_macro'],
                    'F1-Score (Macro)': results['f1_macro'],
                    'Precision (Weighted)': results['precision_weighted'],
                    'Recall (Weighted)': results['recall_weighted'],
                    'F1-Score (Weighted)': results['f1_weighted'],
                    'Precision Low': results['precision_per_class'][0],
                    'Recall Low': results['recall_per_class'][0],
                    'F1 Low': results['f1_per_class'][0],
                    'Precision Medium': results['precision_per_class'][1],
                    'Recall Medium': results['recall_per_class'][1],
                    'F1 Medium': results['f1_per_class'][1],
                    'Precision High': results['precision_per_class'][2],
                    'Recall High': results['recall_per_class'][2],
                    'F1 High': results['f1_per_class'][2]
                }
            else:
                row = {
                    'Model': model_name,
                    'Error': results['error']
                }
            
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        
        # Save if path is specified
        if save_path:
            df_report.to_csv(save_path, index=False, float_format='%.4f')
        
        return df_report
    

    
    def plot_feature_importance(self, save_path: str = None):
        """
        Visualize feature importance.
        
        Args:
            save_path: Path to save the figure
        """
        if self.best_model is None:
            print("No trained model available")
            return
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("Model does not provide feature importance")
            return
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Visualize using configuration parameters
        viz_config = self.config.get('visualization', {})
        fig_width, fig_height = viz_config.get('figure_size', [10, 8])
        dpi = viz_config.get('dpi', 300)
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()
        
        return feature_importance_df
    
    def save_model(self, model_path: str, results: Dict[str, Any] = None):
        """
        Save the trained model and results.
        
        Args:
            model_path: Path to save the model
            results: Evaluation results
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump({
            'model': self.best_model,
            'feature_names': self.feature_names,
            'config': self.config,
            'results': results
        }, model_path)
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.best_model = model_data['model']
        self.feature_names = model_data['feature_names']
    
    def predict_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict ecological risk levels for new input samples.
        
        This method uses the trained best model to classify new samples into
        risk categories (Low, Medium, High) and provides prediction probabilities
        for each class, enabling uncertainty quantification.
        
        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features)
                          containing the same features used during training
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - predictions: Array of predicted risk levels (0=Low, 1=Medium, 2=High)
                - probabilities: Array of prediction probabilities for each class
                               with shape (n_samples, n_classes)
                               
        Raises:
            ValueError: If no trained model is available (call train_best_model first)
            
        Example:
            >>> predictions, probabilities = classifier.predict_risk(new_data)
            >>> print(f"Predicted risk: {predictions[0]}")  # 0, 1, or 2
            >>> print(f"Confidence: {probabilities[0].max():.2f}")
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities