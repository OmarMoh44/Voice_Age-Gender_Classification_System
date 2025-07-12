from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from xgboost import XGBClassifier
import numpy as np
from joblib import Parallel, delayed, dump, load
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import randint, uniform

class ModelTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y.astype(int)  # Ensure y is integer type for classification
        self.test_size = test_size
        self.random_state = random_state
        self.n_classes = len(np.unique(y))
        self.models = {}
        self.results = {}
        self._initialize_default_models()
        self.use_smote = False
        self.use_class_weights = False
        self.param_distributions = {}
        self._initialize_default_param_distributions()
        
    def _initialize_default_models(self):
        """Create default classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(
                objective='multi:softmax',
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=self.random_state
            )
        }
    
    def _initialize_default_param_distributions(self):
        """Define default parameter distributions for RandomizedSearchCV"""
        self.param_distributions = {
            'Logistic Regression': {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': randint(500, 2000)
            },
            'Random Forest': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20),
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'SVM': {
                'C': uniform(0.1, 20),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5)),
                'degree': randint(2, 6)
            },
            'K-Nearest Neighbors': {
                'n_neighbors': randint(3, 30),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean
            },
            'XGBoost': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5),
                'gamma': uniform(0, 5),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
        }
    
    
    def set_param_distribution(self, model_name, param_dist):   # param_dist (dict): Parameter distribution for RandomizedSearchCV
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in the trainer")
        
        self.param_distributions[model_name] = param_dist
        print(f"Custom parameter distribution set for {model_name}")
    
    def enable_smote(self, sampling_strategy='auto', k_neighbors=5):
        self.use_smote = True
        
        # Determine the minimum safe number of neighbors
        class_counts = np.bincount(self.y)
        min_class_count = min(class_counts[c] for c in range(len(class_counts)) if class_counts[c] > 0)
        
        # Adjust k_neighbors if needed
        if min_class_count <= k_neighbors:
            safe_k = min_class_count - 1 if min_class_count > 1 else 1
            print(f"Warning: k_neighbors={k_neighbors} is too large for smallest class with {min_class_count} samples.")
            print(f"Automatically adjusting k_neighbors to {safe_k}")
            k_neighbors = safe_k
        
        self.smote_params = {
            'sampling_strategy': sampling_strategy,
            'k_neighbors': k_neighbors,
            'random_state': self.random_state
        }
        
    def enable_class_weights(self, weight_strategy='balanced'):
        """
        Enable class weights to handle imbalance
        Args:
            weight_strategy (str): Strategy for class weights
                - 'balanced': inversely proportional to class frequencies
                - 'balanced_subsample': (for Random Forest only) balanced with bootstrapping
        """
        self.use_class_weights = True
        self.weight_strategy = weight_strategy
        
        # Update models that support class_weight
        for name, model in self.models.items():
            if hasattr(model, 'class_weight'):
                if name == 'Random Forest' and weight_strategy == 'balanced_subsample':
                    model.set_params(class_weight='balanced_subsample')
                else:
                    model.set_params(class_weight='balanced')
            elif name == 'XGBoost':
                # XGBoost uses a different approach for class weighting
                class_counts = np.bincount(self.y)
                total_samples = len(self.y)
                scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
                
                if len(class_counts) == 2:  # Binary classification
                    model.set_params(scale_pos_weight=scale_pos_weight)
                else:  # Multi-class classification
                    # For multi-class, we compute weights for each class
                    weights = total_samples / (self.n_classes * class_counts)
                    model.set_params(sample_weight=weights)
    
    def add_model(self, name, model, param_dist=None):
        self.models[name] = model
        
        # Apply class weights if enabled and supported by the model
        if self.use_class_weights and hasattr(model, 'class_weight'):
            if name == 'Random Forest' and self.weight_strategy == 'balanced_subsample':
                model.set_params(class_weight='balanced_subsample')
            else:
                model.set_params(class_weight='balanced')
        
        # Set parameter distribution if provided
        if param_dist is not None:
            self.param_distributions[name] = param_dist
        
    def _train_single_model(self, name, model, X_train, X_test, y_train, y_test, cv, use_random_search, n_iter, scoring):
        """Helper function to train and evaluate a single model"""
        try:
            cloned_model = clone(model)
            
            # Use RandomizedSearchCV if enabled and parameter distribution is available
            if use_random_search and name in self.param_distributions:
                print(f"Performing RandomizedSearchCV for {name}...")
                random_search = RandomizedSearchCV(
                    cloned_model,
                    param_distributions=self.param_distributions[name],
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
                random_search.fit(X_train, y_train)
                
                # Get the best model
                best_model = random_search.best_estimator_
                cv_mean_accuracy = random_search.best_score_
                cv_std_accuracy = np.std(random_search.cv_results_['split0_test_score'])
                
                print(f"Best parameters for {name}: {random_search.best_params_}")
                print(f"Best CV score for {name}: {cv_mean_accuracy:.4f}")
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                return {
                    'name': name,
                    'model': best_model,
                    'cv_mean_accuracy': cv_mean_accuracy,
                    'cv_std_accuracy': cv_std_accuracy,
                    'test_accuracy': test_accuracy,
                    'classification_report': report,
                    'confusion_matrix': conf_matrix,
                    'best_params': random_search.best_params_
                }
            else:
                # Regular cross-validation if RandomizedSearchCV is not used
                cv_scores = cross_val_score(
                    cloned_model, X_train, y_train, 
                    cv=cv, scoring=scoring
                )
                
                # Final training and evaluation
                cloned_model.fit(X_train, y_train)
                y_pred = cloned_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Get more detailed metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                return {
                    'name': name,
                    'model': cloned_model,
                    'cv_mean_accuracy': np.mean(cv_scores),
                    'cv_std_accuracy': np.std(cv_scores),
                    'test_accuracy': test_accuracy,
                    'classification_report': report,
                    'confusion_matrix': conf_matrix
                }
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            return None
        
    def train_models(self, cv=5, n_jobs=-1, use_random_search=False, n_iter=20, scoring='accuracy'):
        """
        Train and evaluate all models, optionally using RandomizedSearchCV
        
        Args:
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs (-1 for all cores)
            use_random_search (bool): Whether to use RandomizedSearchCV for hyperparameter tuning
            n_iter (int): Number of parameter combinations to try with RandomizedSearchCV
            scoring (str): Scoring metric to use for model evaluation
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"Training data shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
        
        # Apply SMOTE once here if enabled, instead of in each model training function
        if self.use_smote:
            try:
                smote = SMOTE(**self.smote_params)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Training data shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
            except ValueError as e:
                if "Expected n_neighbors <= n_samples" in str(e):
                    # Count per class in training set
                    train_class_counts = np.bincount(y_train)
                    min_train_class = min(train_class_counts[c] for c in range(len(train_class_counts)) if train_class_counts[c] > 0)
                    
                    # Dynamically adjust k_neighbors
                    safe_k = min_train_class - 1 if min_train_class > 1 else 1
                    print(f"Adjusting k_neighbors from {self.smote_params['k_neighbors']} to {safe_k}")
                    
                    # Create new SMOTE instance with adjusted k_neighbors
                    safe_params = self.smote_params.copy()
                    safe_params['k_neighbors'] = safe_k
                    smote = SMOTE(**safe_params)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"After SMOTE - Training data shape: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
                else:
                    # Re-raise if it's a different error
                    print(f"SMOTE error: {str(e)}. Continuing with original data.")
        
        if use_random_search:
            print(f"Using RandomizedSearchCV with {n_iter} iterations per model")
        
        # Process models in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_single_model)(
                name, model, X_train, X_test, y_train, y_test, cv, use_random_search, n_iter, scoring
            ) for name, model in self.models.items()
        )
        
        # Process results
        self.results = {}
        for result in results:
            if result is not None:
                result_dict = {
                    'model': result['model'],
                    'cv_mean_accuracy': result['cv_mean_accuracy'],
                    'cv_std_accuracy': result['cv_std_accuracy'],
                    'test_accuracy': result['test_accuracy'],
                    'classification_report': result['classification_report'],
                    'confusion_matrix': result['confusion_matrix']
                }
                
                # Add best parameters if available
                if 'best_params' in result:
                    result_dict['best_params'] = result['best_params']
                
                self.results[result['name']] = result_dict
                
                # Update model with trained version
                self.models[result['name']] = result['model']

    def print_results(self):
        """Print formatted results of all model evaluations"""
        print("{:<20} {:<15} {:<15} {:<15}".format(
            'Model', 'CV Mean Acc', 'CV Std Acc', 'Test Acc'))
        print("-" * 65)
        
        for name, result in sorted(self.results.items(), 
                                 key=lambda x: x[1]['cv_mean_accuracy'], 
                                 reverse=True):
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                name,
                result['cv_mean_accuracy'],
                result['cv_std_accuracy'],
                result['test_accuracy']
            ))
            
            # Print best parameters if available
            if 'best_params' in result:
                print(f"  Best parameters: {result['best_params']}")
                print()
    
    def get_best_model(self, metric='cv_mean_accuracy'):
        if not self.results:
            raise ValueError("Run train_models() first")
            
        best_name = max(self.results.items(), 
                        key=lambda x: x[1][metric])[0]
        return best_name, self.results[best_name]['model'], self.results[best_name][metric]
    
    def save_models(self, directory='models'):
        if not self.results:
            raise ValueError("Run train_models() first")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        for name, result in self.results.items():
            model_path = os.path.join(directory, f"{name.replace(' ', '_').lower()}.joblib")
            dump(result['model'], model_path)
            print(f"Model saved to {model_path}")
        
        # Save best parameters if available
        best_params = {}
        for name, result in self.results.items():
            if 'best_params' in result:
                best_params[name] = result['best_params']
        
        if best_params:
            best_params_path = os.path.join(directory, "best_params.joblib")
            dump(best_params, best_params_path)
            print(f"Best parameters saved to {best_params_path}")
        
        # Save a summary of results
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'cv_mean_accuracy': result['cv_mean_accuracy'],
                'cv_std_accuracy': result['cv_std_accuracy'],
                'test_accuracy': result['test_accuracy']
            }
        
        results_path = os.path.join(directory, "results_summary.joblib")
        dump(results_summary, results_path)
        print(f"Results summary saved to {results_path}")
    
    def plot_confusion_matrices(self, figsize=(15, 10)):
        """Plot confusion matrices for all trained models"""
        if not self.results:
            raise ValueError("Run train_models() first")
            
        n_models = len(self.results)
        fig, axes = plt.subplots(nrows=(n_models+1)//2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(sorted(self.results.items(), 
                                              key=lambda x: x[1]['cv_mean_accuracy'], 
                                              reverse=True)):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(self.y),
                       yticklabels=np.unique(self.y),
                       ax=axes[i])
            axes[i].set_title(f"{name} (Acc: {result['test_accuracy']:.4f})")
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        return fig

    def plot_class_distribution(self, figsize=(10, 6)):
        """Plot class distribution before and after SMOTE (if enabled)"""
        X_train, _, y_train, _ = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        # Original distribution
        original_counts = np.bincount(y_train)
        
        # If SMOTE is enabled, get the resampled distribution
        if self.use_smote:
            smote = SMOTE(**self.smote_params)
            _, y_resampled = smote.fit_resample(X_train, y_train)
            resampled_counts = np.bincount(y_resampled)
            
            # Plot
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].bar(range(len(original_counts)), original_counts)
            ax[0].set_title('Original Class Distribution')
            ax[0].set_xlabel('Class')
            ax[0].set_ylabel('Count')
            
            ax[1].bar(range(len(resampled_counts)), resampled_counts)
            ax[1].set_title('After SMOTE Resampling')
            ax[1].set_xlabel('Class')
            ax[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('class_distribution.png')
            return fig
        else:
            # Only plot original distribution
            fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
            ax.bar(range(len(original_counts)), original_counts)
            ax.set_title('Class Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            plt.savefig('class_distribution.png')
            return fig
            
    def plot_parameter_importance(self, model_name=None):
        if not self.results:
            raise ValueError("Run train_models() first")
            
        if model_name is None:
            model_name = self.get_best_model()[0]
            
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results")
            
        result = self.results[model_name]
        model = result['model']
        
        # Check if the model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance for {model_name}')
            plt.tight_layout()
            plt.savefig(f'{model_name.replace(" ", "_").lower()}_feature_importance.png')
            
        else:
            print(f"Model {model_name} does not support feature importance visualization")
            
        # For logistic regression, plot coefficients
        if isinstance(model, LogisticRegression):
            plt.figure(figsize=(10, 8))
            coef = model.coef_
            
            if coef.shape[0] == 1:  # Binary classification
                coefficients = coef[0]
                indices = np.argsort(np.abs(coefficients))[-20:]  # Top 20 features by magnitude
                plt.barh(range(len(indices)), coefficients[indices])
                plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
                plt.xlabel('Coefficient Value')
                plt.title(f'Feature Coefficients for {model_name}')
                
            else:  # Multi-class
                plt.figure(figsize=(12, 10))
                n_classes = coef.shape[0]
                
                for i in range(n_classes):
                    plt.subplot(n_classes, 1, i+1)
                    coefficients = coef[i]
                    indices = np.argsort(np.abs(coefficients))[-10:]  # Top 10 features by magnitude
                    plt.barh(range(len(indices)), coefficients[indices])
                    plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
                    plt.xlabel('Coefficient Value')
                    plt.title(f'Class {i} Coefficients')
                
                plt.tight_layout()
            
            plt.savefig(f'{model_name.replace(" ", "_").lower()}_coefficients.png')
