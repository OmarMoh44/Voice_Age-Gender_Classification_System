import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
import os

class FeatureAnalyzer:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.feature_names = feature_extractor.feature_names
        self.feature_groups = feature_extractor.feature_groups
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

    def normalize_features(self, X_train, X_test=None):
        if not self.is_scaler_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_scaler_fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
            
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def save_scaler(self, filepath='scaler.joblib'):
        if not self.is_scaler_fitted:
            print("Warning: Saving unfitted scaler")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='scaler.joblib'):
        self.scaler = load(filepath)
        self.is_scaler_fitted = True
        print(f"Scaler loaded from {filepath}")
    
    def select_features_univariate(self, X, y, k=55, method='f_classif'):
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)
            
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        print(f"Selected {k} features using {method}:")
        for idx in selected_indices:
            print(f"- {self.feature_names[idx]}")
            
        return X_selected, selected_indices
    
    def select_features_tree_based(self, X, y, k=55):
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_indices = indices[:k]
        X_selected = X[:, selected_indices]
        
        print(f"Selected {k} features using Random Forest importance:")
        for idx in selected_indices:
            print(f"- {self.feature_names[idx]} (importance: {importances[idx]:.4f})")
            
        return X_selected, selected_indices
    
    def _process_feature_group(self, group, X_train, X_test, y_train, y_test):
        group_name, (start, end) = group
        X_group_train = X_train[:, start:end]
        X_group_test = X_test[:, start:end]
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_group_train, y_train)
        y_pred = rf.predict(X_group_test)
        return group_name, accuracy_score(y_test, y_pred)
    
    def analyze_feature_groups(self, X, y, n_jobs=-1):
        # First split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Then normalize using only training data for fitting
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_feature_group)(
                group, X_train_norm, X_test_norm, y_train, y_test
            ) for group in self.feature_groups.items()
        )
        
        group_scores = {}
        for group_name, accuracy in results:
            group_scores[group_name] = accuracy
            print(f"{group_name} features accuracy: {accuracy:.4f}")
            
        return group_scores
    
    def plot_feature_importance(self, X, y, method='rf', n_top=25):
        plt.figure(figsize=(12, 8))
        
        if method == 'rf':
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            title = 'Feature Importance (Random Forest)'
        else:
            f_scores, _ = f_classif(X, y)
            importances = f_scores
            title = 'Feature Importance (F-score)'
        
        indices = np.argsort(importances)[-n_top:]
        plt.barh(range(n_top), importances[indices])
        plt.yticks(range(n_top), [self.feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{method}.png')
        plt.close()
        
    def plot_group_importance(self, group_scores):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(group_scores.keys()), y=list(group_scores.values()))
        plt.xlabel('Feature Group')
        plt.ylabel('Accuracy')
        plt.title('Feature Group Importance')
        plt.tight_layout()
        plt.savefig('feature_group_importance.png')
        plt.close()
        
    def plot_correlation_matrix(self, X):
        corr = np.corrcoef(X.T)
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, xticklabels=False, yticklabels=False)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        plt.close()
    
    def _train_importances(self, X, y_target):
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y_target)
        return rf.feature_importances_
      
    def run_full_analysis(self, X, y, save_model=True, n_jobs=-1):
        print("Starting feature analysis...")
        
        # Keep track of original indices to preserve order
        original_indices = np.arange(len(X))
        
        # Split data before any normalization
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, original_indices, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalize features separately for train and test
        X_train_norm, X_test_norm = self.normalize_features(X_train, X_test)
        
        # Save the fitted scaler if requested
        if save_model:
            self.save_scaler('models/scaler.joblib')
        
        print("\n=== Feature Group Analysis ===")
        # We'll use the already normalized data for analysis
        group_scores = self.analyze_feature_groups(np.vstack((X_train_norm, X_test_norm)), 
                                                 np.concatenate((y_train, y_test)), 
                                                 n_jobs=n_jobs)
        self.plot_group_importance(group_scores)
        
        print("\n=== Feature Importance Analysis ===")
        self.plot_feature_importance(X_train_norm, y_train, method='rf')
        
        print("\n=== Feature Correlation Analysis ===")
        self.plot_correlation_matrix(X_train_norm)
        
        print("\n=== Feature Selection ===")
        # Select features using training data
        X_train_selected, selected_indices = self.select_features_tree_based(X_train_norm, y_train)
        X_test_selected = X_test_norm[:, selected_indices]
        
        print("\n=== Performance with Selected Features ===")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train_selected, y_train)
        y_pred = rf.predict(X_test_selected)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Male 20s', 'Female 20s', 'Male 50s', 'Female 50s']))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Male 20s', 'Female 20s', 'Male 50s', 'Female 50s'],
                   yticklabels=['Male 20s', 'Female 20s', 'Male 50s', 'Female 50s'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Create combined normalized and selected data in original order
        X_norm_combined = np.zeros_like(X, dtype=float)
        X_norm_combined[train_indices] = X_train_norm
        X_norm_combined[test_indices] = X_test_norm
        
        # Create combined selected features in original order
        X_selected_combined = X_norm_combined[:, selected_indices]
        
        if save_model:
            # Save selected indices
            np.save('models/selected_indices.npy', selected_indices)
            
            # Save the trained model
            dump(rf, 'models/analyzer.joblib')
            print("Analyzer saved to models/analyzer.joblib")
            
            # Save list of selected features
            os.makedirs('models', exist_ok=True)
            with open('models/selected_features.txt', 'w') as f:
                for idx in selected_indices:
                    f.write(f"{self.feature_names[idx]}\n")
            
            # Save selected features in original order
            np.save('models/X_selected.npy', X_selected_combined)
            np.save('models/y.npy', y)
        
        return X_selected_combined, y, selected_indices
