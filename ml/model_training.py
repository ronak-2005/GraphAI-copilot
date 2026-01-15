"""
Complete Model Training & Evaluation Pipeline
Production-ready ML workflow with comprehensive evaluation and reporting
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

class Config:
    DATA_FOLDER = "../ml/ready_data/"
    MODELS_FOLDER = "../ml/trained_models/"
    RESULTS_FOLDER = "../ml/model_results/"
    
    TASK_TYPE = "auto"
    
    RANDOM_STATE = 42
    N_JOBS = -1
    CV_FOLDS = 5
    
    PERFORM_TUNING = False
    TUNING_METHOD = "random"
    TUNING_ITERATIONS = 10
    
    MODELS = {
        'linear_ridge': True,
        'lasso': True,
        'decision_tree': True,
        'random_forest': True,
        'gradient_boosting': True,
        'xgboost': XGBOOST_AVAILABLE,
    }

class MLPipeline:
    
    def __init__(self):
        self.log = {'timestamp': datetime.now().isoformat(), 'models': {}}
        self.models = {}
        self.results = {}
        
    def run(self):
        print("="*80)
        print("ML TRAINING PIPELINE")
        print("="*80)
        
        Path(Config.MODELS_FOLDER).mkdir(exist_ok=True, parents=True)
        Path(Config.RESULTS_FOLDER).mkdir(exist_ok=True, parents=True)
        
        self.load_data()
        self.train_baseline()
        self.define_models()
        self.train_all_models()
        self.compare_models()
        self.select_best()
        self.hyperparameter_tuning()
        self.evaluate_test()
        self.feature_importance()
        self.error_analysis()
        self.save_model()
        self.create_inference()
        self.generate_report()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
    
    def load_data(self):
        print("\nSTEP 1: Loading Data")
        self.X_train = pd.read_csv(f"{Config.DATA_FOLDER}X_train.csv")
        self.X_test = pd.read_csv(f"{Config.DATA_FOLDER}X_test.csv")
        
        y_train_df = pd.read_csv(f"{Config.DATA_FOLDER}y_train.csv")
        y_test_df = pd.read_csv(f"{Config.DATA_FOLDER}y_test.csv")
        
        self.y_train = y_train_df.iloc[:, 0].values
        self.y_test = y_test_df.iloc[:, 0].values
        
        if Config.TASK_TYPE == "auto":
            if pd.api.types.is_numeric_dtype(self.y_train):
                Config.TASK_TYPE = "regression"
            else:
                Config.TASK_TYPE = "classification"
            print(f"   Auto-detected task: {Config.TASK_TYPE}")
        
        if Config.TASK_TYPE == "regression":
            self.y_train = pd.to_numeric(self.y_train, errors='coerce')
            self.y_test = pd.to_numeric(self.y_test, errors='coerce')
            
            if np.isnan(self.y_train).any() or np.isnan(self.y_test).any():
                print(f"   Warning: Non-numeric values found in target. Converting to classification task.")
                Config.TASK_TYPE = "classification"
                self.y_train = y_train_df.iloc[:, 0].values
                self.y_test = y_test_df.iloc[:, 0].values
        
        if Config.TASK_TYPE == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train.astype(str))
            self.y_test = self.label_encoder.transform(self.y_test.astype(str))
            print(f"   Classes: {len(self.label_encoder.classes_)} unique values")
        
        print(f"   Train: {len(self.X_train):,} x {self.X_train.shape[1]} features")
        print(f"   Test:  {len(self.X_test):,} samples")
        print(f"   Task:  {Config.TASK_TYPE}")
        print(f"   Target dtype: {self.y_train.dtype}")
    
    def train_baseline(self):
        print("\nSTEP 2: Baseline Model")
        
        if Config.TASK_TYPE == "regression":
            if not np.issubdtype(self.y_train.dtype, np.number):
                print(f"   Error: y_train is not numeric (dtype: {self.y_train.dtype})")
                print(f"   Sample values: {self.y_train[:5]}")
                raise ValueError("Target variable must be numeric for regression")
            
            baseline_pred = np.full_like(self.y_test, float(self.y_train.mean()), dtype=float)
            self.baseline_score = r2_score(self.y_test, baseline_pred)
            print(f"   Baseline R2: {self.baseline_score:.4f}")
        else:
            from scipy.stats import mode
            most_common = mode(self.y_train, keepdims=True)[0][0]
            baseline_pred = np.full_like(self.y_test, most_common)
            self.baseline_score = accuracy_score(self.y_test, baseline_pred)
            print(f"   Baseline Accuracy: {self.baseline_score:.4f}")
    
    def define_models(self):
        print("\nSTEP 3: Defining Models")
        
        if Config.TASK_TYPE == "regression":
            all_models = {
                'linear_ridge': Ridge(random_state=Config.RANDOM_STATE),
                'lasso': Lasso(random_state=Config.RANDOM_STATE),
                'decision_tree': DecisionTreeRegressor(random_state=Config.RANDOM_STATE, max_depth=10),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS),
                'gradient_boosting': GradientBoostingRegressor(random_state=Config.RANDOM_STATE),
                'xgboost': XGBRegressor(random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS) if XGBOOST_AVAILABLE else None
            }
        else:
            all_models = {
                'logistic': LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000),
                'decision_tree': DecisionTreeClassifier(random_state=Config.RANDOM_STATE, max_depth=10),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS),
                'gradient_boosting': GradientBoostingClassifier(random_state=Config.RANDOM_STATE),
                'xgboost': XGBClassifier(random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS) if XGBOOST_AVAILABLE else None
            }
        
        self.models = {k: v for k, v in all_models.items() if v is not None and Config.MODELS.get(k, True)}
        print(f"   Selected: {list(self.models.keys())}")
    
    def train_all_models(self):
        print("\nSTEP 4-6: Training & Cross-Validation")
        
        for name, model in self.models.items():
            print(f"\n   Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                
                if Config.TASK_TYPE == "regression":
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                               cv=Config.CV_FOLDS, scoring='neg_mean_squared_error', n_jobs=Config.N_JOBS)
                    cv_rmse = np.sqrt(-cv_scores)
                    train_pred = model.predict(self.X_train)
                    
                    self.results[name] = {
                        'train_r2': r2_score(self.y_train, train_pred),
                        'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
                        'cv_rmse': cv_rmse.mean(),
                        'cv_std': cv_rmse.std()
                    }
                    print(f"      Train R2: {self.results[name]['train_r2']:.4f} | CV RMSE: {self.results[name]['cv_rmse']:.4f}")
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                               cv=Config.CV_FOLDS, scoring='accuracy', n_jobs=Config.N_JOBS)
                    train_pred = model.predict(self.X_train)
                    
                    self.results[name] = {
                        'train_acc': accuracy_score(self.y_train, train_pred),
                        'cv_acc': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    print(f"      Train Acc: {self.results[name]['train_acc']:.4f} | CV Acc: {self.results[name]['cv_acc']:.4f}")
                
                self.models[name] = model
            except Exception as e:
                print(f"      Error: {e}")
    
    def compare_models(self):
        print("\nSTEP 7: Model Comparison")
        self.comparison = pd.DataFrame(self.results).T
        
        if Config.TASK_TYPE == "regression":
            self.comparison = self.comparison.sort_values('cv_rmse')
            print(self.comparison[['train_r2', 'cv_rmse', 'cv_std']].to_string())
        else:
            self.comparison = self.comparison.sort_values('cv_acc', ascending=False)
            print(self.comparison[['train_acc', 'cv_acc', 'cv_std']].to_string())
        
        self.comparison.to_csv(f"{Config.RESULTS_FOLDER}comparison.csv")
    
    def select_best(self):
        print("\nSTEP 8: Best Model Selection")
        
        if Config.TASK_TYPE == "regression":
            self.best_name = self.comparison['cv_rmse'].idxmin()
        else:
            self.best_name = self.comparison['cv_acc'].idxmax()
        
        self.best_model = self.models[self.best_name]
        print(f"   Best: {self.best_name}")
        print(f"   Metrics: {self.results[self.best_name]}")
    
    def hyperparameter_tuning(self):
        print("\nSTEP 9: Hyperparameter Tuning")
        
        if not Config.PERFORM_TUNING:
            print("   Disabled (set PERFORM_TUNING=True to enable)")
            self.final_model = self.best_model
            return
        
        print(f"   Tuning {self.best_name} with {Config.TUNING_ITERATIONS} iterations...")
        print(f"   This may take several minutes...")
        
        param_grids = {
            'random_forest': {'n_estimators': [50, 100], 'max_depth': [10, 20]},
            'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.3]},
            'xgboost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.3]},
            'lasso': {'alpha': [0.1, 1, 10]},
            'linear_ridge': {'alpha': [0.1, 1, 10]}
        }
        
        if self.best_name not in param_grids:
            print(f"   No grid for {self.best_name}")
            self.final_model = self.best_model
            return
        
        try:
            search = RandomizedSearchCV(
                self.best_model, param_grids[self.best_name],
                n_iter=min(Config.TUNING_ITERATIONS, 10),
                cv=3,
                n_jobs=Config.N_JOBS,
                random_state=Config.RANDOM_STATE,
                verbose=2
            )
            search.fit(self.X_train, self.y_train)
            
            print(f"   Best params: {search.best_params_}")
            print(f"   Best score: {search.best_score_:.4f}")
            
            self.final_model = search.best_estimator_
            self.log['tuning'] = {'params': search.best_params_, 'score': search.best_score_}
        except Exception as e:
            print(f"   Tuning failed: {e}")
            print(f"   Using untuned model")
            self.final_model = self.best_model
    
    def evaluate_test(self):
        print("\nSTEP 10-11: Test Evaluation")
        
        pred = self.final_model.predict(self.X_test)
        
        if Config.TASK_TYPE == "regression":
            r2 = r2_score(self.y_test, pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred))
            mae = mean_absolute_error(self.y_test, pred)
            
            print(f"   R2 Score: {r2:.4f}")
            print(f"   RMSE:     {rmse:.4f}")
            print(f"   MAE:      {mae:.4f}")
            
            self.test_results = {'r2': r2, 'rmse': rmse, 'mae': mae}
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual'); plt.ylabel('Predicted')
            plt.title(f'{self.best_name} - Predictions vs Actual')
            plt.savefig(f"{Config.RESULTS_FOLDER}predictions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            acc = accuracy_score(self.y_test, pred)
            f1 = f1_score(self.y_test, pred, average='weighted')
            
            print(f"   Accuracy: {acc:.4f}")
            print(f"   F1 Score: {f1:.4f}")
            print(f"\n   Classification Report:")
            print(classification_report(self.y_test, pred))
            
            self.test_results = {'accuracy': acc, 'f1': f1}
            
            cm = confusion_matrix(self.y_test, pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted'); plt.ylabel('Actual')
            plt.title(f'{self.best_name} - Confusion Matrix')
            plt.savefig(f"{Config.RESULTS_FOLDER}confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.log['test_results'] = self.test_results
        print(f"   Saved plots to {Config.RESULTS_FOLDER}")
    
    def feature_importance(self):
        print("\nSTEP 12: Feature Importance")
        
        try:
            if hasattr(self.final_model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.final_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n   Top 10 Features:")
                print(imp.head(10).to_string(index=False))
                
                imp.to_csv(f"{Config.RESULTS_FOLDER}feature_importance.csv", index=False)
                
                plt.figure(figsize=(10, 8))
                top20 = imp.head(20)
                plt.barh(range(len(top20)), top20['importance'])
                plt.yticks(range(len(top20)), top20['feature'])
                plt.xlabel('Importance')
                plt.title(f'Top 20 Features - {self.best_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f"{Config.RESULTS_FOLDER}feature_importance.png", dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"   Warning: {e}")
    
    def error_analysis(self):
        print("\nSTEP 13: Error Analysis")
        
        pred = self.final_model.predict(self.X_test)
        
        if Config.TASK_TYPE == "regression":
            errors = self.y_test - pred
            abs_errors = np.abs(errors)
            
            print(f"   Mean Error:    {errors.mean():.4f}")
            print(f"   Std Error:     {errors.std():.4f}")
            print(f"   Max Error:     {abs_errors.max():.4f}")
            print(f"   95th percentile: {np.percentile(abs_errors, 95):.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title(f'Error Distribution - {self.best_name}')
            plt.axvline(0, color='red', linestyle='--', linewidth=2)
            plt.savefig(f"{Config.RESULTS_FOLDER}error_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log['error_analysis'] = {
                'mean_error': float(errors.mean()),
                'std_error': float(errors.std()),
                'max_error': float(abs_errors.max())
            }
    
    def save_model(self):
        print("\nSTEP 14: Saving Model")
        
        model_path = f"{Config.MODELS_FOLDER}{self.best_name}_final.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.final_model, f)
        print(f"   Saved: {model_path}")
        
        with open(f"{Config.RESULTS_FOLDER}experiment_log.json", 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"   Saved: experiment_log.json")
    
    def create_inference(self):
        print("\nSTEP 15: Creating Inference Interface")
        
        inference_code = f'''"""
Production Inference Interface
Generated: {datetime.now().isoformat()}
"""

import pickle
import pandas as pd

def load_model():
    with open("{Config.MODELS_FOLDER}{self.best_name}_final.pkl", "rb") as f:
        return pickle.load(f)

def predict(X):
    model = load_model()
    return model.predict(X)

def predict_from_csv(filepath):
    X = pd.read_csv(filepath)
    return predict(X)

if __name__ == "__main__":
    X_test = pd.read_csv("{Config.DATA_FOLDER}X_test.csv")
    predictions = predict(X_test)
    print(f"Predictions: {{predictions[:5]}}")
'''
        
        with open(f"{Config.MODELS_FOLDER}inference.py", 'w') as f:
            f.write(inference_code)
        print(f"   Created: inference.py")
    
    def generate_report(self):
        print("\nSTEP 16: Final Report")
        
        report = f"""
{'='*80}
MACHINE LEARNING PIPELINE - FINAL REPORT
{'='*80}

CONFIGURATION
-------------
Task Type:       {Config.TASK_TYPE}
Best Model:      {self.best_name}
CV Folds:        {Config.CV_FOLDS}
Random State:    {Config.RANDOM_STATE}

DATA
----
Training Samples:  {len(self.X_train):,}
Test Samples:      {len(self.X_test):,}
Features:          {self.X_train.shape[1]}

BASELINE PERFORMANCE
--------------------
Baseline Score:    {self.baseline_score:.4f}

MODEL COMPARISON (Top 3)
-----------------------
"""
        
        if Config.TASK_TYPE == "regression":
            report += self.comparison.head(3)[['train_r2', 'cv_rmse']].to_string()
            report += f"\n\nFINAL TEST PERFORMANCE\n----------------------\n"
            report += f"R2 Score:  {self.test_results['r2']:.4f}\n"
            report += f"RMSE:      {self.test_results['rmse']:.4f}\n"
            report += f"MAE:       {self.test_results['mae']:.4f}\n"
        else:
            report += self.comparison.head(3)[['train_acc', 'cv_acc']].to_string()
            report += f"\n\nFINAL TEST PERFORMANCE\n----------------------\n"
            report += f"Accuracy:  {self.test_results['accuracy']:.4f}\n"
            report += f"F1 Score:  {self.test_results['f1']:.4f}\n"
        
        report += f"""
OUTPUTS
-------
Models:     {Config.MODELS_FOLDER}
Results:    {Config.RESULTS_FOLDER}
Inference:  {Config.MODELS_FOLDER}inference.py

FILES GENERATED
---------------
- {self.best_name}_final.pkl
- comparison.csv
- experiment_log.json
- predictions.png
- feature_importance.png
- inference.py

{'='*80}
"""
        
        print(report)
        
        with open(f"{Config.RESULTS_FOLDER}final_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nFull report saved to: {Config.RESULTS_FOLDER}final_report.txt")

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run()