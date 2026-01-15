import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats

class Config:
    INPUT_FILE = "../ml/moretoo/Ecommerce/train/cleaned_Customers.csv"
    OUTPUT_FOLDER = "../ml/ready_data/"
    ARTIFACTS_FOLDER = "../ml/artifacts/"
    
    TARGET_COLUMN = None
    TASK_TYPE = "auto"
    
    DROP_HIGH_CARDINALITY = False
    CARDINALITY_THRESHOLD = 100
    DROP_HIGH_MISSING = True
    MISSING_THRESHOLD = 0.7
    DROP_LOW_VARIANCE = True
    VARIANCE_THRESHOLD = 0.01
    
    SCALING_METHOD = "standard"
    APPLY_LOG_TRANSFORM = True
    SKEWNESS_THRESHOLD = 1.0
    
    HANDLE_OUTLIERS = True
    OUTLIER_METHOD = "iqr"
    IQR_MULTIPLIER = 1.5
    
    ENCODING_METHOD = "auto"
    ONEHOT_MAX_CATEGORIES = 20
    
    CREATE_INTERACTIONS = True
    MAX_INTERACTIONS = 20
    HANDLE_MULTICOLLINEARITY = True
    CORRELATION_THRESHOLD = 0.95
    APPLY_FEATURE_SELECTION = True
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    SAMPLE_DATA = False
    SAMPLE_SIZE = 50000
    EXTRACT_DATETIME_FEATURES = True

class FeatureEngineeringPipeline:
    
    def __init__(self):
        self.metadata = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'transformations': [],
            'artifacts': []
        }
        self.scalers = {}
        self.encoders = {}
        
    def run(self):
        print("="*80)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*80)
        
        self._setup_dirs()
        self._load_data()
        self._define_target()
        self._analyze_features()
        self._drop_ineligible()
        self._extract_datetime()
        self._handle_missing()
        self._handle_outliers()
        self._transform_numeric()
        self._encode_categorical()
        self._create_derived()
        self._create_interactions()
        self._handle_multicollinearity()
        self._scale_features()
        self._select_features()
        self._split_data()
        self._save_outputs()
        self._generate_report()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
    
    def _setup_dirs(self):
        Path(Config.OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(Config.ARTIFACTS_FOLDER).mkdir(parents=True, exist_ok=True)
        print("Directories created")
    
    def _load_data(self):
        print("\nSTEP 1: Loading Data")
        self.df = pd.read_csv(Config.INPUT_FILE)
        print(f"   Loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} cols")
        
        if Config.SAMPLE_DATA and len(self.df) > Config.SAMPLE_SIZE:
            self.df = self.df.sample(Config.SAMPLE_SIZE, random_state=Config.RANDOM_STATE)
            print(f"   Sampled: {len(self.df):,} rows")
    
    def _define_target(self):
        print("\nSTEP 2: Target Variable")
        if Config.TARGET_COLUMN is None:
            Config.TARGET_COLUMN = self.df.columns[-1]
            print(f"   Auto-detected: '{Config.TARGET_COLUMN}'")
        
        if Config.TARGET_COLUMN not in self.df.columns:
            raise ValueError(f"Target column '{Config.TARGET_COLUMN}' not found")
        
        self.y = self.df[Config.TARGET_COLUMN].copy()
        self.X = self.df.drop(columns=[Config.TARGET_COLUMN])
        
        if Config.TASK_TYPE == "auto":
            if pd.api.types.is_numeric_dtype(self.y) and self.y.nunique() > 20:
                Config.TASK_TYPE = "regression"
            else:
                Config.TASK_TYPE = "classification"
        
        print(f"   Target: '{Config.TARGET_COLUMN}' ({Config.TASK_TYPE})")
        print(f"   Non-null: {self.y.notna().sum():,} / {len(self.y):,}")
    
    def _analyze_features(self):
        print("\nSTEP 3: Feature Analysis")
        self.numeric_cols = self.X.select_dtypes(include=['int64','float64']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = self.X.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in self.categorical_cols[:]:
            try:
                if self.X[col].str.match(r'\d{4}-\d{2}-\d{2}').sum() > len(self.X) * 0.5:
                    self.datetime_cols.append(col)
                    self.categorical_cols.remove(col)
            except:
                pass
        
        print(f"   Numeric: {len(self.numeric_cols)}")
        print(f"   Categorical: {len(self.categorical_cols)}")
        print(f"   Datetime: {len(self.datetime_cols)}")
    
    def _drop_ineligible(self):
        print("\nSTEP 4: Dropping Ineligible Features")
        cols_to_drop = []
        
        if Config.DROP_HIGH_MISSING:
            for col in self.X.columns:
                if self.X[col].isnull().sum() / len(self.X) > Config.MISSING_THRESHOLD:
                    cols_to_drop.append(col)
        
        if Config.DROP_HIGH_CARDINALITY:
            for col in self.categorical_cols:
                if col not in cols_to_drop and self.X[col].nunique() > Config.CARDINALITY_THRESHOLD:
                    cols_to_drop.append(col)
        
        if Config.DROP_LOW_VARIANCE:
            for col in self.numeric_cols:
                if col not in cols_to_drop:
                    var = self.X[col].var()
                    if not np.isnan(var) and var < Config.VARIANCE_THRESHOLD:
                        cols_to_drop.append(col)
        
        for col in self.X.columns:
            if col not in cols_to_drop:
                if any(kw in col.lower() for kw in ['id','_id','key','uuid']):
                    if self.X[col].nunique() > len(self.X) * 0.8:
                        cols_to_drop.append(col)
        
        cols_to_drop = list(set(cols_to_drop))
        if cols_to_drop:
            self.X = self.X.drop(columns=cols_to_drop)
            self.numeric_cols = [c for c in self.numeric_cols if c not in cols_to_drop]
            self.categorical_cols = [c for c in self.categorical_cols if c not in cols_to_drop]
            print(f"   Dropped: {len(cols_to_drop)} features")
        else:
            print("   All features eligible")
    
    def _extract_datetime(self):
        print("\nSTEP 5: Datetime Extraction")
        if not self.datetime_cols:
            print("   No datetime columns")
            return
        
        new_features = []
        for col in self.datetime_cols:
            dt = pd.to_datetime(self.X[col], errors='coerce')
            self.X[f'{col}_year'] = dt.dt.year
            self.X[f'{col}_month'] = dt.dt.month
            self.X[f'{col}_day'] = dt.dt.day
            self.X[f'{col}_dayofweek'] = dt.dt.dayofweek
            self.X[f'{col}_quarter'] = dt.dt.quarter
            new_features.extend([f'{col}_year',f'{col}_month',f'{col}_day',
                                f'{col}_dayofweek',f'{col}_quarter'])
            self.X = self.X.drop(columns=[col])
        
        self.numeric_cols.extend(new_features)
        print(f"   Created: {len(new_features)} features from {len(self.datetime_cols)} columns")
    
    def _handle_missing(self):
        print("\nSTEP 6: Missing Values")
        missing_before = self.X.isnull().sum().sum()
        
        for col in self.numeric_cols:
            if self.X[col].isnull().any():
                self.X[col].fillna(self.X[col].median(), inplace=True)
        
        for col in self.categorical_cols:
            if self.X[col].isnull().any():
                mode = self.X[col].mode()[0] if not self.X[col].mode().empty else 'Unknown'
                self.X[col].fillna(mode, inplace=True)
        
        print(f"   Imputed: {missing_before:,} values")
    
    def _handle_outliers(self):
        print("\nSTEP 7: Outlier Handling")
        if not Config.HANDLE_OUTLIERS:
            print("   Disabled")
            return
        
        outliers_count = 0
        for col in self.numeric_cols:
            if self.X[col].notna().sum() < 10:
                continue
                
            if Config.OUTLIER_METHOD == "iqr":
                Q1 = self.X[col].quantile(0.25)
                Q3 = self.X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - Config.IQR_MULTIPLIER * IQR
                upper = Q3 + Config.IQR_MULTIPLIER * IQR
                outliers = ((self.X[col] < lower) | (self.X[col] > upper)).sum()
                if outliers > 0:
                    self.X[col] = self.X[col].clip(lower, upper)
                    outliers_count += outliers
        
        print(f"   Handled: {outliers_count:,} outliers")
    
    def _transform_numeric(self):
        print("\nSTEP 8: Numeric Transformation")
        if not Config.APPLY_LOG_TRANSFORM:
            print("   Disabled")
            return
        
        transformed = []
        for col in self.numeric_cols[:]:
            if self.X[col].notna().sum() < 10:
                continue
            skew = self.X[col].skew()
            if abs(skew) > Config.SKEWNESS_THRESHOLD and self.X[col].min() > 0:
                self.X[f'{col}_log'] = np.log1p(self.X[col])
                transformed.append(f'{col}_log')
        
        self.numeric_cols.extend(transformed)
        print(f"   Created: {len(transformed)} log-transformed features")
    
    def _encode_categorical(self):
        print("\nSTEP 9: Categorical Encoding")
        if not self.categorical_cols:
            print("   No categorical features")
            return
        
        encoded = []
        for col in self.categorical_cols:
            n_cat = self.X[col].nunique()
            
            if Config.ENCODING_METHOD == "auto":
                if n_cat <= Config.ONEHOT_MAX_CATEGORIES:
                    dummies = pd.get_dummies(self.X[col], prefix=col, drop_first=True)
                    self.X = pd.concat([self.X, dummies], axis=1)
                    encoded.extend(dummies.columns.tolist())
                else:
                    le = LabelEncoder()
                    self.X[f'{col}_enc'] = le.fit_transform(self.X[col].astype(str))
                    self.encoders[col] = le
                    encoded.append(f'{col}_enc')
                    self._save_artifact(le, f'encoder_{col}')
            elif Config.ENCODING_METHOD == "onehot":
                dummies = pd.get_dummies(self.X[col], prefix=col, drop_first=True)
                self.X = pd.concat([self.X, dummies], axis=1)
                encoded.extend(dummies.columns.tolist())
            else:
                le = LabelEncoder()
                self.X[f'{col}_enc'] = le.fit_transform(self.X[col].astype(str))
                self.encoders[col] = le
                encoded.append(f'{col}_enc')
                self._save_artifact(le, f'encoder_{col}')
        
        self.X = self.X.drop(columns=self.categorical_cols)
        self.numeric_cols.extend(encoded)
        print(f"   Encoded: {len(self.categorical_cols)} -> {len(encoded)} features")
        self.categorical_cols = []
    
    def _create_derived(self):
        print("\nSTEP 10: Derived Features")
        derived = []
        
        if len(self.numeric_cols) >= 2:
            variances = self.X[self.numeric_cols].var()
            valid_cols = variances[variances > 0].index.tolist()
            top_cols = self.X[valid_cols].var().nlargest(min(3, len(valid_cols))).index.tolist()
            
            for i in range(len(top_cols)-1):
                col1, col2 = top_cols[i], top_cols[i+1]
                if (self.X[col2] != 0).all():
                    self.X[f'{col1}_div_{col2}'] = self.X[col1] / (self.X[col2] + 1e-5)
                    derived.append(f'{col1}_div_{col2}')
                self.X[f'{col1}_mul_{col2}'] = self.X[col1] * self.X[col2]
                derived.append(f'{col1}_mul_{col2}')
        
        self.numeric_cols.extend(derived)
        print(f"   Created: {len(derived)} derived features")
    
    def _create_interactions(self):
        print("\nSTEP 11: Feature Interactions")
        if not Config.CREATE_INTERACTIONS or len(self.numeric_cols) < 2:
            print("   Skipped")
            return
        
        variances = self.X[self.numeric_cols].var()
        valid_cols = variances[variances > 0].index.tolist()
        top_feats = self.X[valid_cols].var().nlargest(min(5, len(valid_cols))).index.tolist()
        
        interactions = []
        count = 0
        
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                if count >= Config.MAX_INTERACTIONS:
                    break
                col1, col2 = top_feats[i], top_feats[j]
                self.X[f'{col1}_x_{col2}'] = self.X[col1] * self.X[col2]
                interactions.append(f'{col1}_x_{col2}')
                count += 1
        
        self.numeric_cols.extend(interactions)
        print(f"   Created: {len(interactions)} interactions")
    
    def _handle_multicollinearity(self):
        print("\nSTEP 12: Multicollinearity Control")
        if not Config.HANDLE_MULTICOLLINEARITY or len(self.numeric_cols) < 2:
            print("   Skipped")
            return
        
        corr_matrix = self.X[self.numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > Config.CORRELATION_THRESHOLD)]
        
        if to_drop:
            self.X = self.X.drop(columns=to_drop)
            self.numeric_cols = [c for c in self.numeric_cols if c not in to_drop]
            print(f"   Dropped: {len(to_drop)} correlated features")
        else:
            print("   No high correlation found")
    
    def _scale_features(self):
        print("\nSTEP 13: Feature Scaling")
        
        if not self.numeric_cols:
            print("   No features to scale")
            return
        
        if Config.SCALING_METHOD == "standard":
            scaler = StandardScaler()
        elif Config.SCALING_METHOD == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        self.X[self.numeric_cols] = scaler.fit_transform(self.X[self.numeric_cols])
        self.scalers['main_scaler'] = scaler
        self._save_artifact(scaler, 'scaler_main')
        print(f"   Scaled: {len(self.numeric_cols)} features ({Config.SCALING_METHOD})")
    
    def _select_features(self):
        print("\nSTEP 14: Feature Selection")
        if not Config.APPLY_FEATURE_SELECTION or len(self.X.columns) < 2:
            print("   Skipped")
            return
        
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(self.X)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        dropped = len(self.X.columns) - len(selected_features)
        self.X = self.X[selected_features]
        self.numeric_cols = [c for c in self.numeric_cols if c in selected_features]
        
        print(f"   Selected: {len(selected_features)} features (dropped {dropped})")
    
    def _split_data(self):
        print("\nSTEP 15: Train-Test Split")
        
        stratify = self.y if Config.TASK_TYPE == "classification" and self.y.nunique() < 50 else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=stratify
        )
        
        print(f"   Train: {len(self.X_train):,} samples")
        print(f"   Test:  {len(self.X_test):,} samples")
    
    def _save_outputs(self):
        print("\nSTEP 16: Saving Outputs")
        
        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        test_df = pd.concat([self.X_test, self.y_test], axis=1)
        
        train_df.to_csv(f"{Config.OUTPUT_FOLDER}train.csv", index=False)
        test_df.to_csv(f"{Config.OUTPUT_FOLDER}test.csv", index=False)
        self.X_train.to_csv(f"{Config.OUTPUT_FOLDER}X_train.csv", index=False)
        self.X_test.to_csv(f"{Config.OUTPUT_FOLDER}X_test.csv", index=False)
        self.y_train.to_csv(f"{Config.OUTPUT_FOLDER}y_train.csv", index=False)
        self.y_test.to_csv(f"{Config.OUTPUT_FOLDER}y_test.csv", index=False)
        
        print(f"   Saved: 6 CSV files to {Config.OUTPUT_FOLDER}")
        print(f"   Feature list: {self.X_train.columns.tolist()}")
        
        self.metadata['final_features'] = self.X_train.columns.tolist()
        self.metadata['n_features'] = len(self.X_train.columns)
        self.metadata['n_samples_train'] = len(self.X_train)
        self.metadata['n_samples_test'] = len(self.X_test)
        self.metadata['task_type'] = Config.TASK_TYPE
        self.metadata['target_column'] = Config.TARGET_COLUMN
        
        if Config.TASK_TYPE == "classification":
            self.metadata['n_classes'] = int(self.y.nunique())
            self.metadata['target_dtype'] = str(self.y.dtype)
            self.metadata['class_distribution'] = self.y.value_counts().to_dict()
        else:
            self.metadata['target_dtype'] = str(self.y.dtype)
            self.metadata['target_stats'] = {
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'mean': float(self.y.mean()),
                'median': float(self.y.median())
            }
        
        with open(f"{Config.ARTIFACTS_FOLDER}metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"   Saved: metadata.json")
    
    def _save_artifact(self, obj, name):
        with open(f"{Config.ARTIFACTS_FOLDER}{name}.pkl", 'wb') as f:
            pickle.dump(obj, f)
    
    def _generate_report(self):
        print("\nSTEP 17: Final Report")
        print("="*80)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*80)
        print(f"Original Features: {len(self.df.columns)-1}")
        print(f"Final Features:    {len(self.X_train.columns)}")
        print(f"Target Variable:   {Config.TARGET_COLUMN}")
        print(f"Task Type:         {Config.TASK_TYPE}")
        print(f"Train Samples:     {len(self.X_train):,}")
        print(f"Test Samples:      {len(self.X_test):,}")
        print(f"\nOutputs:")
        print(f"   Data:      {Config.OUTPUT_FOLDER}")
        print(f"   Artifacts: {Config.ARTIFACTS_FOLDER}")
        print("="*80)

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run()