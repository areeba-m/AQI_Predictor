"""
Scikit-learn Model Training Pipeline for AQI Prediction
Includes Random Forest, Gradient Boosting, and Linear models with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SklearnModelTrainer:
    """Comprehensive scikit-learn model training with overfitting detection"""
    
    def __init__(self, target_variable: str = 'us_aqi', random_state: int = 42):
        self.target_variable = target_variable
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare data for training"""
        print("📊 Preparing data for training...")
        
        
        # Remove datetime and target columns for features
        feature_columns = [col for col in df.columns if col not in [self.target_variable, 'datetime']]
        
        X = df[feature_columns].values
        y = df[self.target_variable].values
        
        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        print(f"✅ Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=True
        )
        
        print(f"✅ Training set: {X_train.shape}")
        print(f"✅ Test set: {X_test.shape}")
        
        print(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features for linear models"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def detect_overfitting(self, model_name: str, train_score: float, test_score: float, cv_scores: Optional[np.ndarray] = None) -> bool:
        """Detect and warn about overfitting"""
        print(f"\n🔍 OVERFITTING ANALYSIS for {model_name}")
        print("-" * 40)
        
        overfitting_flags = []
        
        # Check R² scores
        if train_score > 0.98:
            overfitting_flags.append(f"⚠️  Extremely high training R²: {train_score:.4f}")
        
        gap = train_score - test_score
        if gap > 0.15:
            overfitting_flags.append(f"⚠️  Large train-test gap: {gap:.4f}")
        
        if test_score < 0.6 and train_score > 0.8:
            overfitting_flags.append(f"⚠️  Poor generalization: train={train_score:.3f}, test={test_score:.3f}")
        
        if cv_scores is not None:
            cv_std = np.std(cv_scores)
            if cv_std > 0.1:
                overfitting_flags.append(f"⚠️  High CV variance: {cv_std:.4f}")
        
        if len(overfitting_flags) > 0:
            print("🚨 OVERFITTING DETECTED:")
            for flag in overfitting_flags:
                print(f"   {flag}")
            print("💡 Consider: feature reduction, regularization, or more data")
        else:
            print("✅ No overfitting detected")
        
        return len(overfitting_flags) == 0
    
    def evaluate_model(self, model: Any, model_name: str, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray, use_cv: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        print(f"\n📊 TRAINING AND EVALUATING {model_name}")
        print(f"🚀 Training {model_name}...")
        print("-" * 50)
        
        # Fit model
        model.fit(X_train, y_train)
        print(f"✅ {model_name} training completed!")
        
        # Predictions
        print(f"🔮 Making predictions with {model_name}...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        print(f"📈 Calculating metrics for {model_name}...")
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = None
        if use_cv:
            print(f"🔄 Running cross-validation for {model_name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        print(f"📈 Training R²:    {train_r2:.4f}")
        print(f"📈 Training RMSE:  {train_rmse:.2f}")
        print(f"📈 Test R²:        {test_r2:.4f}")
        print(f"📈 Test RMSE:      {test_rmse:.2f}")
        print(f"📈 Test MAE:       {test_mae:.2f}")
        if cv_scores is not None:
            print(f"📈 CV R² mean:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Overfitting check
        is_healthy = self.detect_overfitting(model_name, train_r2, test_r2, cv_scores)
        
        result = {
            'model': model,
            'model_name': model_name,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean() if cv_scores is not None else None,
            'cv_std': cv_scores.std() if cv_scores is not None else None,
            'is_healthy': is_healthy,
            'predictions': y_test_pred,
            'y_test': y_test
        }
        
        print(f"{model_name} evaluation completed - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f}")
        
        return result
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with optimized hyperparameters"""
        print("🤖 Initializing models...")
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=self.random_state
            ),
            'Lasso Regression': Lasso(
                alpha=0.1,
                random_state=self.random_state
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=self.random_state
            )
        }
        
        print(f"✅ Initialized {len(models)} models")
        print(f"Initialized {len(models)} models")
        
        return models
    
    def train_all_models(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models and return results"""
        print("🚀 STARTING COMPREHENSIVE MODEL TRAINING")
        print("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(df, test_size)
        
        # Scale features for linear models
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Initialize models
        models = self.initialize_models()
        self.models = models
        
        # Train each model
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*60}")
            print(f"🤖 TRAINING MODEL: {name}")
            print(f"{'='*60}")
            
            try:
                # Use scaled features for linear models
                if name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                    result = self.evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test)
                else:
                    result = self.evaluate_model(model, name, X_train, X_test, y_train, y_test)
                
                results[name] = result
                print(f"✅ {name} training completed successfully!")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                
                continue
        
        self.results = results
        
        # Find best model
        self._select_best_model()
        
        # Print comparison
        self._print_model_comparison()
        
        # Save feature columns for later use
        self.feature_columns = feature_columns
        
        print(f"Training completed for {len(results)} models")
        
        return results
    
    def _select_best_model(self):
        """Select the best performing model"""
        if not self.results:
            return
        
        print(f"\n🏆 SELECTING BEST MODEL")
        print("="*40)
        
        # Filter healthy models first
        healthy_models = {name: result for name, result in self.results.items() if result['is_healthy']}
        
        if healthy_models:
            # Select best healthy model by test R²
            best_name = max(healthy_models.keys(), key=lambda x: healthy_models[x]['test_r2'])
            self.best_model_name = best_name
            self.best_model = healthy_models[best_name]['model']
            
            print(f"✅ Best healthy model: {best_name}")
            print(f"📈 Test R²: {healthy_models[best_name]['test_r2']:.4f}")
            print(f"📈 Test RMSE: {healthy_models[best_name]['test_rmse']:.2f}")
            
        else:
            # If no healthy models, select best by test R²
            best_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
            self.best_model_name = best_name
            self.best_model = self.results[best_name]['model']
            
            print(f"⚠️  No completely healthy models found")
            print(f"📊 Best available model: {best_name}")
            print(f"📈 Test R²: {self.results[best_name]['test_r2']:.4f}")
        
        print(f"Best model selected: {self.best_model_name}")
    
    def _print_model_comparison(self):
        """Print comprehensive model comparison"""
        if not self.results:
            return
        
        print(f"\n📊 COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = {}
        for name, result in self.results.items():
            comparison_data[name] = {
                'Test R²': result['test_r2'],
                'Test RMSE': result['test_rmse'],
                'Test MAE': result['test_mae'],
                'Train R²': result['train_r2'],
                'CV Mean': result['cv_mean'] if result['cv_mean'] else 0,
                'CV Std': result['cv_std'] if result['cv_std'] else 0,
                'Overfitting': '❌' if not result['is_healthy'] else '✅'
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        print(comparison_df.round(4))
        
        # Print recommendations
        print(f"\n💡 TRAINING RECOMMENDATIONS:")
        print("="*40)
        
        best_test_r2 = max(result['test_r2'] for result in self.results.values())
        
        if best_test_r2 < 0.7:
            print("📈 Model performance could be improved:")
            print("   • Collect more diverse training data")
            print("   • Engineer additional features")
            print("   • Try ensemble methods or deep learning")
        
        unhealthy_count = sum(1 for result in self.results.values() if not result['is_healthy'])
        if unhealthy_count > 0:
            print(f"⚠️  {unhealthy_count} models show overfitting concerns:")
            print("   • Consider reducing model complexity")
            print("   • Add more regularization")
            print("   • Collect more training data")
        
        best_rmse = min(result['test_rmse'] for result in self.results.values())
        if best_rmse > 20:
            print("📊 High prediction error detected:")
            print("   • Review feature selection process")
            print("   • Check for missing important variables")
            print("   • Consider non-linear transformations")
    
    def save_best_model(self, model_dir: str = "models") -> str:
        """Save the best model and metadata locally and to Hopsworks"""
        if self.best_model is None:
            print("❌ No best model to save")
            return ""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"best_sklearn_model_{self.best_model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': 'sklearn',
            'target_variable': self.target_variable,
            'test_r2': self.results[self.best_model_name]['test_r2'],
            'test_rmse': self.results[self.best_model_name]['test_rmse'],
            'test_mae': self.results[self.best_model_name]['test_mae'],
            'is_healthy': self.results[self.best_model_name]['is_healthy'],
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, "sklearn_model_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Best model saved locally:")
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Scaler: {scaler_path}")
        print(f"   📁 Metadata: {metadata_path}")
        
        # Upload to Hopsworks Model Registry
        self._upload_to_hopsworks(model_path, scaler_path, metadata)
        
        print(f"Best model saved: {self.best_model_name}")
        
        return model_path
    
    def _upload_to_hopsworks(self, model_path: str, scaler_path: str, metadata: dict):
        """Upload model to Hopsworks Model Registry"""
        try:
            # Import Hopsworks integration
            import sys
            import os
            # Add the pipelines directory to path
            pipelines_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipelines')
            sys.path.append(pipelines_dir)
            from pipelines.fetch_data import HopsworksIntegration
            
            # Initialize Hopsworks
            hops = HopsworksIntegration()
            if not hops.enabled:
                print("⚠️ Hopsworks not enabled, skipping model registry upload")
                return
            
            print("🚀 Uploading model to Hopsworks Model Registry...")
            
            # Create model directory for upload
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy model files to temp directory
                model_dir = os.path.join(temp_dir, "sklearn_model")
                os.makedirs(model_dir, exist_ok=True)
                
                # Copy files
                shutil.copy2(model_path, model_dir)
                shutil.copy2(scaler_path, model_dir)
                
                # Create model.py file for deployment
                model_py_content = f'''
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class SklearnAQIModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = {metadata['feature_columns']}
        self.model_name = "{metadata['model_name']}"
    
    def load(self, model_dir):
        """Load model and scaler"""
        import os
        self.model = joblib.load(os.path.join(model_dir, "{os.path.basename(model_path)}"))
        self.scaler = joblib.load(os.path.join(model_dir, "{os.path.basename(scaler_path)}"))
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded")
        
        # Ensure feature order
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)
'''
                
                model_py_path = os.path.join(model_dir, "model.py")
                with open(model_py_path, 'w') as f:
                    f.write(model_py_content)
                
                # Upload to Hopsworks
                success = hops.save_model(model_dir, f"aqi_sklearn_{self.best_model_name.lower().replace(' ', '_')}", "sklearn")
                
                if success:
                    print("✅ Model uploaded to Hopsworks Model Registry")
                else:
                    print("⚠️ Model upload to Hopsworks failed")
                    
        except Exception as e:
            print(f"⚠️ Hopsworks model upload failed: {e}")
            print("   Model saved locally only")

def main():
    """Test the sklearn model training pipeline"""
    print("🚀 Testing Scikit-learn Model Training Pipeline")
    print("="*60)
    
    # Create sample data for testing
    from features.feature_engineering import FeatureEngineer, AdvancedFeatureSelector
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='H')
    n_samples = len(dates)
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'temperature': np.random.normal(20, 5, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        'wind_speed': np.random.exponential(5, n_samples),
        'pressure_msl': np.random.normal(1013, 10, n_samples),
        'nitrogen_dioxide': np.random.exponential(30, n_samples),
        'ozone': np.random.exponential(80, n_samples),
        'carbon_monoxide': np.random.exponential(0.5, n_samples),
        'sulphur_dioxide': np.random.exponential(10, n_samples),
        'us_aqi': np.random.normal(50, 20, n_samples),
        'dew_point': np.random.normal(15, 5, n_samples),
        'apparent_temperature': np.random.normal(22, 5, n_samples),
        'precipitation': np.random.exponential(0.5, n_samples),
        'rain': np.random.exponential(0.3, n_samples),
        'snowfall': np.zeros(n_samples),
        'weather_code': np.random.randint(0, 100, n_samples),
        'surface_pressure': np.random.normal(1010, 8, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
        'visibility': np.random.normal(10000, 2000, n_samples),
        'wind_direction': np.random.uniform(0, 360, n_samples),
        'wind_gusts': np.random.exponential(8, n_samples),
        'uv_index': np.random.uniform(0, 10, n_samples)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    
    # Feature engineering
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features(sample_data)
    engineered_data = engineer.handle_missing_values(engineered_data)
    
    # Feature selection
    selector = AdvancedFeatureSelector()
    selected_data, selected_features = selector.select_features(engineered_data, max_features=20)
    
    print(f"📊 Prepared data shape: {selected_data.shape}")
    
    # Train models
    trainer = SklearnModelTrainer()
    results = trainer.train_all_models(selected_data, test_size=0.2)
    
    # Save best model
    model_path = trainer.save_best_model()
    
    print(f"\n✅ Scikit-learn model training pipeline test completed!")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"📁 Saved to: {model_path}")

if __name__ == "__main__":
    main()
