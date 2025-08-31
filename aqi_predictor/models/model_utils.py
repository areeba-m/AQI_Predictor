"""
Model Utilities for AQI Prediction System
Includes model loading, prediction, evaluation, and Hopsworks integration
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import tempfile
import pickle

import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Hopsworks imports with error handling
try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("⚠️ Hopsworks not available. Models will be loaded from local files only.")

# Configure logging

class ModelManager:
    """Manages model loading, prediction, and evaluation"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.sklearn_model = None
        self.dl_model = None
        self.sklearn_metadata = None
        self.dl_metadata = None
        self.sklearn_scaler = None
        self.dl_scaler_X = None
        self.dl_scaler_y = None
        self.feature_columns = []
        
        # Hopsworks connection
        self.project = None
        self.mr = None  # Model Registry
        self.hopsworks_enabled = False
        self._connect_hopsworks()
    
    def _connect_hopsworks(self):
        """Connect to Hopsworks for model registry access"""
        if not HOPSWORKS_AVAILABLE:
            print("⚠️ Hopsworks not available - using local model files")
            return
        
        try:
            # Get credentials from environment variables
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project_name = os.getenv('HOPSWORKS_PROJECT_NAME')
            
            if not api_key or not project_name:
                print("⚠️ Hopsworks credentials not found - using local model files")
                print("   Add HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME to environment variables")
                return
            
            print(f"🔗 Connecting to Hopsworks project: {project_name}")
            
            self.project = hopsworks.login(
                api_key_value=api_key,
                project=project_name
            )
            
            if self.project:
                self.mr = self.project.get_model_registry()
                self.hopsworks_enabled = True
                print(f"✅ Connected to Hopsworks model registry")
            else:
                print("❌ Failed to connect to Hopsworks")
                
        except Exception as e:
            print(f"⚠️ Hopsworks connection failed: {e}")
            print("   Will try to load models from local files")
    
    def _load_model_from_registry(self, model_name: str, version: int = None) -> Optional[Dict[str, Any]]:
        """Load model from Hopsworks model registry"""
        if not self.hopsworks_enabled:
            return None
        
        try:
            print(f"📥 Fetching latest version of model '{model_name}' from model registry...")
            
            # Get all versions of the model to find the latest
            try:
                # Get all versions and find the highest version number
                model_versions = self.mr.get_models(model_name)
                if not model_versions:
                    print(f"❌ No versions found for model '{model_name}'")
                    return None
                
                # Find the latest (highest) version number
                latest_version = max(m.version for m in model_versions)
                print(f"📊 Latest version found: {latest_version}")
                
                # If version is explicitly requested, use it, otherwise use latest
                if version and version != latest_version:
                    print(f"🔄 Requesting specific version {version} instead of latest {latest_version}")
                    model = self.mr.get_model(model_name, version=version)
                else:
                    # Get the latest version model
                    model = self.mr.get_model(model_name, version=latest_version)
                
            except Exception as e:
                print(f"⚠️ Could not determine latest version: {e}")
                # Fallback - try to get without version (should return latest)
                if version:
                    model = self.mr.get_model(model_name, version=version)
                else:
                    model = self.mr.get_model(model_name)
            
            if not model:
                print(f"❌ Model '{model_name}' not found in registry")
                return None
            
            print(f"✅ Found model: {model_name} (version {model.version})")
            
            # Download model to temp directory
            print(f"📁 Downloading model to temp directory...")
            model_dir = model.download()
            print(f"✅ Model downloaded to: {model_dir}")
            
            return {
                'model': model,
                'model_dir': model_dir,
                'version': model.version
            }
            
        except Exception as e:
            print(f"❌ Error loading model from registry: {e}")
            return None
        
    def load_sklearn_model(self) -> bool:
        """Load the best sklearn model and its components"""
        try:
            print("📥 Loading scikit-learn model...")
            
            # First try to load from Hopsworks model registry
            model_info = self._load_model_from_registry("aqi_sklearn_model")
            
            if model_info:
                # Load from Hopsworks
                model_dir = model_info['model_dir']
                
                # Load metadata
                metadata_path = os.path.join(model_dir, "sklearn_model_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.sklearn_metadata = json.load(f)
                else:
                    # Fallback metadata from model registry
                    self.sklearn_metadata = {
                        'model_name': 'Random Forest (from registry)',
                        'test_r2': model_info['model'].training_metrics.get('r2_score', 0),
                        'test_rmse': model_info['model'].training_metrics.get('rmse', 0),
                        'feature_columns': []
                    }
                
                # Look for model file
                model_files = [f for f in os.listdir(model_dir) 
                             if f.endswith('.joblib') or f.endswith('.pkl')]
                
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    self.sklearn_model = joblib.load(model_path)
                    
                    # Load scaler if available
                    scaler_files = [f for f in os.listdir(model_dir) 
                                  if 'scaler' in f.lower() and f.endswith('.joblib')]
                    if scaler_files:
                        scaler_path = os.path.join(model_dir, scaler_files[0])
                        self.sklearn_scaler = joblib.load(scaler_path)
                    
                    self.feature_columns = self.sklearn_metadata.get('feature_columns', [])
                    
                    print(f"✅ Sklearn model loaded from Hopsworks: {self.sklearn_metadata['model_name']}")
                    print(f"   📈 Test R²: {self.sklearn_metadata.get('test_r2', 0):.4f}")
                    return True
            
            # Fallback to local files
            print("⚠️ Hopsworks model not available, trying local files...")
            return self._load_sklearn_model_local()
            
        except Exception as e:
            print(f"❌ Error loading sklearn model from Hopsworks: {e}")
            print("⚠️ Falling back to local files...")
            return self._load_sklearn_model_local()
    
    def _load_sklearn_model_local(self) -> bool:
        """Load sklearn model from local files (fallback)"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                print(f"❌ Model directory '{self.model_dir}' not found")
                return False
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "sklearn_model_metadata.json")
            if not os.path.exists(metadata_path):
                print("❌ Sklearn model metadata not found locally")
                return False
            
            with open(metadata_path, 'r') as f:
                self.sklearn_metadata = json.load(f)
            
            # Load model
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("best_sklearn_model_") and f.endswith(".joblib")]
            if not model_files:
                print("❌ Sklearn model file not found")
                return False
            
            model_path = os.path.join(self.model_dir, model_files[0])
            self.sklearn_model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
            if os.path.exists(scaler_path):
                self.sklearn_scaler = joblib.load(scaler_path)
            
            self.feature_columns = self.sklearn_metadata.get('feature_columns', [])
            
            print(f"✅ Sklearn model loaded: {self.sklearn_metadata['model_name']}")
            print(f"   📈 Test R²: {self.sklearn_metadata['test_r2']:.4f}")
            print(f"   📈 Test RMSE: {self.sklearn_metadata['test_rmse']:.2f}")
            
            print(f"Sklearn model loaded: {self.sklearn_metadata['model_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading sklearn model: {e}")
            
            return False
    
    def load_dl_model(self) -> bool:
        """Load the best deep learning model and its components"""
        try:
            print("📥 Loading deep learning model...")
            
            # First try to load from Hopsworks model registry
            model_info = self._load_model_from_registry("aqi_dl_model")
            
            if model_info:
                # Load from Hopsworks
                model_dir = model_info['model_dir']
                
                # Load metadata
                metadata_path = os.path.join(model_dir, "dl_model_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.dl_metadata = json.load(f)
                else:
                    # Fallback metadata
                    self.dl_metadata = {
                        'model_name': 'Deep Learning (from registry)',
                        'test_r2': model_info['model'].training_metrics.get('r2_score', 0),
                        'test_rmse': model_info['model'].training_metrics.get('rmse', 0),
                        'feature_columns': []
                    }
                
                # Look for model file
                model_files = [f for f in os.listdir(model_dir) 
                             if f.endswith('.h5') or f.endswith('.keras')]
                
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    
                    # Load TensorFlow model
                    try:
                        self.dl_model = tf.keras.models.load_model(
                            model_path,
                            custom_objects={
                                'mse': tf.keras.metrics.MeanSquaredError(),
                                'mae': tf.keras.metrics.MeanAbsoluteError()
                            }
                        )
                    except Exception as e:
                        print(f"⚠️ Error with custom objects, trying compile=False: {e}")
                        self.dl_model = tf.keras.models.load_model(model_path, compile=False)
                        self.dl_model.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                        )
                    
                    # Load scalers if available
                    scaler_files = [f for f in os.listdir(model_dir) 
                                  if 'scaler' in f.lower() and f.endswith('.joblib')]
                    for scaler_file in scaler_files:
                        if 'scaler_X' in scaler_file or 'X_scaler' in scaler_file:
                            self.dl_scaler_X = joblib.load(os.path.join(model_dir, scaler_file))
                        elif 'scaler_y' in scaler_file or 'y_scaler' in scaler_file:
                            self.dl_scaler_y = joblib.load(os.path.join(model_dir, scaler_file))
                    
                    if not self.feature_columns:
                        self.feature_columns = self.dl_metadata.get('feature_columns', [])
                    
                    print(f"✅ Deep learning model loaded from Hopsworks: {self.dl_metadata['model_name']}")
                    print(f"   📈 Test R²: {self.dl_metadata.get('test_r2', 0):.4f}")
                    return True
            
            # Fallback to local files
            print("⚠️ Hopsworks model not available, trying local files...")
            return self._load_dl_model_local()
            
        except Exception as e:
            print(f"❌ Error loading deep learning model from Hopsworks: {e}")
            print("⚠️ Falling back to local files...")
            return self._load_dl_model_local()
    
    def _load_dl_model_local(self) -> bool:
        """Load deep learning model from local files (fallback)"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                print(f"❌ Model directory '{self.model_dir}' not found")
                return False
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "dl_model_metadata.json")
            if not os.path.exists(metadata_path):
                print("❌ Deep learning model metadata not found")
                return False
            
            with open(metadata_path, 'r') as f:
                self.dl_metadata = json.load(f)
            
            # Load model
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("best_dl_model_") and f.endswith(".h5")]
            if not model_files:
                print("❌ Deep learning model file not found")
                return False
            
            model_path = os.path.join(self.model_dir, model_files[0])
            
            # Try to load with custom objects to handle metric issues
            try:
                self.dl_model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'mse': tf.keras.metrics.MeanSquaredError(),
                        'mae': tf.keras.metrics.MeanAbsoluteError()
                    }
                )
            except Exception as e:
                print(f"⚠️ Error with custom objects, trying compile=False: {e}")
                # Try loading without compiling
                self.dl_model = tf.keras.models.load_model(model_path, compile=False)
                # Recompile manually
                self.dl_model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
            
            # Load scalers
            scaler_X_path = os.path.join(self.model_dir, "dl_scaler_X.joblib")
            scaler_y_path = os.path.join(self.model_dir, "dl_scaler_y.joblib")
            
            if os.path.exists(scaler_X_path):
                self.dl_scaler_X = joblib.load(scaler_X_path)
            if os.path.exists(scaler_y_path):
                self.dl_scaler_y = joblib.load(scaler_y_path)
            
            if not self.feature_columns:
                self.feature_columns = self.dl_metadata.get('feature_columns', [])
            
            print(f"✅ Deep learning model loaded: {self.dl_metadata['model_name']}")
            print(f"   📈 Test R²: {self.dl_metadata['test_r2']:.4f}")
            print(f"   📈 Test RMSE: {self.dl_metadata['test_rmse']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading deep learning model: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models"""
        print("📥 Loading all available models...")
        
        results = {
            'sklearn': self.load_sklearn_model(),
            'deep_learning': self.load_dl_model()
        }
        
        loaded_models = [name for name, success in results.items() if success]
        print(f"✅ Loaded {len(loaded_models)} model types: {loaded_models}")
        
        return results
    
    def predict_sklearn(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using sklearn model"""
        if self.sklearn_model is None:
            raise ValueError("Sklearn model not loaded")
        
        # Scale features if scaler is available
        if self.sklearn_scaler is not None:
            # Check if model needs scaled features
            model_name = self.sklearn_metadata.get('model_name', '').lower()
            if any(keyword in model_name for keyword in ['ridge', 'lasso', 'elastic', 'svr']):
                features = self.sklearn_scaler.transform(features)
        
        predictions = self.sklearn_model.predict(features)
        return predictions
    
    def predict_dl(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using deep learning model"""
        if self.dl_model is None:
            raise ValueError("Deep learning model not loaded")
        
        # Scale features
        if self.dl_scaler_X is not None:
            features_scaled = self.dl_scaler_X.transform(features)
        else:
            features_scaled = features
        
        # Check if model expects sequences (LSTM)
        model_name = self.dl_metadata.get('model_name', '').lower()
        if 'lstm' in model_name:
            # For LSTM, we need to create sequences
            # For simplicity, we'll use the last 24 features as sequence
            if len(features_scaled) >= 24:
                sequence = features_scaled[-24:].reshape(1, 24, -1)
                predictions_scaled = self.dl_model.predict(sequence, verbose=0)
            else:
                # Not enough data for sequence, use available data
                sequence = np.zeros((1, 24, features_scaled.shape[1]))
                sequence[0, -len(features_scaled):, :] = features_scaled
                predictions_scaled = self.dl_model.predict(sequence, verbose=0)
        else:
            # Feedforward model
            predictions_scaled = self.dl_model.predict(features_scaled, verbose=0)
        
        # Inverse transform predictions
        if self.dl_scaler_y is not None:
            predictions = self.dl_scaler_y.inverse_transform(predictions_scaled).ravel()
        else:
            predictions = predictions_scaled.ravel()
        
        return predictions
    
    def predict_ensemble(self, features: np.ndarray, weights: Optional[List[float]] = None) -> np.ndarray:
        """Make ensemble predictions using both models"""
        predictions = []
        model_names = []
        
        # Get sklearn predictions
        if self.sklearn_model is not None:
            sklearn_pred = self.predict_sklearn(features)
            predictions.append(sklearn_pred)
            model_names.append('sklearn')
        
        # Get deep learning predictions
        if self.dl_model is not None:
            dl_pred = self.predict_dl(features)
            predictions.append(dl_pred)
            model_names.append('deep_learning')
        
        if not predictions:
            raise ValueError("No models loaded for ensemble prediction")
        
        # Default weights
        if weights is None:
            weights = [1.0] * len(predictions)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        print(f"🔮 Ensemble prediction from {len(predictions)} models: {model_names}")
        
        return ensemble_pred
    
    def predict_from_dataframe(self, df: pd.DataFrame, model_type: str = 'best') -> np.ndarray:
        """Make predictions from a DataFrame with engineered features"""
        # Ensure we have the required feature columns
        if not self.feature_columns:
            raise ValueError("Feature columns not defined. Load a model first.")
        
        # Extract features
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            # Fill missing features with zeros or median
            for col in missing_features:
                df[col] = 0.0
        
        features = df[self.feature_columns].values
        
        if model_type == 'sklearn':
            return self.predict_sklearn(features)
        elif model_type == 'deep_learning':
            return self.predict_dl(features)
        elif model_type == 'ensemble':
            return self.predict_ensemble(features)
        elif model_type == 'best':
            # Choose best model based on metadata
            sklearn_r2 = self.sklearn_metadata.get('test_r2', 0) if self.sklearn_metadata else 0
            dl_r2 = self.dl_metadata.get('test_r2', 0) if self.dl_metadata else 0
            
            if sklearn_r2 > dl_r2 and self.sklearn_model is not None:
                print(f"🏆 Using sklearn model (R²: {sklearn_r2:.4f})")
                return self.predict_sklearn(features)
            elif self.dl_model is not None:
                print(f"🏆 Using deep learning model (R²: {dl_r2:.4f})")
                return self.predict_dl(features)
            else:
                raise ValueError("No models available for prediction")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class AQIPredictor:
    """High-level AQI prediction interface"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_manager = ModelManager(model_dir)
        self.aqi_categories = {
            (0, 50): ("Good", "🟢"),
            (51, 100): ("Moderate", "🟡"),
            (101, 150): ("Unhealthy for Sensitive Groups", "🟠"),
            (151, 200): ("Unhealthy", "🔴"),
            (201, 300): ("Very Unhealthy", "🟣"),
            (301, float('inf')): ("Hazardous", "🟤")
        }
        
    def load_models(self) -> bool:
        """Load all available models"""
        results = self.model_manager.load_all_models()
        return any(results.values())
    
    def get_aqi_category(self, aqi_value: float) -> Tuple[str, str]:
        """Get AQI category and emoji for a given AQI value"""
        for (min_val, max_val), (category, emoji) in self.aqi_categories.items():
            if min_val <= aqi_value <= max_val:
                return category, emoji
        return "Unknown", "❓"
    
    def predict_aqi(self, features_df: pd.DataFrame, model_type: str = 'best') -> Dict[str, Any]:
        """Predict AQI and return comprehensive results"""
        try:
            # Make prediction
            predictions = self.model_manager.predict_from_dataframe(features_df, model_type)
            
            # Handle single prediction
            if len(predictions) == 1:
                aqi_value = float(predictions[0])
                category, emoji = self.get_aqi_category(aqi_value)
                
                result = {
                    'aqi_value': aqi_value,
                    'category': category,
                    'emoji': emoji,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'is_hazardous': aqi_value > 150,
                    'predictions': predictions.tolist()
                }
                
            else:
                # Multiple predictions
                categories = [self.get_aqi_category(pred)[0] for pred in predictions]
                emojis = [self.get_aqi_category(pred)[1] for pred in predictions]
                
                result = {
                    'predictions': predictions.tolist(),
                    'categories': categories,
                    'emojis': emojis,
                    'mean_aqi': float(np.mean(predictions)),
                    'max_aqi': float(np.max(predictions)),
                    'min_aqi': float(np.min(predictions)),
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'is_hazardous': np.any(predictions > 150),
                    'hazardous_count': int(np.sum(predictions > 150))
                }
            
            print(f"AQI prediction completed: {result.get('aqi_value', result.get('mean_aqi')):.1f}")
            return result
            
        except Exception as e:
            
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_forecast(self, features_df: pd.DataFrame, hours_ahead: int = 72, model_type: str = 'best') -> Dict[str, Any]:
        """Predict AQI forecast for the next N hours"""
        print(f"🔮 Generating {hours_ahead} hour AQI forecast using {model_type} model...")
        
        # For simplicity, we'll assume the features represent current conditions
        # In a real implementation, you'd need future weather forecasts
        
        # Make base prediction
        base_prediction = self.predict_aqi(features_df, model_type=model_type)
        
        if 'error' in base_prediction:
            return base_prediction
        
        # Generate forecast by adding some trend and noise
        # This is a simplified approach - in practice, you'd use proper time series forecasting
        base_aqi = base_prediction.get('aqi_value', base_prediction.get('mean_aqi', 50))
        
        # Create forecast with some realistic variation
        np.random.seed(42)  # For reproducible results
        trend = np.linspace(0, 5, hours_ahead)  # Slight upward trend
        noise = np.random.normal(0, 3, hours_ahead)  # Random variation
        
        forecast_values = base_aqi + trend + noise
        forecast_values = np.clip(forecast_values, 0, 500)  # Keep within reasonable bounds
        
        # Generate timestamps
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours_ahead)]
        
        # Categorize forecast
        categories = [self.get_aqi_category(val)[0] for val in forecast_values]
        emojis = [self.get_aqi_category(val)[1] for val in forecast_values]
        
        # Calculate summary statistics
        hazardous_hours = np.sum(forecast_values > 150)
        unhealthy_hours = np.sum(forecast_values > 100)
        
        forecast_result = {
            'forecast_hours': hours_ahead,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'aqi_values': forecast_values.tolist(),
            'categories': categories,
            'emojis': emojis,
            'mean_aqi': float(np.mean(forecast_values)),
            'max_aqi': float(np.max(forecast_values)),
            'min_aqi': float(np.min(forecast_values)),
            'hazardous_hours': int(hazardous_hours),
            'unhealthy_hours': int(unhealthy_hours),
            'forecast_summary': {
                'next_24h_avg': float(np.mean(forecast_values[:24])),
                'next_48h_avg': float(np.mean(forecast_values[:48])),
                'next_72h_avg': float(np.mean(forecast_values[:72])) if hours_ahead >= 72 else None
            },
            'alerts': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate alerts
        if hazardous_hours > 0:
            forecast_result['alerts'].append(f"🚨 HAZARDOUS: {hazardous_hours} hours with AQI > 150")
        if unhealthy_hours > 12:
            forecast_result['alerts'].append(f"⚠️ WARNING: {unhealthy_hours} hours with unhealthy air quality")
        
        print(f"✅ Forecast generated: {hours_ahead} hours, avg AQI: {forecast_result['mean_aqi']:.1f}")
        
        
        return forecast_result
    
    def evaluate_model_performance(self, test_data: pd.DataFrame, target_column: str = 'us_aqi') -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if target_column not in test_data.columns:
            return {'error': f"Target column {target_column} not found in test data"}
        
        try:
            # Make predictions
            predictions = self.model_manager.predict_from_dataframe(test_data, model_type='best')
            actual = test_data[target_column].values
            
            # Calculate metrics
            r2 = r2_score(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            
            # Category accuracy
            actual_categories = [self.get_aqi_category(val)[0] for val in actual]
            predicted_categories = [self.get_aqi_category(val)[0] for val in predictions]
            category_accuracy = np.mean([a == p for a, p in zip(actual_categories, predicted_categories)])
            
            results = {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'category_accuracy': float(category_accuracy),
                'mean_absolute_error': float(mae),
                'samples_evaluated': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"📊 Model evaluation completed:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAE: {mae:.2f}")
            print(f"   Category Accuracy: {category_accuracy:.2%}")
            
            return results
            
        except Exception as e:
            
            return {'error': str(e)}

def main():
    """Test model utilities"""
    print("🚀 Testing Model Utilities")
    print("="*50)
    
    # Initialize predictor
    predictor = AQIPredictor()
    
    # Try to load models
    print("📥 Attempting to load models...")
    success = predictor.load_models()
    
    if not success:
        print("⚠️ No models found. Please train models first.")
        print("   Run: python models/train_sklearn.py")
        print("   Run: python models/train_dl.py")
        return
    
    # Create sample features for prediction
    sample_features = pd.DataFrame({
        'temperature': [25.0],
        'humidity': [65.0],
        'wind_speed': [3.5],
        'pressure_msl': [1013.2],
        'nitrogen_dioxide': [35.0],
        'ozone': [85.0],
        'carbon_monoxide': [0.7],
        'sulphur_dioxide': [12.0],
        # Add more features as needed
    })
    
    # Add missing features with default values
    required_features = predictor.model_manager.feature_columns
    for feature in required_features:
        if feature not in sample_features.columns:
            sample_features[feature] = 0.0
    
    print(f"📊 Sample features shape: {sample_features.shape}")
    
    # Test prediction
    print("🔮 Testing AQI prediction...")
    result = predictor.predict_aqi(sample_features)
    print(f"Result: {result}")
    
    # Test forecast
    print("📈 Testing AQI forecast...")
    forecast = predictor.predict_forecast(sample_features, hours_ahead=24)
    print(f"Forecast summary: Mean AQI = {forecast.get('mean_aqi', 'N/A')}")
    
    print("✅ Model utilities test completed!")

if __name__ == "__main__":
    main()
