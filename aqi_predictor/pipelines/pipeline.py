"""
Main Pipeline Orchestrator for AQI Prediction System
Coordinates data fetching, feature engineering, model training, and Hopsworks integration
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.fetch_data import OpenMeteoDataFetcher, HopsworksIntegration, save_data_locally
from features.feature_engineering import FeatureEngineer, AdvancedFeatureSelector
from models.train_sklearn import SklearnModelTrainer
from models.train_dl import DeepLearningModelTrainer
from models.model_utils import AQIPredictor

class AQIPipeline:
    """Enhanced AQI prediction pipeline with Hopsworks integration"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.data_fetcher = OpenMeteoDataFetcher()
        self.hopsworks = HopsworksIntegration()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = AdvancedFeatureSelector()
        self.sklearn_trainer = SklearnModelTrainer()
        self.dl_trainer = DeepLearningModelTrainer()
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("🚀 Enhanced AQI Prediction Pipeline Initialized")
        if self.hopsworks.enabled:
            print("✅ Hopsworks integration enabled")
        else:
            print("⚠️ Hopsworks integration disabled")
        
    
    def fetch_historical_data(self, years_back: int = 1, save_raw: bool = True) -> Optional[pd.DataFrame]:
        """Fetch complete data including historical and recent data"""
        print(f"\n📊 STEP 1: FETCHING COMPLETE DATASET ({years_back} years + recent)")
        print("="*60)
        
        try:
            # Fetch complete dataset (historical + recent)
            historical_data = self.data_fetcher.create_complete_dataset(
                years_back=years_back, 
                include_recent=True
            )
            
            if historical_data is None:
                print("❌ Failed to fetch complete data")
                return None
            
            # Save raw data
            if save_raw:
                raw_data_path = save_data_locally(historical_data, "complete_raw_data.csv", self.data_dir)
                print(f"💾 Raw data saved: {raw_data_path}")
            
            # Save to Hopsworks feature store if enabled
            if self.hopsworks.enabled:
                self.hopsworks.save_to_feature_store(historical_data, stage="raw")
            
            print(f"✅ Complete data fetching completed!")
            print(f"📊 Dataset shape: {historical_data.shape}")
            print(f"📅 Date range: {historical_data['datetime'].min()} to {historical_data['datetime'].max()}")
            
            return historical_data
            
        except Exception as e:
            print(f"❌ Error fetching complete data: {e}")
            return None
    
    def fetch_latest_data(self, hours_back: int = 48, save_raw: bool = True) -> Optional[pd.DataFrame]:
        """Fetch latest data for real-time predictions"""
        print(f"\n📊 FETCHING LATEST DATA ({hours_back} hours)")
        print("="*50)
        
        try:
            # Fetch recent data using enhanced fetcher
            latest_data = self.data_fetcher.fetch_recent_data(days_back=max(3, hours_back // 24))
            
            if latest_data is None:
                print("❌ Failed to fetch latest data")
                return None
            
            # Save raw data
            if save_raw:
                raw_data_path = save_data_locally(latest_data, "latest_raw_data.csv", self.data_dir)
                print(f"💾 Latest data saved: {raw_data_path}")
            
            print(f"✅ Latest data fetching completed!")
            print(f"📊 Dataset shape: {latest_data.shape}")
            print(f"📅 Date range: {latest_data['datetime'].min()} to {latest_data['datetime'].max()}")
            
            return latest_data
            
        except Exception as e:
            print(f"❌ Error fetching latest data: {e}")
            return None
    
    def engineer_features(self, raw_data: pd.DataFrame, save_features: bool = True) -> Optional[pd.DataFrame]:
        """Engineer features from raw data"""
        print(f"\n🔧 STEP 2: FEATURE ENGINEERING")
        print("="*60)
        
        try:
            # Engineer features
            print("🔧 Engineering features...")
            engineered_data = self.feature_engineer.engineer_features(raw_data)
            
            # Handle missing values
            print("🔧 Handling missing values...")
            engineered_data = self.feature_engineer.handle_missing_values(engineered_data)
            
            # Save engineered features
            if save_features:
                features_path = save_data_locally(engineered_data, "engineered_features.csv", self.data_dir)
                print(f"💾 Engineered features saved: {features_path}")
            
            # Save to Hopsworks feature store if enabled
            if self.hopsworks.enabled:
                self.hopsworks.save_to_feature_store(engineered_data, stage="engineered")
            
            print(f"✅ Feature engineering completed!")
            print(f"📊 Features shape: {engineered_data.shape}")
            
            return engineered_data
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            return None
    
    def select_features(self, engineered_data: pd.DataFrame, max_features: int = 25, 
                       save_features: bool = True) -> Optional[pd.DataFrame]:
        """Select best features from engineered data"""
        print(f"\n🔍 STEP 3: FEATURE SELECTION")
        print("="*60)
        
        try:
            # Select features
            selected_data, selected_feature_names = self.feature_selector.select_features(
                engineered_data, max_features=max_features
            )
            
            # Save selected features
            if save_features:
                selected_path = save_data_locally(selected_data, "selected_features.csv", self.data_dir)
                print(f"💾 Selected features saved: {selected_path}")
                
                # Save feature names
                feature_names_path = os.path.join(self.data_dir, "selected_feature_names.txt")
                with open(feature_names_path, 'w') as f:
                    f.write("\\n".join(selected_feature_names))
                print(f"📝 Feature names saved: {feature_names_path}")
            
            print(f"✅ Feature selection completed!")
            print(f"📊 Selected features: {len(selected_feature_names)}")
            
            print(f"Feature selection completed: {len(selected_feature_names)} features")
            
            return selected_data
            
        except Exception as e:
            print(f"❌ Error in feature selection: {e}")
            
            return None
    
    def train_sklearn_models(self, training_data: pd.DataFrame, test_size: float = 0.2) -> Optional[Dict[str, Any]]:
        """Train scikit-learn models"""
        print(f"\n🤖 STEP 4A: TRAINING SCIKIT-LEARN MODELS")
        print("="*60)
        
        try:
            # Train models
            results = self.sklearn_trainer.train_all_models(training_data, test_size=test_size)
            
            # Save best model
            if results:
                model_path = self.sklearn_trainer.save_best_model(self.models_dir)
                print(f"💾 Best sklearn model saved: {model_path}")
            
            print(f"✅ Scikit-learn training completed!")
            print(f"🏆 Best model: {self.sklearn_trainer.best_model_name}")
            
            print(f"Sklearn training completed. Best: {self.sklearn_trainer.best_model_name}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in sklearn training: {e}")
            
            return None
    
    def train_dl_models(self, training_data: pd.DataFrame, test_size: float = 0.2, epochs: int = 100) -> Optional[Dict[str, Any]]:
        """Train deep learning models"""
        print(f"\n🧠 STEP 4B: TRAINING DEEP LEARNING MODELS")
        print("="*60)
        
        try:
            # Train models
            results = self.dl_trainer.train_all_models(training_data, test_size=test_size, epochs=epochs)
            
            # Save best model
            if results:
                model_path = self.dl_trainer.save_best_model(self.models_dir)
                print(f"💾 Best deep learning model saved: {model_path}")
            
            print(f"✅ Deep learning training completed!")
            print(f"🏆 Best model: {self.dl_trainer.best_model_name}")
            
            print(f"Deep learning training completed. Best: {self.dl_trainer.best_model_name}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in deep learning training: {e}")
            
            return None
    
    def run_full_training_pipeline(self, years_back: int = 1, max_features: int = 25, 
                                  train_sklearn: bool = True, train_dl: bool = True,
                                  dl_epochs: int = 100) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        print("🚀 STARTING COMPLETE AQI TRAINING PIPELINE")
        print("="*70)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': [],
            'final_status': 'running'
        }
        
        try:
            # Step 1: Fetch historical data
            historical_data = self.fetch_historical_data(years_back=years_back)
            if historical_data is None:
                pipeline_results['errors'].append("Failed to fetch historical data")
                pipeline_results['final_status'] = 'failed'
                return pipeline_results
            pipeline_results['steps_completed'].append('data_fetching')
            
            # Step 2: Feature engineering
            engineered_data = self.engineer_features(historical_data)
            if engineered_data is None:
                pipeline_results['errors'].append("Failed to engineer features")
                pipeline_results['final_status'] = 'failed'
                return pipeline_results
            pipeline_results['steps_completed'].append('feature_engineering')
            
            # Step 3: Feature selection
            selected_data = self.select_features(engineered_data, max_features=max_features)
            if selected_data is None:
                pipeline_results['errors'].append("Failed to select features")
                pipeline_results['final_status'] = 'failed'
                return pipeline_results
            pipeline_results['steps_completed'].append('feature_selection')
            
            # Step 4a: Train sklearn models
            sklearn_results = None
            if train_sklearn:
                sklearn_results = self.train_sklearn_models(selected_data)
                if sklearn_results:
                    pipeline_results['steps_completed'].append('sklearn_training')
                    pipeline_results['sklearn_best_model'] = self.sklearn_trainer.best_model_name
                    pipeline_results['sklearn_best_r2'] = self.sklearn_trainer.results[self.sklearn_trainer.best_model_name]['test_r2']
                else:
                    pipeline_results['errors'].append("Failed to train sklearn models")
            
            # Step 4b: Train deep learning models
            dl_results = None
            if train_dl:
                dl_results = self.train_dl_models(selected_data, epochs=dl_epochs)
                if dl_results:
                    pipeline_results['steps_completed'].append('dl_training')
                    pipeline_results['dl_best_model'] = self.dl_trainer.best_model_name
                    pipeline_results['dl_best_r2'] = self.dl_trainer.results[self.dl_trainer.best_model_name]['test_r2']
                else:
                    pipeline_results['errors'].append("Failed to train deep learning models")
            
            # Final status
            if len(pipeline_results['errors']) == 0:
                pipeline_results['final_status'] = 'success'
            elif len(pipeline_results['steps_completed']) >= 3:  # At least data processing completed
                pipeline_results['final_status'] = 'partial_success'
            else:
                pipeline_results['final_status'] = 'failed'
            
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['data_shape'] = selected_data.shape if selected_data is not None else None
            
            # Print summary
            print(f"\n🎉 PIPELINE COMPLETED: {pipeline_results['final_status'].upper()}")
            print("="*60)
            print(f"✅ Steps completed: {', '.join(pipeline_results['steps_completed'])}")
            if pipeline_results['errors']:
                print(f"❌ Errors: {', '.join(pipeline_results['errors'])}")
            if 'sklearn_best_model' in pipeline_results:
                print(f"🏆 Best sklearn model: {pipeline_results['sklearn_best_model']} (R²: {pipeline_results['sklearn_best_r2']:.4f})")
            if 'dl_best_model' in pipeline_results:
                print(f"🏆 Best DL model: {pipeline_results['dl_best_model']} (R²: {pipeline_results['dl_best_r2']:.4f})")
            
            print(f"Pipeline completed with status: {pipeline_results['final_status']}")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['errors'].append(f"Unexpected error: {str(e)}")
            pipeline_results['final_status'] = 'failed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            print(f"❌ Pipeline failed with error: {e}")
            
            
            return pipeline_results
    
    def run_prediction_pipeline(self, hours_back: int = 48, forecast_hours: int = 72) -> Dict[str, Any]:
        """Run the prediction pipeline for real-time forecasting"""
        print("🔮 STARTING AQI PREDICTION PIPELINE")
        print("="*50)
        
        try:
            # Fetch latest data
            latest_data = self.fetch_latest_data(hours_back=hours_back, save_raw=False)
            if latest_data is None:
                return {'error': 'Failed to fetch latest data'}
            
            # Engineer features
            engineered_data = self.engineer_features(latest_data, save_features=False)
            if engineered_data is None:
                return {'error': 'Failed to engineer features'}
            
            # Load trained models and make predictions
            predictor = AQIPredictor(self.models_dir)
            if not predictor.load_models():
                return {'error': 'Failed to load trained models'}
            
            # Current prediction
            current_prediction = predictor.predict_aqi(engineered_data.tail(1))
            
            # Forecast
            forecast = predictor.predict_forecast(engineered_data.tail(1), hours_ahead=forecast_hours)
            
            result = {
                'current_prediction': current_prediction,
                'forecast': forecast,
                'data_timestamp': latest_data['datetime'].max().isoformat(),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Prediction pipeline completed!")
            print(f"🔮 Current AQI: {current_prediction.get('aqi_value', 'N/A')}")
            print(f"📈 Forecast mean: {forecast.get('mean_aqi', 'N/A')}")
            
            
            
            return result
            
        except Exception as e:
            error_result = {'error': str(e), 'timestamp': datetime.now().isoformat()}
            print(f"❌ Prediction pipeline failed: {e}")
            
            return error_result

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="AQI Prediction Pipeline")
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Pipeline mode: train, predict, or full')
    parser.add_argument('--years', type=int, default=1,
                       help='Years of historical data for training')
    parser.add_argument('--features', type=int, default=25,
                       help='Maximum number of features to select')
    parser.add_argument('--no-sklearn', action='store_true',
                       help='Skip sklearn model training')
    parser.add_argument('--no-dl', action='store_true',
                       help='Skip deep learning model training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for deep learning training')
    parser.add_argument('--forecast-hours', type=int, default=72,
                       help='Number of hours to forecast')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AQIPipeline()
    
    if args.mode == 'train':
        print("🚀 Running training pipeline only...")
        
        # Fetch and process data
        historical_data = pipeline.fetch_historical_data(years_back=args.years)
        if historical_data is None:
            return
        
        engineered_data = pipeline.engineer_features(historical_data)
        if engineered_data is None:
            return
        
        selected_data = pipeline.select_features(engineered_data, max_features=args.features)
        if selected_data is None:
            return
        
        # Train models
        if not args.no_sklearn:
            pipeline.train_sklearn_models(selected_data)
        
        if not args.no_dl:
            pipeline.train_dl_models(selected_data, epochs=args.epochs)
        
    elif args.mode == 'predict':
        print("🔮 Running prediction pipeline only...")
        result = pipeline.run_prediction_pipeline(forecast_hours=args.forecast_hours)
        print(f"Prediction result: {result}")
        
    elif args.mode == 'full':
        print("🚀 Running full pipeline...")
        result = pipeline.run_full_training_pipeline(
            years_back=args.years,
            max_features=args.features,
            train_sklearn=not args.no_sklearn,
            train_dl=not args.no_dl,
            dl_epochs=args.epochs
        )
        
        # After training, run a prediction
        if result['final_status'] in ['success', 'partial_success']:
            print("\\n🔮 Running prediction with trained models...")
            pred_result = pipeline.run_prediction_pipeline(forecast_hours=args.forecast_hours)
            print(f"Sample prediction: {pred_result}")

if __name__ == "__main__":
    main()
