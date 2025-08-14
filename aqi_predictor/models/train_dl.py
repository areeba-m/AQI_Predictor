"""
Deep Learning Model Training Pipeline for AQI Prediction
Includes LSTM, feedforward neural networks, and ensemble models using TensorFlow/Keras
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DeepLearningModelTrainer:
    """Comprehensive deep learning model training for AQI prediction"""
    
    def __init__(self, target_variable: str = 'us_aqi', random_state: int = 42):
        self.target_variable = target_variable
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = []
        
        # Configure TensorFlow
        mixed_precision.set_global_policy('mixed_float16')        
        # GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🖥️ GPU detected: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                print(f"⚠️ GPU configuration error: {e}")
        else:
            print("🖥️ Running on CPU")
            
    
    def prepare_data_feedforward(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for feedforward neural networks only"""
        print("📊 Preparing data for feedforward neural network training...")
        
        # Remove datetime column for features
        self.feature_columns = [col for col in df.columns if col not in [self.target_variable, 'datetime']]
        
        X = df[self.feature_columns].values
        y = df[self.target_variable].values.reshape(-1, 1)
        
        print(f"✅ Original shape - X: {X.shape}, y: {y.shape}")
        print(f"✅ Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Scale features and target
        print("🔧 Scaling features and target...")
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled.ravel(), test_size=test_size, 
            random_state=self.random_state, shuffle=True
        )
        
        print(f"✅ Data prepared:")
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for deep learning models"""
        print("📊 Preparing data for deep learning training...")
        
        
        # Remove datetime column for features
        self.feature_columns = [col for col in df.columns if col not in [self.target_variable, 'datetime']]
        
        X = df[self.feature_columns].values
        y = df[self.target_variable].values.reshape(-1, 1)
        
        print(f"✅ Original shape - X: {X.shape}, y: {y.shape}")
        print(f"✅ Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Scale features and target
        print("🔧 Scaling features and target...")
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled.ravel(), test_size=test_size, 
            random_state=self.random_state, shuffle=True
        )
        
        # Create sequences for LSTM
        print(f"🔄 Creating sequences of length {sequence_length} for LSTM...")
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        print(f"✅ Final shapes:")
        print(f"   📊 Dense - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   📊 LSTM - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
        
        print(f"Data prepared - Dense Train: {X_train.shape}, LSTM Train: {X_train_seq.shape}")
        
        return X_train, X_test, y_train, y_test, X_train_seq, X_test_seq, y_train_seq, y_test_seq
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if len(X) < sequence_length:
            print(f"⚠️ Not enough data for sequence length {sequence_length}, using available data")
            sequence_length = len(X) - 1
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_feedforward_model(self, input_dim: int, architecture: str = 'deep') -> Model:
        """Create feedforward neural network"""
        print(f"🧠 Creating {architecture} feedforward model...")
        
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        
        if architecture == 'simple':
            # Simple architecture
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.3))
            model.add(Dense(1))
            
        elif architecture == 'deep':
            # Deep architecture
            model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.3))
            
            model.add(Dense(1))
        
        elif architecture == 'wide':
            # Wide architecture
            model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            
            model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ {architecture.capitalize()} feedforward model created")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, model: Model, model_name: str, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray, epochs: int = 100, 
                   batch_size: int = 32, verbose: int = 1) -> Dict[str, Any]:
        """Train a single model with comprehensive evaluation"""
        print(f"\n🚀 TRAINING {model_name}")
        print("-" * 50)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"🏋️ Training {model_name} for up to {epochs} epochs...")
        print(f"📊 Batch size: {batch_size}")
        print(f"📈 Monitoring validation loss with early stopping")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"✅ {model_name} training completed!")
        
        # Make predictions
        print(f"🔮 Making predictions with {model_name}...")
        y_train_pred_scaled = model.predict(X_train, verbose=0)
        y_test_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        
        # Inverse transform actual values
        y_train_actual = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        print(f"📈 Calculating metrics for {model_name}...")
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
        test_mae = mean_absolute_error(y_test_actual, y_test_pred)
        
        # Get final loss values
        final_train_loss = min(history.history['loss'])
        final_val_loss = min(history.history['val_loss'])
        
        print(f"📈 Training R²:    {train_r2:.4f}")
        print(f"📈 Training RMSE:  {train_rmse:.2f}")
        print(f"📈 Test R²:        {test_r2:.4f}")
        print(f"📈 Test RMSE:      {test_rmse:.2f}")
        print(f"📈 Test MAE:       {test_mae:.2f}")
        print(f"📈 Final train loss: {final_train_loss:.4f}")
        print(f"📈 Final val loss:   {final_val_loss:.4f}")
        
        # Check for overfitting
        is_healthy = self._detect_overfitting_dl(model_name, train_r2, test_r2, final_train_loss, final_val_loss)
        
        result = {
            'model': model,
            'model_name': model_name,
            'history': history.history,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'is_healthy': is_healthy,
            'epochs_trained': len(history.history['loss']),
            'predictions': y_test_pred,
            'y_test': y_test_actual
        }
        
        print(f"{model_name} training completed - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f}")
        
        return result
    
    def _detect_overfitting_dl(self, model_name: str, train_r2: float, test_r2: float, 
                              train_loss: float, val_loss: float) -> bool:
        """Detect overfitting in deep learning models"""
        print(f"\n🔍 OVERFITTING ANALYSIS for {model_name}")
        print("-" * 40)
        
        overfitting_flags = []
        
        # Check R² scores
        if train_r2 > 0.98:
            overfitting_flags.append(f"⚠️  Extremely high training R²: {train_r2:.4f}")
        
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.15:
            overfitting_flags.append(f"⚠️  Large R² gap: {r2_gap:.4f}")
        
        # Check loss values
        loss_ratio = val_loss / train_loss if train_loss > 0 else 1
        if loss_ratio > 1.3:
            overfitting_flags.append(f"⚠️  Validation loss much higher: {loss_ratio:.2f}x")
        
        if test_r2 < 0.6 and train_r2 > 0.8:
            overfitting_flags.append(f"⚠️  Poor generalization: train={train_r2:.3f}, test={test_r2:.3f}")
        
        if len(overfitting_flags) > 0:
            print("🚨 OVERFITTING DETECTED:")
            for flag in overfitting_flags:
                print(f"   {flag}")
            print("💡 Consider: more dropout, regularization, or early stopping")
        else:
            print("✅ No overfitting detected")
        
        return len(overfitting_flags) == 0
    
    def train_all_models(self, df: pd.DataFrame, test_size: float = 0.2, epochs: int = 100) -> Dict[str, Any]:
        """Train all deep learning models"""
        print("🚀 STARTING COMPREHENSIVE DEEP LEARNING TRAINING")
        print("="*70)
        
        # Prepare data (feedforward networks only - no sequences needed)
        X_train, X_test, y_train, y_test = self.prepare_data_feedforward(df, test_size)
        
        input_dim = X_train.shape[1]
        
        print(f"📊 Input dimensions - Dense: {input_dim}")
        
        # Define models to train (feedforward networks only - LSTM removed due to underperformance)
        models_to_train = [
            ('Simple Feedforward', lambda: self.create_feedforward_model(input_dim, 'simple'), X_train, X_test, y_train, y_test),
            ('Deep Feedforward', lambda: self.create_feedforward_model(input_dim, 'deep'), X_train, X_test, y_train, y_test),
            ('Wide Feedforward', lambda: self.create_feedforward_model(input_dim, 'wide'), X_train, X_test, y_train, y_test),
        ]
        
        # Train models
        results = {}
        
        for name, model_creator, X_tr, X_te, y_tr, y_te in models_to_train:
            print(f"\n{'='*70}")
            print(f"🧠 TRAINING MODEL: {name}")
            print(f"{'='*70}")
            
            try:
                model = model_creator()
                result = self.train_model(model, name, X_tr, X_te, y_tr, y_te, epochs=epochs)
                results[name] = result
                
                print(f"✅ {name} training completed successfully!")
                
                # Clear memory
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                
                continue
        
        self.results = results
        
        # Find best model
        self._select_best_model()
        
        # Print comparison
        self._print_model_comparison()
        
        print(f"Deep learning training completed for {len(results)} models")
        
        return results
    
    def _select_best_model(self):
        """Select the best performing model"""
        if not self.results:
            return
        
        print(f"\n🏆 SELECTING BEST DEEP LEARNING MODEL")
        print("="*50)
        
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
        
        print(f"Best deep learning model selected: {self.best_model_name}")
    
    def _print_model_comparison(self):
        """Print comprehensive model comparison"""
        if not self.results:
            return
        
        print(f"\n📊 DEEP LEARNING MODEL COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = {}
        for name, result in self.results.items():
            comparison_data[name] = {
                'Test R²': result['test_r2'],
                'Test RMSE': result['test_rmse'],
                'Test MAE': result['test_mae'],
                'Train R²': result['train_r2'],
                'Epochs': result['epochs_trained'],
                'Final Val Loss': result['final_val_loss'],
                'Healthy': '✅' if result['is_healthy'] else '❌'
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        print(comparison_df.round(4))
    
    def save_best_model(self, model_dir: str = "models") -> str:
        """Save the best model and metadata"""
        if self.best_model is None:
            print("❌ No best model to save")
            return ""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"best_dl_model_{self.best_model_name.lower().replace(' ', '_')}.h5")
        self.best_model.save(model_path)
        
        # Save scalers
        scaler_X_path = os.path.join(model_dir, "dl_scaler_X.joblib")
        scaler_y_path = os.path.join(model_dir, "dl_scaler_y.joblib")
        
        import joblib
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': 'deep_learning',
            'target_variable': self.target_variable,
            'test_r2': float(self.results[self.best_model_name]['test_r2']),
            'test_rmse': float(self.results[self.best_model_name]['test_rmse']),
            'test_mae': float(self.results[self.best_model_name]['test_mae']),
            'epochs_trained': int(self.results[self.best_model_name]['epochs_trained']),
            'is_healthy': bool(self.results[self.best_model_name]['is_healthy']),
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, "dl_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Best deep learning model saved:")
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Scaler X: {scaler_X_path}")
        print(f"   📁 Scaler y: {scaler_y_path}")
        print(f"   📁 Metadata: {metadata_path}")
        
        print(f"Best deep learning model saved: {self.best_model_name}")
        
        return model_path

def main():
    """Test the deep learning model training pipeline"""
    print("🚀 Testing Deep Learning Model Training Pipeline")
    print("="*60)
    
    # Create sample data for testing
    from features.feature_engineering import FeatureEngineer, AdvancedFeatureSelector
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='H')
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
    trainer = DeepLearningModelTrainer()
    results = trainer.train_all_models(selected_data, test_size=0.2, epochs=10)  # Few epochs for testing
    
    # Save best model
    model_path = trainer.save_best_model()
    
    print(f"\n✅ Deep learning model training pipeline test completed!")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"📁 Saved to: {model_path}")

if __name__ == "__main__":
    main()
