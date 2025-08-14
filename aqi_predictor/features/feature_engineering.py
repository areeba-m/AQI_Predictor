"""
Feature Engineering Pipeline for AQI Prediction
Comprehensive feature engineering based on the notebook implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Configure logging

class FeatureEngineer:
    """Comprehensive feature engineering for AQI prediction"""
    
    def __init__(self, target_variable: str = 'us_aqi'):
        self.target_variable = target_variable
        self.leaky_features = [
            'pm2_5', 'pm10',           # Direct AQI inputs - PRIMARY POLLUTANTS
            'european_aqi',            # Alternative AQI measure
            'dust'                     # Often highly correlated with PM values
        ]
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for AQI prediction using Open-Meteo data
        IMPORTANT: Excludes PM2.5, PM10, and European AQI to prevent data leakage
        
        Parameters:
        df (pd.DataFrame): Raw data with datetime and weather/air quality features
        
        Returns:
        pd.DataFrame: DataFrame with engineered features
        """
        print("🔧 Starting comprehensive feature engineering...")
        df_features = df.copy()
        print(f"🎯 Target Variable: {self.target_variable}")
        
        # =============================================
        # DATA LEAKAGE PREVENTION
        # =============================================
        # Remove features that directly contribute to AQI calculation or are highly correlated
        leaky_features_found = []
        for col in df_features.columns:
            for leaky in ['pm2_5', 'pm10', 'european_aqi', 'dust']:
                if leaky in col.lower():
                    leaky_features_found.append(col)
        
        leaky_features_found = list(set(leaky_features_found))  # Remove duplicates
        if leaky_features_found:
            print(f"⚠️  Removing leaky features: {leaky_features_found}")
            df_features = df_features.drop(columns=leaky_features_found, errors='ignore')
        
        # =============================================
        # 1. TIME-BASED FEATURES
        # =============================================
        print("📅 Creating time-based features...")
        df_features['hour'] = df_features['datetime'].dt.hour
        df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
        df_features['month'] = df_features['datetime'].dt.month
        df_features['day_of_year'] = df_features['datetime'].dt.dayofyear
        df_features['season'] = df_features['month'].map({12:0, 1:0, 2:0,  # Winter
                                                         3:1, 4:1, 5:1,     # Spring  
                                                         6:2, 7:2, 8:2,     # Summer
                                                         9:3, 10:3, 11:3})  # Autumn
        
        # =============================================
        # 2. CYCLICAL ENCODING FOR PERIODIC FEATURES
        # =============================================
        print("🔄 Creating cyclical features...")
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # =============================================
        # 3. WEATHER INTERACTIONS (NO PM FEATURES)
        # =============================================
        print("🌤️ Creating weather interaction features...")
        df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
        df_features['wind_pressure'] = df_features['wind_speed'] * df_features['pressure_msl']
        df_features['temp_wind'] = df_features['temperature'] * df_features['wind_speed']
        df_features['humidity_pressure'] = df_features['humidity'] * df_features['surface_pressure']
        df_features['cloud_wind'] = df_features['cloud_cover'] * df_features['wind_speed']
        df_features['temp_pressure'] = df_features['temperature'] * df_features['pressure_msl']
        
        # =============================================
        # 4. DERIVED WEATHER FEATURES
        # =============================================
        print("🌡️ Creating derived weather features...")
        df_features['heat_index'] = df_features['temperature'] + 0.5 * (df_features['humidity'] - 50)
        df_features['pressure_diff'] = df_features['pressure_msl'] - df_features['surface_pressure']
        
        # Wind chill calculation with safety checks
        wind_speed_safe = np.maximum(df_features['wind_speed'], 0.1)  # Prevent zero wind speed
        df_features['wind_chill'] = (35.74 + 0.6215 * df_features['temperature'] - 
                                   35.75 * (wind_speed_safe ** 0.16) + 
                                   0.4275 * df_features['temperature'] * (wind_speed_safe ** 0.16))
        
        df_features['dew_point_spread'] = df_features['temperature'] - df_features['dew_point']
        df_features['temp_range'] = df_features['apparent_temperature'] - df_features['temperature']
        
        # =============================================
        # 5. SAFE POLLUTION INTERACTIONS (NO PM)
        # =============================================
        print("🌬️ Creating pollution interaction features...")
        # Use only secondary pollutants that don't directly calculate AQI
        # Add small epsilon to prevent division by zero/infinity
        epsilon = 1e-8
        df_features['no2_co_ratio'] = df_features['nitrogen_dioxide'] / (df_features['carbon_monoxide'] + epsilon)
        df_features['temp_ozone'] = df_features['temperature'] * df_features['ozone']
        df_features['humidity_no2'] = df_features['humidity'] * df_features['nitrogen_dioxide']
        df_features['wind_so2'] = df_features['wind_speed'] * df_features['sulphur_dioxide']
        df_features['pressure_co'] = df_features['pressure_msl'] * df_features['carbon_monoxide']
        
        # =============================================
        # 6. TIME SERIES FEATURES (SAFE VARIABLES ONLY)
        # =============================================
        print("📈 Creating time series features...")
        safe_variables = ['temperature', 'humidity', 'wind_speed', 'pressure_msl', 
                         'nitrogen_dioxide', 'ozone', 'carbon_monoxide', 'sulphur_dioxide']
        
        for window in [3, 6, 12, 24]:  # 3h, 6h, 12h, 24h windows
            for var in safe_variables:
                if var in df_features.columns:
                    df_features[f'{var}_roll_{window}h'] = df_features[var].rolling(window, min_periods=1).mean()
                    df_features[f'{var}_std_{window}h'] = df_features[var].rolling(window, min_periods=1).std()
        
        # =============================================
        # 7. LAG FEATURES (SAFE VARIABLES ONLY)
        # =============================================
        print("⏰ Creating lag features...")
        for lag in [1, 3, 6, 12, 24]:
            for var in safe_variables:
                if var in df_features.columns:
                    df_features[f'{var}_lag_{lag}h'] = df_features[var].shift(lag)
        
        # =============================================
        # 8. CHANGE RATES (SAFE VARIABLES ONLY)
        # =============================================
        print("📊 Creating change rate features...")
        for var in safe_variables:
            if var in df_features.columns:
                df_features[f'{var}_change_1h'] = df_features[var].diff(1)
                df_features[f'{var}_change_3h'] = df_features[var].diff(3)
                # Safe percentage change calculation
                df_features[f'{var}_pct_change_1h'] = df_features[var].pct_change(1).replace([np.inf, -np.inf], 0)
        
        # =============================================
        # 9. WEATHER CATEGORIES
        # =============================================
        print("🏷️ Creating categorical features...")
        df_features['is_hot'] = (df_features['temperature'] > df_features['temperature'].quantile(0.75)).astype(int)
        df_features['is_cold'] = (df_features['temperature'] < df_features['temperature'].quantile(0.25)).astype(int)
        df_features['is_humid'] = (df_features['humidity'] > 70).astype(int)
        df_features['is_dry'] = (df_features['humidity'] < 30).astype(int)
        df_features['is_windy'] = (df_features['wind_speed'] > df_features['wind_speed'].quantile(0.75)).astype(int)
        df_features['is_calm'] = (df_features['wind_speed'] < df_features['wind_speed'].quantile(0.25)).astype(int)
        df_features['is_rainy'] = (df_features['precipitation'] > 0).astype(int)
        df_features['is_cloudy'] = (df_features['cloud_cover'] > 75).astype(int)
        df_features['is_clear'] = (df_features['cloud_cover'] < 25).astype(int)
        
        # =============================================
        # 10. TIME CATEGORIES
        # =============================================
        print("⏰ Creating time-based categories...")
        df_features['is_rush_hour'] = ((df_features['hour'].between(7, 9)) | 
                                      (df_features['hour'].between(17, 19))).astype(int)
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
        df_features['is_morning'] = (df_features['hour'].between(6, 11)).astype(int)
        df_features['is_afternoon'] = (df_features['hour'].between(12, 17)).astype(int)
        df_features['is_evening'] = (df_features['hour'].between(18, 21)).astype(int)
        
        # =============================================
        # FINAL CLEANUP: REMOVE DATETIME
        # =============================================
        # Keep datetime for now, will remove later if needed
        
        print(f"\n✅ Feature engineering completed!")
        print(f"📊 Original features: {len(df.columns)}")
        print(f"📊 Engineered features: {len(df_features.columns)}")
        print(f"📊 Total features created: {len(df_features.columns) - len(df.columns)}")
        print(f"🎯 Target variable: {self.target_variable}")
        
        return df_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values created during feature engineering"""
        print("🔧 Handling missing values...")
        
        missing_before = df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Handle infinite values first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill for time series continuity
        df = df.fillna(method='ffill')
        
        # Backward fill for remaining missing values
        df = df.fillna(method='bfill')
        
        # Fill any remaining missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        
        return df

class AdvancedFeatureSelector:
    """Advanced feature selection with data leakage prevention"""
    
    def __init__(self, target_variable: str = 'us_aqi'):
        self.target_variable = target_variable
        self.selected_features = []
        
    def select_features(self, df: pd.DataFrame, max_features: int = 25) -> Tuple[pd.DataFrame, List[str]]:
        """
        Comprehensive feature selection pipeline
        
        Parameters:
        df (pd.DataFrame): DataFrame with engineered features
        max_features (int): Maximum number of features to select
        
        Returns:
        Tuple[pd.DataFrame, List[str]]: Selected dataset and feature names
        """
        print(f"\n🔍 ADVANCED FEATURE SELECTION PIPELINE")
        
        if self.target_variable not in df.columns:
            raise ValueError(f"Target variable {self.target_variable} not found in data")
        
        feature_columns = [col for col in df.columns if col not in [self.target_variable, 'datetime']]
        X = df[feature_columns].copy()
        y = df[self.target_variable].copy()
        
        # Clean infinite and NaN values from features
        print("🧹 Cleaning infinite and NaN values...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Also clean target variable
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median())
        
        print(f"🎯 Target variable: {self.target_variable}")
        print(f"📊 Initial features: {len(feature_columns)}")
        
        # =============================================
        # STEP 1: REMOVE ZERO/LOW VARIANCE FEATURES
        # =============================================
        print(f"\n🔍 STEP 1: REMOVING LOW VARIANCE FEATURES")
        
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        variance_features = X.columns[variance_selector.get_support()].tolist()
        
        removed_variance = set(feature_columns) - set(variance_features)
        print(f"🗑️  Removed {len(removed_variance)} low variance features")
        
        # =============================================
        # STEP 2: REMOVE HIGHLY CORRELATED FEATURES
        # =============================================
        print(f"\n🔍 STEP 2: REMOVING HIGHLY CORRELATED FEATURES (r > 0.95)")
        
        X_var = pd.DataFrame(X_variance, columns=variance_features)
        correlation_matrix = X_var.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove (correlation > 0.95)
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        
        X_corr = X_var.drop(columns=high_corr_features)
        correlation_features = X_corr.columns.tolist()
        
        print(f"🗑️  Removed {len(high_corr_features)} highly correlated features")
        
        # =============================================
        # STEP 3: DATA LEAKAGE CHECK
        # =============================================
        print(f"\n🔍 STEP 3: DATA LEAKAGE DETECTION")
        
        # Check correlations with target
        target_correlations = X_corr.corrwith(y).abs().sort_values(ascending=False)
        
        # Flag suspiciously high correlations (might indicate leakage)
        suspicious_features = target_correlations[target_correlations > 0.85].index.tolist()
        
        if len(suspicious_features) > 0:
            print(f"⚠️  WARNING: {len(suspicious_features)} features have suspiciously high correlation with target (>0.85):")
            for feat in suspicious_features[:5]:
                print(f"    {feat}: {target_correlations[feat]:.3f}")
            print(f"   Consider reviewing these features for potential data leakage!")
        else:
            print(f"✅ No suspicious correlations detected (all < 0.85)")
        
        # =============================================
        # STEP 4: MUTUAL INFORMATION SELECTION
        # =============================================
        print(f"\n🔍 STEP 4: MUTUAL INFORMATION ANALYSIS")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X_corr.fillna(0), y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': correlation_features,
            'mutual_info_score': mi_scores
        }).sort_values('mutual_info_score', ascending=False)
        
        # Select top features by mutual information
        top_mi_features = mi_df.head(max_features)['feature'].tolist()
        
        print(f"📈 Top 10 Features by Mutual Information:")
        for i, (_, row) in enumerate(mi_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} ({row['mutual_info_score']:.4f})")
        
        # =============================================
        # STEP 5: COMBINE AND SCORE FEATURES
        # =============================================
        print(f"\n🔍 STEP 5: FINAL FEATURE SELECTION")
        
        # Create scoring system
        feature_scores = {}
        
        for i, (_, row) in enumerate(mi_df.head(max_features).iterrows()):
            feature = row['feature']
            score = max_features - i  # Higher rank = more points
            
            # Penalty for high correlation with target (potential leakage)
            target_corr = abs(target_correlations.get(feature, 0))
            if target_corr > 0.7:
                score -= 5  # Penalty for suspicious correlation
            
            feature_scores[feature] = score
        
        # Sort by score and select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        final_features = [feat for feat, score in sorted_features[:max_features]]
        
        print(f"📈 TOP {len(final_features)} SELECTED FEATURES:")
        print("="*60)
        for i, (feature, score) in enumerate(sorted_features[:max_features]):
            mi_score = mi_df[mi_df['feature'] == feature]['mutual_info_score'].iloc[0]
            corr_score = target_correlations.get(feature, 0)
            print(f"{i+1:2d}. {feature:<30} | Score: {score:3d} | MI: {mi_score:.3f} | Corr: {corr_score:.3f}")
        
        # Create final dataset
        final_columns = final_features + [self.target_variable]
        if 'datetime' in df.columns:
            final_columns = ['datetime'] + final_columns
        
        df_final = df[final_columns].copy()
        
        print(f"\n📊 FEATURE SELECTION SUMMARY:")
        print("="*50)
        print(f"📊 Initial features:        {len(feature_columns)}")
        print(f"📊 After variance filter:   {len(variance_features)}")
        print(f"📊 After correlation filter: {len(correlation_features)}")
        print(f"📊 Final selected features: {len(final_features)}")
        print(f"🎯 Target variable:         {self.target_variable}")
        print(f"📅 Dataset shape:           {df_final.shape}")
        
        self.selected_features = final_features
        
        return df_final, final_features

def main():
    """Test feature engineering pipeline"""
    print("🚀 Testing Feature Engineering Pipeline")
    print("="*50)
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='H')
    n_samples = len(dates)
    
    # Create sample data that mimics Open-Meteo API response
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
    
    # Test feature engineering
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features(sample_data)
    engineered_data = engineer.handle_missing_values(engineered_data)
    
    print(f"✅ Feature engineering completed!")
    print(f"📊 Engineered data shape: {engineered_data.shape}")
    
    # Test feature selection
    selector = AdvancedFeatureSelector()
    selected_data, selected_features = selector.select_features(engineered_data, max_features=15)
    
    print(f"✅ Feature selection completed!")
    print(f"📊 Selected data shape: {selected_data.shape}")
    print(f"📋 Selected features: {len(selected_features)}")
    
    return selected_data, selected_features

if __name__ == "__main__":
    main()
