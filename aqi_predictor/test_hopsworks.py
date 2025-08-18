"""
Simple test script to verify Hopsworks data insertion and model registry upload
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.fetch_data import HopsworksIntegration

def test_data_insertion():
    """Test inserting a small dataset into Hopsworks feature store"""
    print("🧪 Testing Hopsworks Data Insertion")
    print("="*50)
    
    # Create simple test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='H')
    n_samples = len(dates)
    
    test_data = pd.DataFrame({
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
    
    print(f"📊 Created test data: {test_data.shape}")
    print(f"📅 Date range: {test_data['datetime'].min()} to {test_data['datetime'].max()}")
    
    # Test Hopsworks integration
    hops = HopsworksIntegration()
    
    if not hops.enabled:
        print("❌ Hopsworks integration not enabled!")
        print("   Check your .env file and Hopsworks configuration")
        return False
    
    print("✅ Hopsworks integration enabled")
    
    # Test data insertion
    success = hops.save_to_feature_store(test_data, stage="test")
    
    if success:
        print("\n🎉 SUCCESS! Test data inserted to Hopsworks!")
        print("   💡 Check Hopsworks UI to verify data appears")
        return True
    else:
        print("\n❌ FAILED to insert test data to Hopsworks")
        return False

def test_model_upload():
    """Test uploading a simple model to Hopsworks model registry"""
    print("\n🧪 Testing Hopsworks Model Upload")
    print("="*50)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        import joblib
        import tempfile
        
        # Create simple test model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        print("✅ Created test Random Forest model")
        
        # Save model temporarily with proper metadata
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            joblib.dump(model, model_path)
            
            # Test model upload
            hops = HopsworksIntegration()
            
            if not hops.enabled:
                print("❌ Hopsworks integration not enabled!")
                return False
            
            success = hops.save_model(temp_dir, "test_aqi_model", "sklearn")
            
            if success:
                print("🎉 SUCCESS! Test model uploaded to Hopsworks model registry!")
                return True
            else:
                print("❌ FAILED to upload test model to Hopsworks")
                return False
                
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all Hopsworks tests"""
    print("🚀 Hopsworks Integration Test Suite")
    print("="*60)
    
    # Test 1: Data insertion
    data_success = test_data_insertion()
    
    # Test 2: Model upload
    model_success = test_model_upload()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print(f"   📊 Data insertion: {'✅ PASS' if data_success else '❌ FAIL'}")
    print(f"   🤖 Model upload: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if data_success and model_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your Hopsworks integration is working correctly")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("   Check your Hopsworks configuration and network connection")

if __name__ == "__main__":
    main()