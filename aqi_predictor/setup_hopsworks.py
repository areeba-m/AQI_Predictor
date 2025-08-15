#!/usr/bin/env python3
"""
Complete setup script for Hopsworks integration and pipeline execution
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🚀 AQI PREDICTOR SETUP WITH HOPSWORKS")
print("="*50)

# Check .env file
print("\n📋 CHECKING .ENV CONFIGURATION:")
api_key = os.getenv('HOPSWORKS_API_KEY')
project_name = os.getenv('HOPSWORKS_PROJECT_NAME')

if api_key:
    print(f"✅ HOPSWORKS_API_KEY: {'*' * (len(api_key) - 4) + api_key[-4:]}")
else:
    print("❌ HOPSWORKS_API_KEY not found")

if project_name:
    print(f"✅ HOPSWORKS_PROJECT_NAME: {project_name}")
else:
    print("❌ HOPSWORKS_PROJECT_NAME not found")

if not api_key or not project_name:
    print("\n🔧 PLEASE UPDATE YOUR .ENV FILE:")
    print("   1. Open the .env file in the project root")
    print("   2. Replace 'your_api_key_here' with your actual Hopsworks API key")
    print("   3. Update the project name if needed")
    print("\n   Example .env content:")
    print("   HOPSWORKS_API_KEY=your_actual_api_key_here")
    print("   HOPSWORKS_PROJECT_NAME=pearls_aqi_prediction")
    sys.exit(1)

# Test Hopsworks connection
print("\n🔌 TESTING HOPSWORKS CONNECTION:")
try:
    import hopsworks
    
    print("   Attempting to connect...")
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name
    )
    
    print(f"✅ Successfully connected to: {project.name}")
    
    # Test feature store access
    fs = project.get_feature_store()
    print(f"✅ Feature store access: {fs.name}")
    
    # Test model registry access
    mr = project.get_model_registry()
    print(f"✅ Model registry access available")
    
except Exception as e:
    print(f"❌ Hopsworks connection failed: {e}")
    print("\n🔧 TROUBLESHOOTING:")
    print("   1. Check your API key is correct")
    print("   2. Verify your project name exists")
    print("   3. Ensure you have access to the project")
    sys.exit(1)

print("\n🎉 SETUP COMPLETE! You can now run:")
print("   1. Training: python pipelines/pipeline.py --mode train")
print("   2. Prediction: python pipelines/pipeline.py --mode predict") 
print("   3. Streamlit: python -m streamlit run webapp/app.py")

# Check current environment variables
current_api_key = os.getenv('HOPSWORKS_API_KEY')
current_project = os.getenv('HOPSWORKS_PROJECT_NAME')

print(f"Current API Key: {'Set' if current_api_key else 'Not set'}")
print(f"Current Project: {current_project or 'Not set'}")

print("\n📝 To set up Hopsworks integration:")
print("1. Go to https://www.hopsworks.ai/")
print("2. Sign up for a free account")
print("3. Create a project called 'pearls_aqi_prediction'")
print("4. Go to Project Settings > API Keys")
print("5. Generate a new API key")
print("6. Set environment variables in PowerShell:")
print("   $env:HOPSWORKS_API_KEY = 'your-api-key-here'")
print("   $env:HOPSWORKS_PROJECT_NAME = 'pearls_aqi_prediction'")

print("\n⚠️ Alternative: If you don't want to use Hopsworks now:")
print("   - The pipeline will automatically fall back to local storage")
print("   - Data will be saved in the 'data/' folder")
print("   - You can enable Hopsworks later")

if not current_api_key:
    print("\n🏃‍♂️ For now, running with local storage only...")
    print("✅ Pipeline will work normally, just using local files")
