#!/usr/bin/env python3
"""
Complete pipeline runner with Hopsworks integration
"""

import sys
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_command(cmd, description):
    """Run a command and show its output"""
    print(f"\n🚀 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def main():
    if len(sys.argv) < 2:
        print("🎯 AQI PREDICTOR WITH HOPSWORKS")
        print("="*40)
        print("Usage: python setup_and_run.py [command]")
        print()
        print("Commands:")
        print("  setup     - Test Hopsworks connection")
        print("  train     - Train models from scratch")
        print("  predict   - Run prediction pipeline")
        print("  app       - Launch Streamlit app")
        print("  full      - Full pipeline (train + predict)")
        return
    
    command = sys.argv[1].lower()
    python_exe = r".\aqi_env\Scripts\python.exe"
    
    if command == "setup":
        run_command(f"{python_exe} setup_hopsworks.py", "Testing Hopsworks setup")
    
    elif command == "train":
        print("🔄 Starting training pipeline with Hopsworks integration...")
        run_command(f"{python_exe} pipelines\\pipeline.py --mode train --years 1 --features 25 --epochs 50", 
                   "Training pipeline")
    
    elif command == "predict":
        print("🔮 Running prediction pipeline...")
        run_command(f"{python_exe} pipelines\\pipeline.py --mode predict --forecast-hours 72", 
                   "Prediction pipeline")
    
    elif command == "app":
        print("🌐 Launching Streamlit app...")
        run_command(f"{python_exe} -m streamlit run webapp\\app.py", 
                   "Streamlit app")
    
    elif command == "full":
        print("🚀 Running full pipeline...")
        if run_command(f"{python_exe} pipelines\\pipeline.py --mode full --years 1 --features 25 --epochs 50", 
                      "Full pipeline"):
            print("\n🌐 Now launching Streamlit app...")
            run_command(f"{python_exe} -m streamlit run webapp\\app.py", 
                       "Streamlit app")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: setup, train, predict, app, or full")

if __name__ == "__main__":
    main()
