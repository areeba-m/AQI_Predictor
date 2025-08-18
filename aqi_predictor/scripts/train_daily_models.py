#!/usr/bin/env python3
"""
Daily Model Training Script for GitHub Actions
Performs feature engineering and model training using Hopsworks data
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipelines.fetch_data import run_daily_model_pipeline

def main():
    """Run daily model training pipeline"""
    print("🤖 DAILY MODEL TRAINING JOB STARTED")
    print("="*60)
    
    try:
        success = run_daily_model_pipeline()
        
        if success:
            print("\n✅ DAILY TRAINING JOB COMPLETED SUCCESSFULLY")
            sys.exit(0)
        else:
            print("\n❌ DAILY TRAINING JOB FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 DAILY TRAINING JOB CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
