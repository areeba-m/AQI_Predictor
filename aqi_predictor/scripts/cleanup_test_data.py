#!/usr/bin/env python3
"""
Cleanup Script to Remove Test Data from Hopsworks
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipelines.fetch_data import HopsworksIntegration

def main():
    """Clean test data from Hopsworks"""
    print("🧹 CLEANING TEST DATA FROM HOPSWORKS")
    print("="*50)
    
    try:
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            print("❌ Hopsworks not enabled - check your .env configuration")
            sys.exit(1)
        
        success = hops.clean_test_data()
        
        if success:
            print("\n✅ CLEANUP COMPLETED SUCCESSFULLY")
            print("🔍 Check Hopsworks UI to verify test data is removed")
        else:
            print("\n⚠️ CLEANUP COMPLETED WITH SOME WARNINGS")
            
    except Exception as e:
        print(f"\n💥 CLEANUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
