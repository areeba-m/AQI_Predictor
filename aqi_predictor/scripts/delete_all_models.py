#!/usr/bin/env python3
"""
Delete All Models from Hopsworks Model Registry
Script to clean up all models from the Hopsworks model registry
Use with caution - this will permanently delete ALL models!
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def delete_all_models():
    """
    Delete all models from Hopsworks model registry
    This is a destructive operation - use with caution!
    """
    print("🗑️ HOPSWORKS MODEL REGISTRY CLEANUP")
    print("="*60)
    print("⚠️ WARNING: This will delete ALL models from Hopsworks!")
    print("⚠️ This operation cannot be undone!")
    print("="*60)
    
    # Get confirmation
    confirmation = input("Type 'DELETE ALL MODELS' to confirm: ")
    if confirmation != "DELETE ALL MODELS":
        print("❌ Operation cancelled - confirmation text did not match")
        return False
    
    try:
        # Import Hopsworks integration
        from pipelines.fetch_data import HopsworksIntegration
        
        print("\n🔗 Connecting to Hopsworks...")
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            print("❌ Hopsworks integration is not enabled")
            print("   Check your .env file and config.yaml")
            return False
        
        print("✅ Connected to Hopsworks")
        print(f"📁 Project: {hops.project.name}")
        
        # Get model registry
        mr = hops.mr
        if mr is None:
            print("❌ Could not access model registry")
            return False
        
        print("\n🔍 Scanning for existing models...")
        
        # Get all models by trying common model names first
        try:
            all_models = []
            
            # Common model names to check
            common_model_names = [
                'sklearn_aqi_model', 'dl_aqi_model', 
                'ridge_regression_model', 'random_forest_model',
                'deep_feedforward_model', 'best_sklearn_model', 'best_dl_model'
            ]
            
            print("🔍 Searching for models with common names...")
            for model_name in common_model_names:
                try:
                    models = mr.get_models(name=model_name)
                    if models:
                        all_models.extend(models)
                        print(f"   Found {len(models)} versions of {model_name}")
                except:
                    # Model name doesn't exist, continue
                    pass
            
            # Alternative approach: Try to get all models from the project
            # Some Hopsworks versions support this
            try:
                # Try to get models without name parameter (older API)
                import inspect
                sig = inspect.signature(mr.get_models)
                if 'name' not in sig.parameters or sig.parameters['name'].default is not inspect.Parameter.empty:
                    additional_models = mr.get_models()
                    if additional_models:
                        # Filter out duplicates
                        existing_model_ids = {(m.name, m.version) for m in all_models}
                        for model in additional_models:
                            if (model.name, model.version) not in existing_model_ids:
                                all_models.append(model)
            except:
                pass
            
            print(f"📊 Found {len(all_models)} total models in registry")
            
            if len(all_models) == 0:
                print("✅ No models found - registry is already clean")
                print("💡 If you have models with different names, you can:")
                print("   1. Check the Hopsworks UI for model names")
                print("   2. Use --model NAME to delete specific models")
                return True
            
            # Remove duplicates (same name and version)
            unique_models = {}
            for model in all_models:
                key = (model.name, model.version)
                if key not in unique_models:
                    unique_models[key] = model
            
            models = list(unique_models.values())
            print(f"📊 Found {len(models)} unique models after deduplication")
            
            # List all models
            print("\n📋 Models to be deleted:")
            print("-" * 60)
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model.name} (v{model.version}) - {model.description or 'No description'}")
            
            # Final confirmation
            print(f"\n⚠️ About to delete {len(models)} models!")
            final_confirm = input("Type 'YES DELETE' to proceed: ")
            if final_confirm != "YES DELETE":
                print("❌ Operation cancelled")
                return False
            
            # Delete all models
            print(f"\n🗑️ Deleting {len(models)} models...")
            deleted_count = 0
            failed_count = 0
            
            for i, model in enumerate(models, 1):
                try:
                    print(f"🗑️ Deleting {i}/{len(models)}: {model.name} v{model.version}...")
                    model.delete()
                    deleted_count += 1
                    print(f"✅ Deleted: {model.name} v{model.version}")
                except Exception as e:
                    print(f"❌ Failed to delete {model.name} v{model.version}: {e}")
                    failed_count += 1
            
            print(f"\n📊 DELETION SUMMARY:")
            print(f"✅ Successfully deleted: {deleted_count} models")
            if failed_count > 0:
                print(f"❌ Failed to delete: {failed_count} models")
            
            if deleted_count > 0:
                print(f"\n🎉 Model registry cleanup completed!")
                print(f"   Deleted {deleted_count} models at {datetime.now()}")
                return True
            else:
                print(f"❌ No models were deleted")
                return False
                
        except Exception as e:
            print(f"❌ Failed to get models from registry: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error during model deletion: {e}")
        return False

def delete_specific_model(model_name: str = None, model_version: int = None):
    """
    Delete a specific model or all versions of a model
    
    Parameters:
    model_name (str): Name of the model to delete
    model_version (int): Specific version to delete (if None, deletes all versions)
    """
    print(f"🗑️ DELETING SPECIFIC MODEL: {model_name}")
    print("="*60)
    
    try:
        from pipelines.fetch_data import HopsworksIntegration
        
        print("🔗 Connecting to Hopsworks...")
        hops = HopsworksIntegration()
        
        if not hops.enabled:
            print("❌ Hopsworks integration is not enabled")
            return False
        
        mr = hops.mr
        
        if model_version:
            # Delete specific version
            try:
                model = mr.get_model(model_name, version=model_version)
                print(f"🗑️ Deleting {model_name} v{model_version}...")
                model.delete()
                print(f"✅ Deleted {model_name} v{model_version}")
                return True
            except Exception as e:
                print(f"❌ Failed to delete {model_name} v{model_version}: {e}")
                return False
        else:
            # Delete all versions of the model
            try:
                models = mr.get_models(name=model_name)
                print(f"📊 Found {len(models)} versions of {model_name}")
                
                deleted_count = 0
                for model in models:
                    try:
                        print(f"🗑️ Deleting {model.name} v{model.version}...")
                        model.delete()
                        deleted_count += 1
                        print(f"✅ Deleted {model.name} v{model.version}")
                    except Exception as e:
                        print(f"❌ Failed to delete {model.name} v{model.version}: {e}")
                
                print(f"✅ Deleted {deleted_count}/{len(models)} versions of {model_name}")
                return deleted_count > 0
                
            except Exception as e:
                print(f"❌ Failed to get models named {model_name}: {e}")
                return False
        
    except Exception as e:
        print(f"❌ Error during specific model deletion: {e}")
        return False

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete models from Hopsworks model registry")
    parser.add_argument("--all", action="store_true", help="Delete ALL models (requires confirmation)")
    parser.add_argument("--model", type=str, help="Delete specific model (all versions)")
    parser.add_argument("--version", type=int, help="Delete specific version (requires --model)")
    
    args = parser.parse_args()
    
    if args.all:
        success = delete_all_models()
    elif args.model:
        success = delete_specific_model(args.model, args.version)
    else:
        print("🗑️ HOPSWORKS MODEL REGISTRY CLEANUP TOOL")
        print("="*50)
        print("Options:")
        print("  --all              Delete ALL models (with confirmation)")
        print("  --model NAME       Delete all versions of a specific model")
        print("  --model NAME --version N    Delete specific model version")
        print("\nExamples:")
        print("  python scripts/delete_all_models.py --all")
        print("  python scripts/delete_all_models.py --model sklearn_aqi_model")
        print("  python scripts/delete_all_models.py --model dl_aqi_model --version 3")
        return
    
    if success:
        print(f"\n🎉 Model deletion completed successfully!")
    else:
        print(f"\n❌ Model deletion failed or was cancelled")
        sys.exit(1)

if __name__ == "__main__":
    main()
