import os
import config
from feature_factory import FeatureFactory
from feature_store import FeatureStore
from serving_engine import ServingEngine

def run_pipeline():
    # 1. Initialize Components
    factory = FeatureFactory(config.BASE_DATA_DIR)
    store = FeatureStore()
    
    # Clean slate for offline file
    if os.path.exists(config.OFFLINE_FILE):
        os.remove(config.OFFLINE_FILE)
        print(f"Deleted old feature file: {config.OFFLINE_FILE}")
    
    # ---------------------------------------------------------
    # PART A: TRAINING (Ingest & Train on 'dev_seen')
    # ---------------------------------------------------------
    print("\n=== PART A: TRAINING PHASE ===")
    print(f"Ingesting TRAINING data from: {config.JSONL_FILE}")
    train_df = factory.ingest_data(limit=10, override_file=config.JSONL_FILE)
    
    if not train_df.empty:
        # Process Training Data
        train_features = factory.process_batch(train_df)
        
        # Store to Offline
        print("\n--- Storing Training Data ---")
        store.write_offline(train_features)
        
        # Train Model
        print("\n--- Training Model ---")
        engine = ServingEngine(store)
        engine.train_model()
    else:
        print("Error: No training data found.")
        return

    # ---------------------------------------------------------
    # PART B: SERVING / INFERENCE (Ingest & Test on 'dev_unseen')
    # ---------------------------------------------------------
    print("\n=== PART B: INFERENCE PHASE ===")
    print(f"Ingesting TEST data from: {config.test}")
    # Using 'override_file' to load the test set
    test_df = factory.ingest_data(limit=10, override_file=config.test)
    
    if not test_df.empty:
        # Process Test Data (We need features to query the engine, 
        # normally in prod this comes one-by-one, here we batch process for simulation)
        print("\n--- Processing Test Batch ---")
        test_features = factory.process_batch(test_df)
        
        # Update Online Store so serving engine can look them up by ID
        # (In a real system, features would be computed in real-time)
        store.write_online(test_features)
        
        # Serve Predictions
        print("\n--- Serving Predictions on TEST Set ---")
        # Predict on ALL ingested test items
        sample_ids = test_features['post_id'].tolist()
        
        for pid in sample_ids:
            decision = engine.serve_prediction(pid)
            print(f"Post {pid}: Action={decision['action']} (Score={decision.get('score')})")

if __name__ == "__main__":
    run_pipeline()
