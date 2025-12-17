import os
import random
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
import config
from feature_store import FeatureStore

class ServingEngine:
    """The lightweight serving layer."""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.model = None

    def train_model(self):
        """Step 9: Train simple model on Offline Data."""
        print("\nTraining Final Ranker Model...")
        if not os.path.exists(config.OFFLINE_FILE):
            print("No training data found.")
            return

        df = pd.read_csv(config.OFFLINE_FILE)
        
        # X: Binary Tags
        feature_cols = [c for c in df.columns if c.startswith("Is_")]
        X = df[feature_cols]
        
        # Y: Synthetic Label (0=Block/Bad, 1=Display/Good)
        y = []
        for _, row in df.iterrows():
            if row.get('Is_Harmful_Content', 0) == 1:
                label = 0 if random.random() > 0.1 else 1
            else:
                label = 1 if random.random() > 0.1 else 0
            y.append(label)
        
        self.model = LogisticRegression()
        try:
             self.model.fit(X, y)
             print(f"Model Trained successfully on {len(df)} rows.")
        except Exception as e:
             print(f"Model training failed (likely not enough class variance): {e}")

    def serve_prediction(self, post_id: str) -> Dict[str, Any]:
        """Step 10: Enforcement Endpoint."""
        # 1. Fetch
        features = self.feature_store.get_online_features(post_id)
        if not features:
            return {"post_id": post_id, "action": "ERROR_NOT_FOUND"}

        # 2. Predict
        if self.model:
            feature_vector = [list(features.values())] 
            prob = self.model.predict_proba(feature_vector)[0][1] 
        else:
            # Fallback
            prob = 0.0 if features.get('Is_Harmful_Content') == 1 else 1.0

        # 3. Decide
        action = "DISPLAY"
        if prob < 0.2:
            action = "BLOCK"
        elif prob < 0.5:
            action = "DEMOTE"
            
        return {
            "post_id": post_id,
            "score": round(prob, 4),
            "action": action,
        }
