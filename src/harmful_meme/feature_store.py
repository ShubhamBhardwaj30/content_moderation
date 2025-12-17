import os
import pandas as pd
from typing import Dict
import config

class FeatureStore:
    """Manages Offline and Online stores."""
    
    def __init__(self):
        self.online_store = {}
        self.offline_file = config.OFFLINE_FILE

    def write_offline(self, df: pd.DataFrame):
        """Step 6: Offline Storage (Append to CSV)."""
        if os.path.exists(self.offline_file):
            df.to_csv(self.offline_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.offline_file, mode='w', header=True, index=False)
        print(f"Written {len(df)} records to Offline Store ({self.offline_file})")

    def write_online(self, df: pd.DataFrame):
        """Step 7: Online Storage (Update Mock Redis)."""
        print("Updating Online Store (Mock Redis)...")
        feature_cols = [c for c in df.columns if c.startswith("Is_")]
        
        for _, row in df.iterrows():
            post_id = str(row['post_id'])
            features = row[feature_cols].to_dict()
            self.online_store[post_id] = features
        
        print(f"Online Store now has {len(self.online_store)} keys.")

    def get_online_features(self, post_id: str) -> Dict[str, int]:
        """Step 8: Feature Retrieval."""
        return self.online_store.get(str(post_id), None)
