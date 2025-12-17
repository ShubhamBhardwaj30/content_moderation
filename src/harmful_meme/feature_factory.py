import os
import json
import pandas as pd
import random
import base64
import requests
from typing import List, Dict
from PIL import Image
import config

# Optional: Import Transformers for Offline VLM
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class FeatureFactory:
    """
    Stage 1: The Granular Feature Factory.
    Converts raw media (Image/Text) into standardized Binary Tags.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = config.BASE_DATA_DIR
        self.img_dir = os.path.join(self.data_dir, config.IMG_DIR)
        self.input_file = os.path.join(self.data_dir, config.JSONL_FILE)
        self.policy_thresholds = config.POLICY_THRESHOLDS
        
        # Initialize Offline VLM (Lazy load)
        self.offline_pipeline = None

        # Load Prompt
        self.base_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Loads the base prompt from the external file."""
        prompt_path = os.path.join(self.data_dir, "../../../", config.PROMPT_FILE) 
        # Attempt relative to cwd/execution point first, then fallback
        candidates = [
            config.PROMPT_FILE, 
            os.path.join(os.path.dirname(__file__), config.PROMPT_FILE)
        ]
        
        for p in candidates:
            if os.path.exists(p):
                print(f"Loading external prompt from: {p}")
                with open(p, 'r') as f:
                    return f.read().strip()
        
        print("Warning: prompt.txt not found. Using default.")
        return """
        Analyze this image for a content moderation system.
        1. Provide a concise visual summary (what is happening?).
        2. Extract ALL text visible in the image exactly as written (OCR).
        3. Make sure to mention if the post is sarcastic or not.
        4. if it is sarcastic, then provide a reason for it and specify if this is harmful content or not.
        """

    def _init_offline_model(self):
        """Loads a lightweight offline VLM (e.g., BLIP)."""
        if not pipeline:
            raise ImportError("Error: 'transformers' library not found. Install it for offline VLM.")

        print("Loading Offline VLM (this may take a moment)...")
        self.offline_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print("Offline VLM loaded.")

    def ingest_data(self, limit: int = None, override_file: str = None) -> pd.DataFrame:
        """Step 1: Ingest & Encode."""
        data = []
        
        # Determine strict target file
        if override_file:
            # If absolute, use as is. If relative, join with data_dir
            if os.path.isabs(override_file):
                 target_file = override_file
            else:
                 target_file = os.path.join(self.data_dir, override_file)
        else:
            target_file = self.input_file

        print(f"Reading data from {target_file}...")
        try:
            with open(target_file, 'r') as f:
                for idx, line in enumerate(f):
                    if limit and idx >= limit:
                        break
                    entry = json.loads(line)
                    abs_path = os.path.join(self.data_dir, entry['img'])
                    
                    if os.path.exists(abs_path):
                        entry['img_abs_path'] = abs_path
                        data.append(entry)
                    else:
                         # Skip missing images so we don't crash later
                         pass
        except FileNotFoundError:
             print(f"Error: File not found at {target_file}")
             return pd.DataFrame()

        df = pd.DataFrame(data)
        print(f"Ingested {len(df)} posts.")
        return df

    def _encode_image(self, image_path: str) -> str:
        """Helper to encode image to base64 for API transmission."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_offline_vlm(self, image_path: str) -> Dict[str, str]:
        """Offline implementation using HuggingFace Transformers."""
        if not self.offline_pipeline:
            self._init_offline_model()
            
        image = Image.open(image_path)
        # caption only for BLIP
        results = self.offline_pipeline(image)
        caption = results[0]['generated_text']
        ocr_text = "[Offline OCR Not Implemented in BLIP]" 
        return {"visual_summary": caption, "ocr_text": ocr_text}

    def _call_ollama_vlm(self, image_path: str) -> Dict[str, str]:
        """Network Local VLM implementation using Ollama API."""
        base64_image = self._encode_image(image_path)
        
        # Combine User Prompt + System Layout Instruction
        # We append schema instruction strictly to ensure VLM follows
        full_prompt = f"""
        {self.base_prompt}
        
        CRITICAL INSTRUCTION: You MUST return the result in valid JSON format with exactly these keys:
        - "visual_summary": [Your description]
        - "ocr_text": [Extracted text]
        - "is_sarcastic": [True/False]
        - "reason": [Reason for sarcasm]
        """
        
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": full_prompt,
            "images": [base64_image],
            "stream": False,
            "format": "json" # Force JSON output
        }
        
        # print(f"Calling Ollama ({config.OLLAMA_MODEL})...")
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '{}')
        
        # Parse JSON from Ollama response
        try:
            parsed = json.loads(response_text)
            print(parsed)
            return {
                "visual_summary": parsed.get("visual_summary", "Summary not found"),
                "ocr_text": parsed.get("ocr_text", "")
            }
        except json.JSONDecodeError:
            # Fallback if model didn't return strict JSON
            return {
                "visual_summary": response_text[:200], # Trucate raw text
                "ocr_text": "[Raw Output: JSON Parse Failed]"
            }

    def generate_caption(self, image_path: str, use_offline: bool = False, use_ollama: bool = True) -> Dict[str, str]:
        """
        Step 2: VLM Analysis (Visual + OCR).
        """
        # Prioritize Offline if requested
        if use_offline:
            return self._call_offline_vlm(image_path)
        
        # Default to Ollama
        return self._call_ollama_vlm(image_path)


    def summarize_text(self, post_text: str, vlm_output: Dict[str, str]) -> List[str]:
        """Step 3: Text Transformer."""
        summary = vlm_output.get("visual_summary", "")
        ocr_text = vlm_output.get("ocr_text", "")
        combined_context = f"Post Text: {post_text}. Image shows: {summary}. Image text says: {ocr_text}"
        keywords = list(set([w.lower() for w in combined_context.split() if len(w) > 3]))
        return keywords

    def classify_to_scores(self, keywords: List[str]) -> Dict[str, float]:
        """Step 4: Tag Generation."""
        triggers = ["hate", "kill", "attack", "stupid"]
        base_score = random.random() * 0.5 
        if any(t in keywords for t in triggers):
            base_score += 0.4
        scores = {
            "Harmful_Content": min(1.0, base_score + random.uniform(-0.1, 0.1)),
            "Political_Content": random.random(),
            "Spam": random.random() * 0.3,
            "Copyright_Infringement": random.random() * 0.1
        }
        return scores

    def apply_policy(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Step 5: Policy Threshold."""
        tags = {}
        for category, score in scores.items():
            threshold = self.policy_thresholds.get(category, 0.5)
            tags[f"Is_{category}"] = 1 if score >= threshold else 0
        return tags

    def process_batch(self, df: pd.DataFrame, use_offline: bool = False, use_ollama: bool = True) -> pd.DataFrame:
        results = []
        print("\n--- Processing Batch (Stage 1) ---")
        for i, row in df.iterrows():
            post_id = row['id']
            post_text = row.get('text', '')
            img_path = row['img_abs_path']
            
            # 1. VLM Analysis
            try:
                # use_ollama is default True now
                vlm_output = self.generate_caption(img_path, use_offline=use_offline, use_ollama=use_ollama)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                vlm_output = {"visual_summary": "ERROR", "ocr_text": "ERROR"}

            # 2. Text Transformer
            keywords = self.summarize_text(post_text, vlm_output)
            
            # 3. Classify
            scores = self.classify_to_scores(keywords)
            
            # 4. Binary Tags
            binary_tags = self.apply_policy(scores)
            
            result_row = {
                'post_id': post_id,
                'post_text': post_text,
                'keywords': keywords, 
                **vlm_output,
                **scores,
                **binary_tags
            }
            results.append(result_row)
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(df)} posts...")
                
        return pd.DataFrame(results)
