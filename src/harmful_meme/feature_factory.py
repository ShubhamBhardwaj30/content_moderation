import os
import json
import pandas as pd
import random
import base64
import requests
from typing import List, Dict
from PIL import Image
import config


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
        

        # Load Prompts
        self.base_prompt = self._load_prompt(config.PROMPT_FILE)
        self.llm_prompt = self._load_prompt(config.LLM_PROMPT_FILE)

    def _load_prompt(self, filename: str) -> str:
        """Loads the base prompt from the external file."""
        prompt_path = os.path.join(self.data_dir, "../../../", filename) 
        # Attempt relative to cwd/execution point first, then fallback
        candidates = [
            filename, 
            os.path.join(os.path.dirname(__file__), filename)
        ]
        
        for p in candidates:
            if os.path.exists(p):
                print(f"Loading external prompt from: {p}")
                with open(p, 'r') as f:
                    return f.read().strip()
        
        print(f"Warning: {filename} not found. Using default.")
        return """
        Analyze this image for a content moderation system.
        1. Provide a concise visual summary (what is happening?).
        2. Extract ALL text visible in the image exactly as written (OCR).
        3. Make sure to mention if the post is sarcastic or not.
        4. if it is sarcastic, then provide a reason for it and specify if this is harmful content or not.
        """


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


    def _call_vlm(self, image_path: str) -> Dict[str, str]:
        """Network Local VLM implementation using Ollama API."""
        base64_image = self._encode_image(image_path)
        
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": self.base_prompt,
            "images": [base64_image],
            "stream": False,
            "format": "json" # Force JSON output
        }

        print(f"Calling Ollama ({config.OLLAMA_URL})...")
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_payload = result.get('response', '{}')
        
        # Parse JSON from Ollama response
        try:
            parsed = json.loads(response_payload)
            return {
                "visual_summary": parsed.get("visual_summary", "Summary not found"),
                "ocr_text": parsed.get("ocr_text", ""),
                "is_sarcastic": parsed.get("is_sarcastic", False),
                "is_sarcastic_reason": parsed.get("is_sarcastic_reason", ""),
                "is_harmful": parsed.get("is_harmful", False),
                "is_harmful_reason": parsed.get("is_harmful_reason", ""),
                "is_offensive": parsed.get("is_offensive", False),
                "is_offensive_reason": parsed.get("is_offensive_reason", ""),
                "is_violent": parsed.get("is_violent", False),
                "is_violent_reason": parsed.get("is_violent_reason", ""),
                "is_sexual": parsed.get("is_sexual", False),
                "is_sexual_reason": parsed.get("is_sexual_reason", ""),
                "is_explicit": parsed.get("is_explicit", False),
                "is_explicit_reason": parsed.get("is_explicit_reason", ""),
                "is_terrorist": parsed.get("is_terrorist", False),
                "is_terrorist_reason": parsed.get("is_terrorist_reason", ""),
                "is_extremist": parsed.get("is_extremist", False),
                "is_extremist_reason": parsed.get("is_extremist_reason", ""),
                "is_child_exploitation": parsed.get("is_child_exploitation", False),
                "is_child_exploitation_reason": parsed.get("is_child_exploitation_reason", ""),
                "is_hate_speech": parsed.get("is_hate_speech", False),
                "is_hate_speech_reason": parsed.get("is_hate_speech_reason", ""),
                "is_spam": parsed.get("is_spam", False),
                "is_spam_reason": parsed.get("is_spam_reason", ""),
                "is_racist_or_racial_slur": parsed.get("is_racist_or_racial_slur", False),
                "is_racist_or_racial_slur_reason": parsed.get("is_racist_or_racial_slur_reason", ""),
            }
        except json.JSONDecodeError:
            # Fallback if model didn't return strict JSON
            return {
                "visual_summary": response_text[:200], # Truncate raw text
                "ocr_text": "[Raw Output: JSON Parse Failed]",
                "is_sarcastic": False,
                "is_sarcastic_reason": "",  
                "is_harmful": False,
                "is_harmful_reason": "", 
                "is_offensive": False,
                "is_offensive_reason": "", 
                "is_violent": False,
                "is_violent_reason": "", 
                "is_sexual": False,
                "is_sexual_reason": "", 
                "is_terrorist": False,
                "is_terrorist_reason": "", 
                "is_extremist": False,
                "is_extremist_reason": "", 
                "is_child_exploitation": False,
                "is_child_exploitation_reason": "", 
                "is_hate_speech": False,
                "is_hate_speech_reason": "", 
                "is_spam": False,
                "is_spam_reason": "", 
                "is_racist_or_racial_slur": False,
                "is_racist_or_racial_slur_reason": "",
            }

    def generate_caption(self, image_path: str) -> Dict[str, str]:
        """
        Step 2: VLM Analysis (Visual + OCR).
        """
        return self._call_vlm(image_path)


    def summarize_text(self, post_text: str, vlm_output: Dict[str, str]) -> List[str]:
        """Step 3: Text Transformer (using Ollama)."""
        summary = vlm_output.get("visual_summary", "")
        ocr_text = vlm_output.get("ocr_text", "")
        
        # Format the prompt
        prompt = self.llm_prompt.format(
            post_text=post_text,
            visual_summary=summary,
            ocr_text=ocr_text
        )

        payload = {
            "model": config.OLLAMA_TEXT_MODEL,
            "prompt": prompt,
            "stream": False
        }

        print(f"Calling Ollama (Text) for summary...")
        try:
             response = requests.post(config.OLLAMA_URL, json=payload, timeout=30)
             response.raise_for_status()
             result = response.json()
             response_text = result.get('response', '')
             
             # Clean up the response to get a list
             keywords = [k.strip() for k in response_text.split(',')]
             return keywords
             
        except Exception as e:
            print(f"Error calling Ollama for text summary: {e}")
            # Fallback to simple keyword extraction
            combined_context = f"Post Text: {post_text}. Image shows: {summary}. Image text says: {ocr_text}"
            return list(set([w.lower() for w in combined_context.split() if len(w) > 3]))

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

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        print("\n--- Processing Batch (Stage 1) ---")
        for i, row in df.iterrows():
            post_id = row['id']
            post_text = row.get('text', '')
            img_path = row['img_abs_path']
            
            # 1. VLM Analysis
            try:
                # use_ollama is default True now
                vlm_output = self.generate_caption(img_path)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                vlm_output = {
                    "visual_summary": "ERROR", 
                    "ocr_text": "ERROR",
                    "is_sarcastic": False,
                    "is_sarcastic_reason": "ERROR",
                    "is_harmful": False,
                    "is_harmful_reason": "ERROR",
                    "is_offensive": False,
                    "is_offensive_reason": "ERROR",
                    "is_violent": False,
                    "is_violent_reason": "ERROR",
                    "is_sexual": False,
                    "is_sexual_reason": "ERROR",
                    "is_explicit": False,
                    "is_explicit_reason": "ERROR",
                    "is_terrorist": False,
                    "is_terrorist_reason": "ERROR",
                    "is_extremist": False,
                    "is_extremist_reason": "ERROR",
                    "is_child_exploitation": False,
                    "is_child_exploitation_reason": "ERROR",
                    "is_hate_speech": False,
                    "is_hate_speech_reason": "ERROR",
                    "is_spam": False,
                    "is_spam_reason": "ERROR",
                    "is_racist_or_racial_slur": False,
                    "is_racist_or_racial_slur_reason": "ERROR"
                }

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
