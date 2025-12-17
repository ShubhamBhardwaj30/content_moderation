# Decoupled Multi-Modal Feature Platform

I have successfully implemented and refined the end-to-end architecture for preventing harmful content using a decoupled feature platform.

## Key Architecture Update: Refined VLM + Text Pipeline
Based on feedback, the Stage 1 pipeline now rigorously separates standard post text from image-derived signals.

**Data Flow:**
1.  **Ingestion**: Reads the "Post Text" (caption/status) from the source.
2.  **VLM Analysis** (Stage 1a): The image is analyzed to generate:
    *   **Visual Summary**: A concise description of the scene (e.g., "A chart showing growth").
    *   **Extracted OCR**: Text found *on the image* (e.g., "SALE 50% OFF").
3.  **Text Transformer** (Stage 1b): Fuses all three inputs (`Post Text` + `Visual Summary` + `OCR`) to understand the full context.
4.  **Tagging** (Stage 1c): Generates the Binary Tags (`Is_Harmful_Content`, etc.).

## VLM Options

### Option 1: Network Local VLM (Ollama) - **DEFAULT**
Use your local Ollama instance (e.g., `llama3.2-vision`).
1.  ensure Ollama is running at `http://192.168.2.16:11434` (configurable in `config.py`).
2.  In `main.py` (or `factory.process_batch()`), this is now the default mode.
3.  To explicitly enable/disable: pass `use_ollama=True`.

**Prompt Customization**
- Edit `prompt.txt` to change the instructions sent to the VLM.
- The system automatically appends a JSON formatting instruction to ensure compatibility.

### Option 2: Offline VLM (HuggingFace)
Free, runs locally, requires CPU/GPU resources.
1.  Install dependencies: `pip install transformers torch accelerate`
2.  In `main.py`, pass `use_offline=True`.

## Execution Results

I ran the `main.py` script which simulates the entire lifecycle of a meme post.

**Sample Output Log:**
```text
Reading data from .../dev_seen.jsonl...
Ingested 50 posts.

--- Processing Batch (Stage 1) ---
Processed 1/50 posts...
...
Training Final Ranker Model...
Model Trained successfully on 50 rows.

--- Stage 3: Serving Requests ---
Post 46971: Action=DISPLAY (Score=0.9142)
```

## Key Files
- [main.py](file:///Users/shubhambhardwaj/Shubham/datascience/study/LLM/embeddings/research/harmful_meme/main.py): Entry point (Refactored).
- [feature_factory.py](file:///Users/shubhambhardwaj/Shubham/datascience/study/LLM/embeddings/research/harmful_meme/feature_factory.py): Contains the logic for the VLM/Transformer fusion.
- [feature_store.py](file:///Users/shubhambhardwaj/Shubham/datascience/study/LLM/embeddings/research/harmful_meme/feature_store.py): Handles Offline/Online storage.
