from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import sys
from pathlib import Path
import numpy as np

# Thêm path của model vào sys.path
model_path = Path(__file__).parent / 'model' / 'model' / 'text_recognition'
sys.path.insert(0, str(model_path))

from utils.utils import build_model, process_input, translate
from config.config import Config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1. Load Model từ training_info checkpoint
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config
config_path = model_path / 'logs' / 'training_info' / 'config' / 'model_config.yml'
checkpoint_path = model_path / 'logs' / 'training_info' / 'checkpoints' / 'transformerocr--val_loss=0.64--wer=0.20--last.pth'

print(f"Loading model from {checkpoint_path}...")
try:
    config = Config.load_config_from_file(str(config_path))
    config['device'] = device
    config['predictor']['beamsearch'] = False
    
    model, vocab = build_model(config)
    
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}!")
    print(f"Vocab size: {len(vocab)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vocab = None
    config = None

# --------------------------------------------------
# 2. Hàm đọc ảnh và chạy model
# --------------------------------------------------
def predict_text(pil_img):
    if model is None:
        return "MODEL_NOT_LOADED", 0.0

    try:
        # Process image
        img_tensor = process_input(
            pil_img,
            config['dataset']['image_height'],
            config['dataset']['image_min_width'],
            config['dataset']['image_max_width']
        )
        
        img_tensor = img_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            s, confidence = translate(img_tensor, model)
            s = s[0].tolist()
            confidence = confidence[0]
        
        # Decode to text
        text = vocab.decode(s)
        
        return text, float(confidence)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"ERROR: {str(e)}", 0.0


# --------------------------------------------------
# 3. API xử lý nhiều ảnh
# --------------------------------------------------
@app.post("/api/ocr")
async def run_ocr(images: list[UploadFile] = File(...)):
    results = []

    for img in images:
        img_bytes = await img.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        text, confidence = predict_text(pil_img)

        results.append({
            "filename": img.filename,
            "ocr_text": text,
            "confidence": confidence
        })
        print(f"Processed {img.filename}: {text} (result: {results})")
    return { "results": results }
