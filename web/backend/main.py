from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1. Load CHECKPOINT từ file .ckpt (bạn sẽ wget về)
# --------------------------------------------------

CHECKPOINT_PATH = "./model.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Thay bằng kiến trúc model thật của bạn
# ví dụ:
# model = MyOCRModel(...)
# checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
# model.load_state_dict(checkpoint['state_dict'])
# model.to(device)
# model.eval()

model = None
print("⚠️ Chưa load model. Hãy thay ModelClass và state_dict.")

# --------------------------------------------------
# 2. Hàm đọc ảnh và chạy model
# --------------------------------------------------
def predict_text(pil_img):
    if model is None:
        return "MODEL_NOT_LOADED"

    # TODO: thay bằng inference thật
    # ví dụ:
    # tensor = transform(pil_img).unsqueeze(0).to(device)
    # output = model(tensor)
    # text = decode(output)
    # return text

    return "dummy_text"


# --------------------------------------------------
# 3. API xử lý nhiều ảnh
# --------------------------------------------------
@app.post("/api/ocr")
async def run_ocr(images: list[UploadFile] = File(...)):
    results = []

    for img in images:
        img_bytes = await img.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        text = predict_text(pil_img)

        results.append({
            "filename": img.filename,
            "ocr_text": text
        })

    return { "results": results }
