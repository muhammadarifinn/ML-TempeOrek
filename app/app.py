import json

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

# ===== LOAD MODEL (file berada di folder yang sama dengan app.py) =====
ART = "."
meta = json.load(open(f"{ART}/model_meta.json"))
CLASSES = meta["classes"]
IMG_SIZE = meta["img_size"]
MEAN, STD = meta["mean"], meta["std"]

# Pemetaan kode -> nama tumor. WAJIB dicek dengan dokumentasi dataset!
# Jika CLASSES sudah berupa nama (glioma, dst), biarkan saja: kunci yang tidak
# ditemukan akan dipakai apa adanya.
NAMA_TAMPIL = {"1": "Meningioma", "2": "Glioma", "3": "Pituitary"}
CLASSES_DISPLAY = [NAMA_TAMPIL.get(c, c) for c in CLASSES]

device = "cpu"
model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
model.load_state_dict(torch.load(f"{ART}/best_model.pth", map_location=device))
model.to(device).eval()

# ===== PREPROCESSING (harus sama dengan eval_tf di Pipeline 1) =====
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def standardize(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def looks_grayscale(img: Image.Image, tol=12) -> bool:
    a = np.asarray(img).astype(int)
    return (np.abs(a[..., 0] - a[..., 1]).mean() < tol and
            np.abs(a[..., 1] - a[..., 2]).mean() < tol)


def validate_image(img: Image.Image):
    w, h = img.size
    if min(w, h) < 64:
        return False, "Gambar terlalu kecil (minimal 64 piksel)."
    if not looks_grayscale(img):
        return False, ("Gambar berwarna terdeteksi. MRI otak umumnya grayscale, "
                       "jadi gambar ini kemungkinan bukan MRI.")
    return True, "OK"


# ===== INFERENSI + GRAD-CAM =====
def predict_with_cam(img: Image.Image):
    store = {}

    def fwd_hook(m, i, o):
        store["feat"] = o
        o.register_hook(lambda g: store.__setitem__("grad", g))

    h = model.layer4[-1].register_forward_hook(fwd_hook)
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.enable_grad():
        model.zero_grad()
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        cls = int(probs.argmax())
        out[0, cls].backward()
    h.remove()

    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["feat"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear")[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    return probs.detach().cpu().numpy(), cam


def overlay(img: Image.Image, cam, alpha=0.4):
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(float)
    return Image.fromarray(((1 - alpha) * base + alpha * heat).astype(np.uint8))


def normalized_entropy(p):
    p = np.clip(p, 1e-8, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def analyze(img: Image.Image, conf_threshold=0.80):
    img = standardize(img)
    ok, msg = validate_image(img)
    if not ok:
        return {"status": "ditolak", "message": msg}

    probs, cam = predict_with_cam(img)
    top = int(probs.argmax())
    conf = float(probs[top])
    ent = normalized_entropy(probs)

    if conf < conf_threshold or ent > 0.6:
        status = "tidak_yakin"
        message = (f"Model tidak yakin (confidence {conf:.0%}). Pastikan gambar adalah "
                   "MRI otak yang jelas, atau konsultasikan dengan tenaga medis.")
    else:
        status = "ok"
        message = f"Prediksi: **{CLASSES_DISPLAY[top]}** (confidence {conf:.0%})"

    return {
        "status": status,
        "message": message,
        "probs": {CLASSES_DISPLAY[i]: float(probs[i]) for i in range(len(CLASSES))},
        "cam_image": overlay(img, cam),
    }


# ===== ANTARMUKA =====
DISCLAIMER = ("⚠️ **Disclaimer:** Sistem ini adalah proyek edukasi/percobaan dan "
              "**bukan alat diagnosis medis**. Hasil tidak boleh dipakai untuk "
              "keputusan klinis. Konsultasikan dengan dokter.")


def ui_predict(image, threshold):
    if image is None:
        return {}, "Silakan upload gambar terlebih dahulu.", None
    r = analyze(image, conf_threshold=threshold)
    if r["status"] == "ditolak":
        return {}, f"❌ {r['message']}", None
    prefix = "✅ " if r["status"] == "ok" else "⚠️ "
    return r["probs"], prefix + r["message"], r["cam_image"]


with gr.Blocks(title="Klasifikasi Tumor Otak (Demo Edukasi)") as demo:
    gr.Markdown("# 🧠 Klasifikasi Citra MRI Tumor Otak")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload MRI otak (PNG/JPG)")
            thr = gr.Slider(0.5, 0.99, value=0.80, step=0.01,
                            label="Ambang confidence minimum")
            btn = gr.Button("Prediksi", variant="primary")
        with gr.Column():
            msg = gr.Markdown()
            lbl = gr.Label(num_top_classes=len(CLASSES), label="Probabilitas tiap kelas")
            cam = gr.Image(label="Grad-CAM (area yang diperhatikan model)")
    btn.click(ui_predict, [inp, thr], [lbl, msg, cam])

if __name__ == "__main__":
    demo.launch()
