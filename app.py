# =========================================================
# IMPORT LIBRARY
# =========================================================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Penyakit Mata Berbasis Citra Retina",
    layout="wide"
)

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# CLASS
# =========================================================
class_names = [
    'cataract',
    'diabetic_retinopathy',
    'glaucoma',
    'normal'
]

# =========================================================
# CONFIDENCE THRESHOLD
# =========================================================
THRESHOLD = 0.70

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #070B14;
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #131722;
    border: 1px solid #2A2F3A;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stSelectbox"] {
    background-color: #131722;
    border-radius: 10px;
}

.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background-color: #2563EB;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

.result-box {
    background-color: #131722;
    border: 1px solid #2A2F3A;
    border-radius: 15px;
    padding: 20px;
}

.model-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
}

.pred-text {
    font-size: 18px;
    color: #D1D5DB;
}

.conf-text {
    font-size: 24px;
    color: #22C55E;
    font-weight: bold;
}

.warning-text {
    font-size: 20px;
    color: #FACC15;
    font-weight: bold;
}

hr {
    border: 1px solid #222831;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>
Klasifikasi Penyakit Retina
</h1>

<p style='
text-align:center;
font-size:18px;
color:#D1D5DB;
max-width:900px;
margin:auto;
line-height:1.8;
'>

Sistem ini membandingkan hasil klasifikasi penyakit retina 
menggunakan <b>SqueezeNet</b> dan <b>ShuffleNet</b>
pada berbagai skenario pelatihan.

</p>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# VALIDASI RETINA
# =========================================================
def is_retina_image(img_pil):

    # =====================================================
    # RESIZE
    # =====================================================
    img = np.array(img_pil.resize((224, 224)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # =====================================================
    # RGB CHECK
    # Retina biasanya dominan merah/oranye
    # =====================================================
    r_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    b_mean = np.mean(img[:, :, 2])

    color_score = (
        r_mean > 50 and
        r_mean > g_mean * 0.9
    )

    # =====================================================
    # DETEKSI LINGKARAN
    # =====================================================
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=70,
        maxRadius=110
    )

    if circles is None:
        return False

    circles = np.round(circles[0, :]).astype("int")

    x, y, r = circles[0]

    # =====================================================
    # MASK FUNDUS
    # =====================================================
    mask = np.zeros(gray.shape, dtype=np.uint8)

    cv2.circle(mask, (x, y), r, 255, -1)

    inside_pixels = gray[mask == 255]
    outside_pixels = gray[mask == 0]

    inside_mean = np.mean(inside_pixels)
    outside_mean = np.mean(outside_pixels)

    # =====================================================
    # BACKGROUND GELAP
    # Retina luar biasanya hitam
    # =====================================================
    background_score = (
        outside_mean < 70 and
        inside_mean > 40
    )

    # =====================================================
    # CLAHE
    # =====================================================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    # =====================================================
    # DETEKSI VESSEL
    # =====================================================
    vessels = cv2.Canny(enhanced, 30, 100)

    vessel_density = np.mean(vessels > 0)

    # Retina punya vessel kecil bercabang
    vessel_score = (
        vessel_density > 0.07 and
        vessel_density < 0.35
    )

    # =====================================================
    # KONTRAS
    # =====================================================
    contrast = np.std(gray)

    contrast_score = contrast > 25

    # =====================================================
    # BORDER DARKNESS
    # Retina biasanya punya border gelap
    # =====================================================
    h, w = gray.shape

    border = np.concatenate([
        gray[:15, :].flatten(),
        gray[-15:, :].flatten(),
        gray[:, :15].flatten(),
        gray[:, -15:].flatten()
    ])

    border_dark_ratio = np.mean(border < 60)

    border_score = border_dark_ratio > 0.20

    # =====================================================
    # FINAL VALIDATION
    # =====================================================
    final_score = (
        color_score and
        background_score and
        vessel_score and
        contrast_score and
        border_score
    )

    return final_score

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_models(training_option):

    # =====================================================
    # SQUEEZENET
    # =====================================================
    squeezenet = models.squeezenet1_1(pretrained=True)

    squeezenet.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, 4, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    )

    # =====================================================
    # SHUFFLENET
    # =====================================================
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

    in_features = shufflenet.fc.in_features

    shufflenet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 4)
    )

    # =====================================================
    # PATH MODEL
    # =====================================================
    if training_option == "Fixed Feature":

        squeeze_path = "models/best_squeezenet_fixed_feature_non_aug.pth"
        shuffle_path = "models/best_shufflenet_fixed_feature_non_aug.pth"

    else:

        squeeze_path = "models/best_squeezenet_partial_finetuning_non_aug.pth"
        shuffle_path = "models/best_shufflenet_partial_finetuning_non_aug.pth"

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================
    squeezenet.load_state_dict(
        torch.load(squeeze_path, map_location=device)
    )

    shufflenet.load_state_dict(
        torch.load(shuffle_path, map_location=device)
    )

    squeezenet.to(device)
    shufflenet.to(device)

    squeezenet.eval()
    shufflenet.eval()

    return squeezenet, shufflenet

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## Pilih Skenario Pelatihan")

training_option = st.selectbox(
    "",
    [
        "Fixed Feature",
        "Partial Fine-Tuning"
    ]
)

uploaded_file = st.file_uploader(
    "Upload gambar retina",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:

    # =====================================================
    # LOAD IMAGE
    # =====================================================
    img_pil = Image.open(uploaded_file)

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # =====================================================
    # VALIDASI RETINA
    # =====================================================
    if not is_retina_image(img_pil):

        st.error("""
        ❌ Gambar yang diupload bukan citra fundus retina yang valid.

        Sistem hanya menerima gambar retina/fundus mata.
        """)

        st.stop()

    # =====================================================
    # LOAD MODEL
    # =====================================================
    with st.spinner("Memproses gambar..."):

        squeezenet, shufflenet = load_models(training_option)

        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        img = np.array(
            img_pil.resize((224, 224))
        ).astype(np.float32) / 255.0

    # =====================================================
    # PREDICT FUNCTION
    # =====================================================
    def predict_model(model):

        with torch.no_grad():

            output = model(input_tensor)

            probs = torch.softmax(output, dim=1)[0]

            pred_class = torch.argmax(probs).item()

            confidence = probs[pred_class].item()

        if confidence < THRESHOLD:

            pred_name = "Tidak dikenali"

        else:

            pred_name = class_names[pred_class]

        return probs, pred_name, confidence, pred_class

    # =====================================================
    # PREDICTION
    # =====================================================
    sq_probs, sq_pred, sq_conf, sq_class = predict_model(squeezenet)

    sh_probs, sh_pred, sh_conf, sh_class = predict_model(shufflenet)

    # =====================================================
    # GRADCAM FUNCTION
    # =====================================================
    def generate_gradcam(model, target_layers, pred_class):

        for param in model.parameters():
            param.requires_grad = True

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)]
        )[0]

        cam_image = show_cam_on_image(
            img,
            grayscale_cam,
            use_rgb=True
        )

        return cam_image

    # =====================================================
    # GENERATE CAM
    # =====================================================
    sq_cam = generate_gradcam(
        squeezenet,
        [squeezenet.features[-1]],
        sq_class
    )

    sh_cam = generate_gradcam(
        shufflenet,
        [shufflenet.stage4[-1]],
        sh_class
    )

    # =====================================================
    # INPUT IMAGE
    # =====================================================
    st.markdown("---")

    st.markdown("## Gambar Input")

    center_col = st.columns([1,2,1])

    with center_col[1]:

        st.image(
            img_pil,
            width=300
        )

    # =====================================================
    # RESULT
    # =====================================================
    st.markdown("---")

    st.markdown(f"""
    ## Hasil Prediksi ({training_option})
    """)

    col1, col2 = st.columns(2)

    # =====================================================
    # SQUEEZENET
    # =====================================================
    with col1:

        st.markdown(f"""
        <div class='result-box'>

        <div class='model-title'>
        SqueezeNet
        </div>

        <br>

        <div class='pred-text'>
        Prediksi: <b>{sq_pred}</b>
        </div>

        <br>

        <div class='conf-text'>
        Confidence: {sq_conf*100:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Original vs Grad-CAM")

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:

            st.image(
                img,
                caption="Original"
            )

        with cam_col2:

            st.image(
                sq_cam,
                caption="Grad-CAM"
            )

        st.markdown("### Probabilitas Kelas")

        sq_df = pd.DataFrame({
            "Class": class_names,
            "Probability": sq_probs.cpu().numpy()
        })

        st.bar_chart(
            sq_df.set_index("Class")
        )

    # =====================================================
    # SHUFFLENET
    # =====================================================
    with col2:

        st.markdown(f"""
        <div class='result-box'>

        <div class='model-title'>
        ShuffleNet
        </div>

        <br>

        <div class='pred-text'>
        Prediksi: <b>{sh_pred}</b>
        </div>

        <br>

        <div class='conf-text'>
        Confidence: {sh_conf*100:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Original vs Grad-CAM")

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:

            st.image(
                img,
                caption="Original"
            )

        with cam_col2:

            st.image(
                sh_cam,
                caption="Grad-CAM"
            )

        st.markdown("### Probabilitas Kelas")

        sh_df = pd.DataFrame({
            "Class": class_names,
            "Probability": sh_probs.cpu().numpy()
        })

        st.bar_chart(
            sh_df.set_index("Class")
        )

# =========================================================
# EMPTY STATE
# =========================================================
else:

    st.info("""
    ℹ️ Upload citra fundus retina untuk memulai prediksi.

    Sistem hanya menerima gambar retina/fundus mata.
    """)
