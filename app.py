# ======================
# IMPORT
# ======================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Klasifikasi Penyakit Retina",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

THRESHOLD = 0.70

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1,h2,h3,h4,h5,h6,p,label,div {
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #1A1D24;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #2E3440;
}

[data-testid="stSelectbox"] {
    background-color: #1A1D24;
    border-radius: 10px;
}

.stButton > button {
    width: 100%;
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #4338CA;
}

.result-card {
    background-color: #1A1D24;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #2E3440;
}

.pred-title {
    font-size: 28px;
    font-weight: bold;
    color: #FFFFFF;
    margin-bottom: 10px;
}

.pred-label {
    font-size: 18px;
    color: #A1A1AA;
}

.conf-text {
    font-size: 20px;
    color: #22C55E;
    font-weight: bold;
}

hr {
    border: 1px solid #27272A;
}

</style>
""", unsafe_allow_html=True)

# ======================
# VALIDASI RETINA
# ======================
def is_retina_image(img_pil):

    img = np.array(img_pil.resize((224, 224)))

    r_mean = np.mean(img[:,:,0])
    g_mean = np.mean(img[:,:,1])
    b_mean = np.mean(img[:,:,2])

    color_score = (r_mean > g_mean) and (r_mean > b_mean)

    gray = np.mean(img, axis=2)

    edge_dark_ratio = np.mean(gray < 30)
    circular_score = edge_dark_ratio > 0.08

    contrast = np.std(gray)
    contrast_score = contrast > 20

    return color_score and circular_score and contrast_score

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_models(training_option):

    # ======================
    # SQUEEZENET
    # ======================
    squeezenet = models.squeezenet1_1(pretrained=True)

    squeezenet.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, 4, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1))
    )

    # ======================
    # SHUFFLENET
    # ======================
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

    in_features = shufflenet.fc.in_features

    shufflenet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 4)
    )

    # ======================
    # PATH MODEL
    # ======================
    if training_option == "Fixed Feature":

        squeeze_path = r"models/best_squeezenet_fixed_feature_non_aug.pth"
        shuffle_path = r"models/best_shufflenet_fixed_feature_non_aug.pth"

    else:

        squeeze_path = r"models/best_squeezenet_partial_finetuning_non_aug.pth"
        shuffle_path = r"models/best_shufflenet_partial_finetuning_non_aug.pth"

    # ======================
    # LOAD WEIGHT
    # ======================
    squeezenet.load_state_dict(torch.load(squeeze_path, map_location=device))
    shufflenet.load_state_dict(torch.load(shuffle_path, map_location=device))

    squeezenet.to(device)
    shufflenet.to(device)

    squeezenet.eval()
    shufflenet.eval()

    return squeezenet, shufflenet

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center;'>
Klasifikasi Penyakit Retina
</h1>

<p style='text-align:center; font-size:18px; color:#D4D4D8;'>
Sistem ini membandingkan hasil klasifikasi penyakit retina
menggunakan <b>SqueezeNet</b> dan <b>ShuffleNet</b>
pada berbagai skenario pelatihan.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================
# INPUT SECTION
# ======================
st.markdown("### Pilih Skenario Pelatihan")

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

# ======================
# PREDICTION
# ======================
if uploaded_file is not None:

    img_pil = Image.open(uploaded_file)

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # ======================
    # VALIDASI RETINA
    # ======================
    if not is_retina_image(img_pil):

        st.error("❌ Gambar bukan citra retina yang valid.")
        st.stop()

    with st.spinner("Memproses prediksi..."):

        # ======================
        # LOAD MODEL
        # ======================
        squeezenet, shufflenet = load_models(training_option)

        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0

        # ======================
        # PREDICT FUNCTION
        # ======================
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

        # ======================
        # SQUEEZENET PREDICTION
        # ======================
        sq_probs, sq_pred, sq_conf, sq_class = predict_model(squeezenet)

        # ======================
        # SHUFFLENET PREDICTION
        # ======================
        sh_probs, sh_pred, sh_conf, sh_class = predict_model(shufflenet)

        # ======================
        # GRADCAM FUNCTION
        # ======================
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

        # ======================
        # GENERATE CAM
        # ======================
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

    # ======================
    # INPUT IMAGE
    # ======================
    st.markdown("## Gambar Input")

    col_img = st.columns([1,2,1])

    with col_img[1]:
        st.image(
            img_pil,
            width=300
        )

    st.markdown("---")

    # ======================
    # HASIL PREDIKSI
    # ======================
    st.markdown(f"## Hasil Prediksi ({training_option})")

    col1, col2 = st.columns(2)

    # ======================
    # SQUEEZENET CARD
    # ======================
    with col1:

        st.markdown(f"""
        <div class="result-card">

        <div class="pred-title">
        SqueezeNet
        </div>

        <div class="pred-label">
        Prediksi: <b>{sq_pred}</b>
        </div>

        <br>

        <div class="conf-text">
        Confidence: {sq_conf*100:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        # ORIGINAL & GRADCAM
        st.markdown("### Original vs Grad-CAM")

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:
            st.image(img, caption="Original")

        with cam_col2:
            st.image(sq_cam, caption="Grad-CAM")

        # PROBABILITAS
        st.markdown("### Probabilitas Kelas")

        sq_df = pd.DataFrame({
            "Class": class_names,
            "Probability": sq_probs.cpu().numpy()
        })

        st.bar_chart(
            sq_df.set_index("Class")
        )

    # ======================
    # SHUFFLENET CARD
    # ======================
    with col2:

        st.markdown(f"""
        <div class="result-card">

        <div class="pred-title">
        ShuffleNet
        </div>

        <div class="pred-label">
        Prediksi: <b>{sh_pred}</b>
        </div>

        <br>

        <div class="conf-text">
        Confidence: {sh_conf*100:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        # ORIGINAL & GRADCAM
        st.markdown("### Original vs Grad-CAM")

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:
            st.image(img, caption="Original")

        with cam_col2:
            st.image(sh_cam, caption="Grad-CAM")

        # PROBABILITAS
        st.markdown("### Probabilitas Kelas")

        sh_df = pd.DataFrame({
            "Class": class_names,
            "Probability": sh_probs.cpu().numpy()
        })

        st.bar_chart(
            sh_df.set_index("Class")
        )

else:
    st.info("Silakan upload citra retina terlebih dahulu.")
