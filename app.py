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
st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
THRESHOLD = 0.7

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model(model_option):

    if "SqueezeNet" in model_option:
        model = models.squeezenet1_1(pretrained=True)

        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if "Fixed" in model_option:
            path = r"models/best_squeezenet_fixed_feature_non_aug.pth"
        else:
            path = r"models/best_squeezenet_partial_finetuning_non_aug.pth"

    else:
        model = models.shufflenet_v2_x1_0(pretrained=True)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 4)
        )

        if "Fixed" in model_option:
            path = r"models/best_shufflenet_fixed_feature_non_aug.pth"
        else:
            path = r"models/best_shufflenet_partial_finetuning_non_aug.pth"

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model


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
<h1 style='text-align: center;'>👁️ Klasifikasi Penyakit Retina</h1>
<p style='text-align: center; font-size:18px;'>
Menggunakan <b>SqueezeNet</b> & <b>ShuffleNet</b><br>
dengan interpretasi visual berbasis <b>Grad-CAM</b>
</p>
<hr>
""", unsafe_allow_html=True)

# ======================
# INPUT SECTION
# ======================
st.markdown("## 📥 Input Data")

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Upload Gambar Retina", type=["jpg", "png", "jpeg"])

with col2:
    model_option = st.selectbox(
        "Pilih Model",
        [
            "SqueezeNet - Fixed Feature",
            "SqueezeNet - Partial Fine-Tuning",
            "ShuffleNet - Fixed Feature",
            "ShuffleNet - Partial Fine-Tuning"
        ]
    )

run_btn = st.button("🚀 Jalankan Prediksi")

# ======================
# PREDICTION
# ======================
if run_btn:

    if uploaded_file is not None:

        with st.spinner("Memproses gambar..."):

            model = load_model(model_option)

            img_pil = Image.open(uploaded_file).convert('RGB')
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_class = torch.argmax(probs).item()

            confidence = probs[pred_class].item()

            if confidence < THRESHOLD:
                pred_name = "Tidak dikenali"
                is_valid = False
            else:
                pred_name = class_names[pred_class]
                is_valid = True

        # ======================
        # RESULT SECTION
        # ======================
        st.markdown("---")
        st.markdown("## 📊 Hasil Prediksi")

        col1, col2 = st.columns([1,1])

        img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0

        with col1:
            st.image(img, caption="Citra Input", use_container_width=True)

        with col2:
            if not is_valid:
                st.error("❌ Gambar tidak dikenali oleh model")
            else:
                st.success(f"### {pred_name.upper()}")

            st.metric("Confidence", f"{confidence*100:.2f}%")

        # ======================
        # GRAD-CAM
        # ======================
        if is_valid:
            st.markdown("---")
            st.markdown("## 🔥 Grad-CAM Visualization")

            for param in model.parameters():
                param.requires_grad = True

            if "SqueezeNet" in model_option:
                target_layers = [model.features[-1]]
            else:
                target_layers = [model.stage4[-1]]

            cam = GradCAM(model=model, target_layers=target_layers)

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(pred_class)]
            )[0]

            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption="Original", use_container_width=True)

            with col2:
                st.image(cam_image, caption="Grad-CAM", use_container_width=True)

        # ======================
        # PROBABILITAS
        # ======================
        st.markdown("---")
        st.markdown("## 📈 Probabilitas Kelas")

        df_prob = pd.DataFrame({
            "Class": class_names,
            "Probability": probs.cpu().numpy()
        })

        st.bar_chart(df_prob.set_index("Class"))

    else:
        st.warning("Silakan upload gambar terlebih dahulu.")