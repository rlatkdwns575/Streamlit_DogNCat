# app.py — Leaf 다중 분류 웹 서비스
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gdown
import os

# ── 모델 설정 ─────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?id=1BqXvzfG48wN3zsKHMYaE-m-fsbCOTWBQ"
MODEL_PATH = "resnet50.pt"

# 반드시 학습 때의 클래스 순서와 동일하게 입력해야 함
CLASS_NAMES = [
    "Strawberry___Leaf_scorch",
    "Corn___Northern_Leaf_Blight",
    "Tomato___Early_blight",
    "Apple___healthy",
    "Cherry___healthy",
    "Strawberry___healthy",
    "Potato___Early_blight",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Grape___healthy",
    "Pepper,_bell___healthy",
    "Apple___Apple_scab",
    "Tomato___healthy",
    "Tomato___Tomato_mosaic_virus",
    "Pepper,_bell___Bacterial_spot",
    "Corn___Common_rust",
    "Tomato___Leaf_Mold",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Corn___healthy",
    "Cherry___Powdery_mildew",
    "Apple___Black_rot",
    "Tomato___Late_blight",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Tomato___Target_Spot",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Potato___Late_blight",
    "Grape___Black_rot",
    "Tomato___Septoria_leaf_spot",
    "Apple___Cedar_apple_rust",
    "Grape___Esca_(Black_Measles)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
]

# ── 모델 로드 (앱 시작 시 1회만 실행) ──────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # 이 파일은 state_dict가 아니라 "모델 전체"가 저장된 파일
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    return model

model = load_model()

# ── 이미지 전처리 (학습 시와 동일하게 맞추는 것이 중요) ──────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(page_title="Leaf Type 식별기", page_icon="🌿")
st.title("🌿 Leaf Type 식별기")
st.caption("이미지를 업로드하면 잎의 종류를 다중 분류합니다.")

# ── 이미지 업로드 ─────────────────────────────────────────
uploaded = st.file_uploader(
    "이미지를 선택하세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    # ── 예측 ──────────────────────────────────────────────
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)              # shape: [1, 33]
        probs = F.softmax(logits, dim=1)[0]       # 33개 클래스 확률
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    # ── 결과 표시 ─────────────────────────────────────────
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"

    st.markdown(f"### 예측 결과: **{pred_label}**")
    st.metric("확신도", f"{confidence:.1%}")

    # Top-5 확률 출력
    st.subheader("Top-5 예측 결과")
    top5_probs, top5_indices = torch.topk(probs, k=5)

    top5_data = []
    for p, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        top5_data.append((label, p))

    for label, p in top5_data:
        st.write(f"- {label}: {p:.2%}")

    # 전체 클래스 확률 시각화
    st.subheader("전체 클래스 확률")
    chart_data = {
        (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"): float(probs[i])
        for i in range(len(probs))
    }
    st.bar_chart(chart_data)

else:
    st.info("왼쪽 위의 업로드 버튼을 눌러 이미지를 선택하세요.")