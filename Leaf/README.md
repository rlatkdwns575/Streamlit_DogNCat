# 🌿 Plant Leaf Disease Classification Web App

Streamlit과 PyTorch를 이용해 만든 **식물 잎 이미지 질병 분류 웹 애플리케이션**입니다.  
사용자가 잎 이미지를 업로드하면, 학습된 ResNet 기반 모델이 **33개 클래스 중 하나로 분류**하고, 예측 결과와 확률을 함께 제공합니다.

---

## 1. 프로젝트 개요

본 프로젝트는 식물 잎 이미지를 입력받아  
해당 식물이 **정상 상태인지**, 혹은 **어떤 질병에 해당하는지**를 분류하는 **다중 분류(Multi-class Classification)** 모델입니다.

웹 서비스는 Streamlit으로 구현했으며,  
학습된 PyTorch 모델을 불러와 이미지 1장에 대해 실시간 예측을 수행합니다.

---

## 2. 주요 기능

- 잎 이미지 업로드 (`jpg`, `jpeg`, `png`)
- 학습된 모델을 이용한 **33개 클래스 다중 분류**
- 예측된 클래스 출력
- 예측 확률(확신도) 출력
- Top-5 예측 결과 출력
- 전체 클래스 확률 시각화

---

## 3. 사용 기술

- **Python**
- **PyTorch**
- **Torchvision**
- **Streamlit**
- **Pillow**
- **gdown**

---

## 4. 모델 설명

본 프로젝트의 분류 모델은 **ResNet 계열 CNN 모델**이며,  
최종 출력층은 **33개 클래스**를 예측하도록 구성되어 있습니다.

- 입력 이미지 크기: `224 x 224`
- 입력 채널: `RGB (3채널)`
- 출력 클래스 수: `33`
- 문제 유형: **다중 분류**

기존 이진 분류(Binary Classification) 방식이 아니라,  
`softmax`를 사용하여 33개 클래스 중 가장 확률이 높은 클래스를 선택합니다.

---

## 5. 클래스 목록

본 모델은 아래 33개 클래스를 분류합니다.

```python
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
