# Streamlit Classification

Streamlit과 PyTorch를 활용해 **직접 학습한 이미지 분류 모델을 웹 서비스 형태로 배포**한 프로젝트 모음입니다.  
사용자는 웹 페이지에서 이미지를 업로드하고, 학습된 모델이 예측한 분류 결과를 바로 확인할 수 있습니다.

이 저장소는 여러 분류 실험을 하나의 레포지토리로 정리한 형태이며, 현재 확인되는 주요 프로젝트는 다음과 같습니다.

- **DogsNCats**: 개 / 고양이 이진 분류 웹 앱
- **Leaf**: 식물 잎 질병 다중 분류 웹 앱
- **Ball_classification**: 공 이미지 분류 관련 폴더
- **Yolo26**: Yolo26이용 이미지 분류 관련 폴더

---

## 1. 프로젝트 개요

이 프로젝트는 단순히 모델을 학습하는 데서 끝나지 않고,  
**학습한 모델을 실제 사용 가능한 분류 사이트로 연결**하는 데 목적이 있습니다.

즉, 이미지 분류 모델을 만든 뒤 다음 과정을 하나의 흐름으로 구현했습니다.

1. 이미지 분류 모델 학습
2. 학습된 가중치 저장
3. Streamlit 웹 앱으로 추론 인터페이스 구현
4. 사용자가 이미지를 업로드하면 실시간 분류 결과 제공

---

## 2. 저장소 구성

```bash
Streamlit_Classification/
├─ Ball_classification/
├─ DogsNCats/
│  ├─ README.md
│  ├─ app.py
│  ├─ CatsnDogs.ipynb
│  └─ requirements.txt
├─ Leaf/
│  ├─ README.md
│  ├─ app.py
│  ├─ leaf.ipynb
│  └─ requirements.txt
├─ Yolo26/
│  ├─ README.md
│  ├─ app.py
│  └─ requirements.txt
└─ LICENSE
