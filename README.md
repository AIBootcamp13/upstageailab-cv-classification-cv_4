# Title (Please modify the title)
## Team

| ![이민우](https://avatars.githubusercontent.com/u/156163982?v=4) | ![조선미](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이준석](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이나경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![황준엽](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이민우](https://github.com/UpstageAILab)             |            [조선미](https://github.com/UpstageAILab)             |            [이준석](https://github.com/UpstageAILab)             |            [이나경](https://github.com/UpstageAILab)             |            [황준엽](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |



# 🧠 Document Classification - AI Bootcamp CV Competition

## 0. Overview

### Environment

- OS: Ubuntu 20.04.6 LTS
- Python: 3.10.x
- PyTorch: 2.0.1
- CUDA: 11.7
- Jupyter Notebook / VSCode for development
- GPU: A100 / T4 (Google Colab, AI Stages 활용)
- Model Training & Experiment Tracking: MLflow, TensorBoard

### Requirements


pip install -r requirements.txt


## 1. Competition Info

### Overview

- **대회명**: 패스트캠퍼스 AI 부트캠프 13기 - CV 경진대회
- **주제**: 문서 이미지 분류 (Document Classification)
- **목표**: 학습 데이터(1,570장)를 활용하여 테스트 데이터(3,140장)를 17개 문서 클래스로 분류하는 모델 개발
- **세부 내용**:
  - 문서 종류는 금융, 의료, 보험, 물류 등 다양한 도메인 포함
  - 문서 이미지의 회전, 조도 불균형, 노이즈 등 실제 사용 환경에서 발생할 수 있는 이슈를 고려해야 함
  - 모델 구조부터 실험 전략, 협업 방식까지 전 과정을 팀원 주도로 구성

### Timeline

- 📌 2025.06.30 - 대회 시작 (문제 및 데이터 공개)
- 📈 2025.07.01 ~ 2025.07.09 - 데이터 분석, 모델 학습 및 성능 개선
- 📨 2025.07.10 - 최종 제출 마감
- 🗣️ 2025.07.11 - 발표 및 회고 세션 진행


### Directory

'''
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   ├── train.py
│   ├── test.py
│   └── utils/
│       ├── data_utils.py
│       └── model_utils.py
├── docs
│   ├── presentation
│   │   └── [패스트캠퍼스]-AI-부트캠프-13기_CV-경진대회-발표.pdf
│   └── paper
├── input
│   └── data
│       ├── train/
│       ├── eval/
│       └── test/
├── output
│   ├── models/
│   └── predictions/
└── README.md
'''
## 3. Data descrption

### 🔍 문제점 분석

- **클래스 불균형 (Class Imbalance)**
  - 일부 클래스에 데이터가 집중되어 있어 성능 저하 우려
  - → Stratified K-Fold 및 Weighted Loss 등을 고려

- **조도 및 그림자 문제 (Lighting & Shadows)**
  - 그림자가 짙거나 밝기가 너무 높은 문서 이미지 존재
  - → `Brightness Adjustment`, `Shadow Simulation` 등의 **조도 관련 Augmentation** 적용

- **회전 문제 (Rotation Issue)**
  - 일부 문서가 **90도, 180도** 회전되어 있음
  - → `Rotation(±90°, ±180°)` 중심의 **회전 Augmentation** 적용

---

### 🛠 적용된 해결 방법

- ✅ 밝기 조절: `transforms.ColorJitter` 및 Augraphy 기반 조명 보정
- ✅ 회전 대응: `transforms.RandomRotation` 및 고정 각도 회전 적용
- ✅ 데이터 증강: 온라인 & 오프라인 증강 전략 병행

---

### 📌 요약

| 문제 항목 | 인사이트 | 대응 전략 |
|-----------|----------|-----------|
| 클래스 불균형 | 성능 저하 유발 가능 | Stratified Split, Class Weights |
| 밝기/그림자 | OCR 및 인식 저해 | 조도 보정 Augmentation |
| 문서 회전 | 정방향 학습 방해 | 회전 Augmentation 추가 |

---

> 📁 해당 실험은 `eda/augmentation_analysis.ipynb` 및 `notebooks/preprocessing.ipynb`에서 확인할 수 있습니다.


## 5. Result
<img width="1238" height="468" alt="image" src="https://github.com/user-attachments/assets/d5116fde-a92d-4456-a0f7-8db5f6abddff" />

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

https://docs.google.com/presentation/d/1iR3Ts_w_UZA_SSlnCyUWTRFi_FIz5OfY/edit?slide=id.p7#slide=id.p7

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
