# Title (Please modify the title)
## Team

| ![이민우](https://avatars.githubusercontent.com/u/156163982?v=4) | ![조선미](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이준석](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이나경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![황준엽](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이민우](https://github.com/UpstageAILab)             |            [조선미](https://github.com/UpstageAILab)             |            [이준석](https://github.com/UpstageAILab)             |            [이나경](https://github.com/UpstageAILab)             |            [황준엽](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |
## 📊 경진대회 수행 결과

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

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- _Write competition information_

### Timeline

- ex) January 10, 2024 - Start Date
- ex) February 10, 2024 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
