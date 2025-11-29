readme_md = """# Helmet Detection Classification Pipeline

Progressive helmet detection models from traditional ML → lightweight CNN → state-of-the-art CNN, implemented as Jupyter notebooks (.ipynb) runnable on standard PCs (CPU-only).

Dataset Used: [Kaggle Dataset](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset/data)

## 🚀 Features

| Model           | Accuracy | Training Time | Model Size | Hardware      |
|-----------------|----------|---------------|-----------:|---------------|
| HOG + SVM       | 85–92%   | < 1 min       |    ~1 MB   | Any PC        |
| MobileNetV2     | 92–97%   | 5–10 min      |    ~3 MB   | Standard PC   |
| EfficientNetV2S | 95–98%   | 15–30 min     |   ~15 MB   | Standard PC   |

## 📁 Dataset Structure
```
Helmet_Dataset/
├── Helmet/
│ ├── img1.jpg
│ └── img2.jpg
├── no_person/
│ ├── img1.jpg
│ └── img2.jpg
└── Person_no_helmet/
├── img1.jpg
└── img2.jpg

Test_Files/
├── Person_with_helmet.jpg
├── Person_with_helmet_2.jpg
└── Person_with_helmet_3.jpg
```



## 🛠️ Installation

Core dependencies
```
pip install opencv-python scikit-learn scikit-image joblib tensorflow
```
Optional: For EfficientNetV2S
```
pip install tensorflow-addons
```

## 📊 Model Comparison


| Stage   | Method          | Features              | Best For                              |
|---------|-----------------|-----------------------|---------------------------------------|
| Stage 1 | HOG + SVM       | Handcrafted features  | Ultra-low resource, quick prototyping |
| Stage 2 | MobileNetV2     | Transfer learning     | Balanced accuracy/speed               |
| Stage 3 | EfficientNetV2S | SOTA CNN + fine-tuning| Production-grade accuracy             |


## 🎯 Quick Start

### 1. Traditional ML (HOG + SVM) – Ultra Fast
```
jupyter notebook Image_Classification_HOG.ipynb
```
Train and save: helmet_classifier.pkl
Test on: Test_Files/Person_with_helmet.jpg



### 2. Lightweight CNN (MobileNetV2)
```
jupyter notebook Image_Classification_MobileNetV2.ipynb
```
Train and save: helmet_mobilenetv2.h5
Test on: Test_Files/Person_with_helmet_2.jpg

### 3. State-of-the-Art CNN (EfficientNetV2S)
```
jupyter notebook train_efficientnetv2s.ipynb
```
Train and save: helmet_efficientnetv2s.h5
Test on: Test_Files/Person_with_helmet_3.jpg


## 📈 Expected Performance

```
Dataset: Helmet Detection (3 classes)
├── Training Split: 75%
├── Test Split: 25%
└── Image Size: Auto-scaled (32x32 → 380x380)

HOG+SVM: ~89% [Lightning fast]
MobileNetV2: ~95% [Balanced]
EfficientNetV2S: ~97% [SOTA production]
```

## ⚙️ Model Files Generated
```
models/
├── helmet_hog_svm.pkl # HOG + SVM (1 MB)
├── helmet_mobilenetv2.h5 # MobileNetV2 (3 MB)
└── helmet_efficientnetv2s.h5 # EfficientNetV2S (15 MB)
```

## 🖥️ Hardware Requirements

| Model           | CPU Cores | RAM  | Training Time |
|-----------------|-----------|------|---------------|
| HOG + SVM       | 1 core    | 2 GB | < 1 min       |
| MobileNetV2     | 4 cores   | 4 GB | 5–10 min      |
| EfficientNetV2S | 4–8 cores | 8 GB | 15–30 min     |

All models run inference in < 0.1 s/image on CPU.

## 🎓 Learning Path
```
1.    HOG + SVM → Traditional ML foundations
      └── Feature engineering, classical algorithms

2.    MobileNetV2 → Transfer learning basics
      └── Pretrained models, data augmentation

3.    EfficientNetV2S → SOTA deep learning
      └── Fine-tuning, advanced callbacks, optimization
```

## 🔗 Class Mapping
```
0: Helmet
1: no_person
2: Person_no_helmet
```


## 📚 References

- [HOG + SVM ](https://www.digitalocean.com/community/tutorials/image-classification-without-neural-networks) image classification without deep learning.
- [MobileNetV2](https://slogix.in/source-code/python/deep-learning-samples/how-to-build-an-image-classification-model-with-mobilenetv2-for-cat-and-dog-images/) transfer learning examples.
- [EfficientNetV2](https://labelyourdata.com/articles/image-classification-models) and other modern image classification models.

## 🙌 Contributing

Add dataset improvements, new models, or deployment notebooks!

