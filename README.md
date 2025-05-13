# Cat-dog-panda-classifier
Top-1 classification model for Cat-Dog-Panda task, 98.3% accuracy.
# 🐱🐶🐼 Cat-Dog-Panda Classifier

A high-accuracy image classifier using **ResNet34 + Transfer Learning** in PyTorch, trained to distinguish between cats, dogs, and pandas.

> 🏆 **Leaderboard Rank:** 1st Place  
> 🎯 **Final Score:** 0.98333 accuracy  
> 🚀 Trained from scratch → fine-tuned with pretrained ResNet34

---

## 📌 Features

- ✅ ResNet34 with pretrained ImageNet weights
- ✅ Custom classification head with Dropout & BatchNorm
- ✅ Cosine Annealing Learning Rate scheduler
- ✅ Early stopping & checkpoint saving
- ✅ Data augmentation with normalization
- ✅ TensorBoard logging
- ✅ Confusion matrix for detailed evaluation

---

## 🧠 Model Architecture

```python
self.base_model = resnet34(pretrained=True)
self.base_model.fc = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(self.base_model.fc.in_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 3)  # 3 output classes: cat, dog, panda
)
