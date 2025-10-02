# CIFAR10-CNN-Distributed-Computing
Image classification on CIFAR-10 dataset using Convolutional Neural Networks (CNNs) integrated with Apache Spark for distributed training across VMs. Includes preprocessing, model design, training pipeline, evaluation (accuracy, confusion matrix), and scalability experiments.
# 🖼️ CIFAR-10 Image Classification with CNN and Distributed Computing (Spark)

This project demonstrates how **Convolutional Neural Networks (CNNs)** combined with **Apache Spark** can classify **CIFAR-10 images** at scale. By leveraging distributed computing across multiple VMs, we improved **training efficiency, scalability, and generalization**.  

---

## 📌 Project Overview  
- **Problem:** Training CNNs on large datasets is computationally intensive.  
- **Solution:** Implement CNN on CIFAR-10 with Spark distributed computing for scalability.  
- **Impact:** Enables **faster training, better generalization, and scalable image classification pipelines**.  

---

## 🚀 Implementation  

### Dataset  
- **CIFAR-10:** 60,000 images (32x32, RGB) in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  
- **Train/Test Split:** 50,000 training, 10,000 testing.  
- **Preprocessing:**  
  - Normalization (pixel scaling [0,1])  
  - One-Hot Encoding of class labels  

### CNN Architecture  
- **Conv Layer 1:** 32 filters, 3x3, ReLU + MaxPooling  
- **Conv Layer 2:** 64 filters, 3x3, ReLU + MaxPooling  
- **Conv Layer 3:** 128 filters, 3x3, ReLU + MaxPooling  
- **Flatten + Dense:** 128 units, ReLU, Dropout(0.5)  
- **Output Layer:** 10 units, Softmax  

### Distributed Spark Setup  
- **Executor Memory:** 2 GB  
- **Cores per Executor:** 2  
- **Cluster:** 2 VMs with Spark 3.5.0  
- **Integration:** Spark handled distributed batch training  

### Training Strategy  
- Epochs: 30  
- Batch Size: 64  
- Learning Rate: 0.001  
- Early Stopping applied to avoid overfitting  

---

## 📊 Results  

- **Training Accuracy:** ~72%  
- **Validation Accuracy:** ~74%  
- **Test Accuracy:** ~73.9%  
- **Confusion Matrix:** Provided insights into class-level misclassifications.  

📈 **Accuracy vs Epochs**  
- Training and validation accuracy improved steadily (no major overfitting).  

📉 **Loss vs Epochs**  
- Training and validation losses decreased consistently.  

---

## ⚡ Key Challenges & Solutions  
- **Overfitting →** Used data augmentation, dropout, and early stopping  
- **Computational limits →** Used Spark distributed training across VMs  
- **Hyperparameter tuning →** Adjusted learning rate, dropout rate, and batch size  

---

## 🔮 Future Enhancements  
- **Transfer Learning:** Use pretrained models (ResNet, VGG).  
- **Explainability:** Add Grad-CAM visualizations.  
- **Hyperparameter Optimization:** Bayesian search for fine-tuning.  
- **Deployment:** Package as a scalable ML pipeline.  

---

👩‍💻 **Team Members:**  
- Saketha Kusu → Spark setup, preprocessing, distributed training  
- Varshitha Reddy Davarapalli → CNN architecture & hyperparameter tuning  
- Karunakar Uppalapati → Evaluation metrics & visualization  

🎓 *MS Health Informatics – Michigan Technological University*  
