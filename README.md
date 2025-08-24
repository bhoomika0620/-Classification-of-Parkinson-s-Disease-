# Parkinson's Disease Classification using Machine Learning

## Overview
This project focuses on the **early detection of Parkinson's Disease (PD)** using supervised machine learning algorithms. Parkinson’s disease is a neurodegenerative disorder caused by the loss of dopamine-producing brain cells, leading to symptoms such as tremors, stiffness, and difficulty with balance and coordination.  
The goal of this project is to classify patients as **healthy or Parkinson’s positive** based on their vocal features. By applying machine learning techniques, the model can aid in the early diagnosis of Parkinson's, which is crucial for effective treatment and improved quality of life.  

## Objectives
- To analyze and classify voice features from the **UCI Parkinson’s dataset**. <br>
- To compare the performance of **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** classifiers. <br>
- To evaluate models based on **Accuracy, Sensitivity, Specificity, Precision, and F-measure**. <br>

## Dataset
- **Source**: [UCI Machine Learning Repository - Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)  <br>
- **Size**: 195 voice recordings from 31 individuals <br>
  - 23 diagnosed with Parkinson’s <br>
  - 8 healthy <br>
- **Features**: Vocal measures such as jitter, shimmer, NHR, HNR, DFA, spread1, PPE, etc.  <br>
- **Target variable**: `status` (0 = Healthy, 1 = Parkinson’s)  <br>

##  Methodology

1. **Data Preprocessing**:  
   - Extracted relevant vocal features <br>
   - Computed statistical measures (mean, std, variance, skewness, kurtosis) <br>
     
2. **Algorithms Used**:  
   - Support Vector Machine (SVM) <br>
   - K-Nearest Neighbors (KNN) <br>
     
3. **Evaluation Metrics**:  
   - Accuracy <br>
   - Sensitivity (Recall) <br>
   - Specificity <br>
   - Precision <br> 
   - F1-Score <br>

## Results
```bash
| Classifier | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|------------|----------|-------------|-------------|-----------|----------|
| **SVM**    | 89.74%   | 80.43%      | 92.61%      | 77.08%    | 78.71%   |
| **KNN**    | 93.84%   | 86.00%      | 96.50%      | 89.58%    | 87.75%   |
```

**KNN outperformed SVM**, achieving the highest accuracy of **93.84%**.  

## Technologies Used
- Python 🐍 <br>
- Scikit-learn <br>
- Pandas, NumPy <br>
- Matplotlib, Seaborn  

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/parkinsons-disease-classification.git
   cd parkinsons-disease-classification
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook/script to train and test models:
```bash
python main.py
```

## Applications
Early detection of Parkinson’s Disease <br>
Clinical decision support for neurologists <br>
Voice-based health monitoring systems <br>

## References
UCI Parkinson’s Dataset <br>
Gunduz, H. Deep Learning-Based Parkinson’s Disease Classification Using Vocal Feature Sets, IEEE Access, 2019. <br>
Alzubaidi, M.S. et al. The Role of Neural Networks for Detection of Parkinson’s Disease, Healthcare, 2021.
