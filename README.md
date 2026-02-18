# Multi-Class Classification and Model Tuning

## 📋 Project Overview

This project implements and compares three different machine learning models for **multi-class classification** of student academic performance. The goal is to predict student performance categories (Poor, Average, Good, Excellent) based on various demographic, academic, and personal factors.

The project demonstrates:
- Building and training multiple classification models (Logistic Regression, SVM, MLP)
- Hyperparameter tuning using GridSearchCV
- Model evaluation with cross-validation
- Performance comparison and analysis
- Handling imbalanced datasets

## 🎯 Project Objectives

1. Train and evaluate three different multi-class classifiers
2. Perform hyperparameter tuning to optimize model performance
3. Use 5-fold cross-validation for robust model evaluation
4. Visualize model performance using confusion matrices
5. Analyze classification results and identify model strengths/weaknesses
6. Provide recommendations for future improvements

## 📊 Dataset

### Dataset Information
- **File**: `performance.csv`
- **Total Records**: 1,009 student records
- **Features**: 33 features (mix of categorical and numerical)
- **Target Variable**: Student performance (4 classes)

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Poor | 503 | 49.9% |
| Average | 272 | 27.0% |
| Good | 178 | 17.6% |
| Excellent | 56 | 5.5% |

**Note**: The dataset exhibits significant class imbalance, with "Excellent" being highly underrepresented.

### Features
The dataset includes:
- **Demographic**: Gender, Age, Program, Admission Year
- **Academic Metrics**: SGPA, CGPA, Study Hours, Attendance, Credits Earned
- **Support Factors**: Scholarship Status, Smartphone Access, PC Access, Probation Status
- **Personal Factors**: Health Issues, Physical Disabilities, Relationship Status, Part-time Work
- **Academic Background**: Skills and Interest Areas

## 🤖 Models Implemented

### 1. Logistic Regression
- **Algorithm**: Multi-class logistic regression with SAGA solver
- **Tuning Parameter**: Regularization strength (C)
- **Tuning Range**: [0.01, 0.1, 1, 10, 100]

### 2. Support Vector Machine (SVM)
- **Algorithm**: SVC with RBF kernel
- **Tuning Parameter**: Regularization parameter (C)
- **Tuning Range**: [0.01, 0.1, 1, 10, 100]

### 3. Multi-Layer Perceptron (MLP)
- **Algorithm**: Neural network classifier
- **Tuning Parameter**: Hidden layer architecture
- **Tuning Options**: [(128,), (64,64), (128,64), (128,64,32)]

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Handling**: Addressed missing values in Skills1 (1 missing) and Interest_Area1 (7 missing)
2. **Categorical Encoding**: One-hot encoding for categorical variables
3. **Feature Scaling**: StandardScaler normalization for numerical features
4. **Train-Test Split**: 70/30 split (700 training samples, 301 testing samples)

### Model Training & Evaluation
1. **Baseline Training**: Initial model training with default parameters
2. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
3. **Performance Metrics**: Accuracy, Precision, Recall, F1-score
4. **Visualization**: Confusion matrices for detailed performance analysis

## 📈 Results

### Before Hyperparameter Tuning
| Model | Accuracy | Key Observations |
|-------|----------|------------------|
| Logistic Regression | 53% | Strong bias toward "Poor" class |
| SVM | 53% | Severe class bias, zero recall for minority classes |
| MLP | 68% | **Best performance**, but struggles with "Excellent" class |

### After Hyperparameter Tuning
| Model | Accuracy | Best Parameters | Change |
|-------|----------|-----------------|--------|
| Logistic Regression | 53% | C=0.01 | No improvement |
| SVM | 53% | C=0.01 | No change |
| MLP | 63% | hidden_layer_sizes=(128, 64) | Slight decrease |

### Key Findings
- ✅ **MLP emerged as the best-performing model** with 68% accuracy before tuning
- ⚠️ **"Excellent" class had the worst performance** across all models (often zero recall/precision)
- ⚠️ **Class imbalance significantly affected** minority class prediction
- ⚠️ **Overlapping feature distributions** between classes caused confusion
- ℹ️ **Hyperparameter tuning** did not significantly improve performance, suggesting fundamental data challenges

## 💻 Installation & Usage

### Prerequisites
```bash
Python 3.12.6 or higher
```

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Clone this repository:
```bash
git clone https://github.com/DataDarling/Multi-Class-Classification-and-Model-Tuning.git
cd Multi-Class-Classification-and-Model-Tuning
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `Multi-Class Classification and Model Tuning.ipynb`

4. Run all cells sequentially to reproduce the analysis

## 📁 Project Structure
```
Multi-Class-Classification-and-Model-Tuning/
│
├── Multi-Class Classification and Model Tuning.ipynb  # Main analysis notebook
├── performance.csv                                    # Dataset (not included in repo)
└── README.md                                          # Project documentation
```

## 🔧 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and tools
  - `LogisticRegression`: Logistic regression classifier
  - `SVC`: Support vector classifier
  - `MLPClassifier`: Multi-layer perceptron classifier
  - `GridSearchCV`: Hyperparameter tuning
  - `StandardScaler`: Feature scaling
  - `train_test_split`: Data splitting
- **matplotlib**: Data visualization

## 🚀 Future Improvements

Based on the analysis, the following improvements are recommended:

1. **Address Class Imbalance**:
   - Implement SMOTE (Synthetic Minority Over-sampling Technique)
   - Use class weight adjustments
   - Try undersampling majority classes

2. **Feature Engineering**:
   - Create interaction features
   - Perform feature selection to reduce noise
   - Engineer domain-specific features

3. **Try Advanced Models**:
   - Ensemble methods (Random Forest, XGBoost, LightGBM)
   - Deep learning architectures with dropout and regularization
   - Voting classifiers combining multiple models

4. **Alternative Evaluation Strategies**:
   - Use stratified sampling for better class representation
   - Focus on macro-averaged metrics for imbalanced data
   - Implement cost-sensitive learning

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**DataDarling**
- GitHub: [@DataDarling](https://github.com/DataDarling)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!