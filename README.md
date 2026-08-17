# 🎓 Student Performance Prediction and Analytics System

> **An end-to-end Machine Learning and Analytics application for predicting students' final academic marks and generating performance insights, risk levels, and personalized recommendations.**

---

## 📌 Project Overview

The **Student Performance Prediction and Analytics System** is a Machine Learning-based application developed to analyze student-related academic and personal factors and predict their expected **Final Marks**.

The system uses a dataset containing **5,000 student records** and applies data preprocessing, feature transformation, and a trained **Linear Regression** model to generate predictions.

The trained model is integrated into an interactive **Streamlit web application**, allowing users to enter student information and receive an estimated final mark along with additional academic insights.

### 🎯 Main Goal

The main goal of this project is to demonstrate how Machine Learning and Data Analytics can be used to support **data-driven academic analysis and early performance assessment**.

---

# ✨ Key Features

* 📊 Student performance prediction
* 🤖 Machine Learning-based prediction
* 📈 Data exploration and visualization
* 🧹 Automated data preprocessing
* 🔤 Categorical feature encoding
* 📏 Numerical feature scaling
* 🎯 Final marks prediction
* 📋 Performance categorization
* ⚠️ Risk-level identification
* 💡 Personalized academic recommendations
* 📊 Feature coefficient analysis
* 📈 Model performance analysis
* 🎯 Goal Planner
* 🌐 Interactive Streamlit dashboard
* 💾 Saved trained ML pipeline
* 📓 Jupyter Notebook-based EDA and model development

---

# 🧠 Machine Learning Approach

This project follows an end-to-end Machine Learning workflow:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Data Cleaning & Preprocessing
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
Categorical Encoding
     ↓
Numerical Feature Scaling
     ↓
Linear Regression
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Application
     ↓
Student Prediction
```

---

# 📊 Dataset

The project uses a dataset containing **5,000 student records**.

### Important Features

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| Student_ID         | Unique student identifier                   |
| Gender             | Student gender                              |
| Study_Hours        | Average study hours                         |
| Attendance_Percent | Attendance percentage                       |
| Previous_Marks     | Previous academic marks                     |
| Assignment_Score   | Assignment performance                      |
| Quiz_Score         | Quiz performance                            |
| Internal_Marks     | Internal assessment marks                   |
| Internet_Access    | Internet availability                       |
| Extracurricular    | Participation in extracurricular activities |
| Sleep_Hours        | Average sleep hours                         |
| Final_Marks        | Target variable                             |

### Target Variable

```text
Final_Marks
```

The model predicts the student's expected final academic marks.

> **Note:** `Student_ID` is treated as an identifier and is not used as a predictive feature.

---

# 🔧 Data Preprocessing

Before training the model, the data goes through several preprocessing steps.

### 1. Feature and Target Separation

```text
X → Input Features
y → Final_Marks
```

### 2. Identifier Removal

`Student_ID` is removed because it is an identifier rather than a meaningful predictive feature.

### 3. Numerical Feature Scaling

Numerical variables are standardized using:

```text
StandardScaler
```

### 4. Categorical Encoding

Categorical variables are converted into numerical representations using:

```text
OneHotEncoder
```

The encoder uses:

```text
handle_unknown = "ignore"
```

to safely handle previously unseen categories during prediction.

### 5. ColumnTransformer

Different preprocessing operations are applied to numerical and categorical features using:

```text
ColumnTransformer
```

### 6. Machine Learning Pipeline

The preprocessing and model are combined into a single Scikit-learn pipeline.

This ensures that the same preprocessing procedure is applied during both training and prediction.

---

# 🤖 Machine Learning Model

## Linear Regression

The final prediction model used in the project is:

```text
Linear Regression
```

### Why Linear Regression?

Linear Regression is suitable because:

* The target variable is numerical.
* `Final_Marks` is a continuous value.
* The model is relatively simple and interpretable.
* Feature coefficients can provide useful information about learned relationships.

### Model Pipeline

```text
ColumnTransformer
       ↓
Preprocessing
       ↓
Linear Regression
       ↓
Predicted Final Marks
```

---

# 📈 Model Performance

The project reports an:

```text
R² Score = 0.81
```

An R² score of **0.81** indicates that the model explains approximately **81% of the variation in Final_Marks on the evaluated data**.

> **Important:** R² should not be interpreted as classification accuracy. It is a regression evaluation metric.

---

# 💾 Model Serialization

After training, the trained Machine Learning pipeline is saved using **Joblib**.

```text
models/
└── student_performance_model.pkl
```

This allows the Streamlit application to load the trained model directly without retraining it every time the application starts.

---

# 🌐 Streamlit Application

The trained model is integrated into a Streamlit-based web application.

The application provides an interactive interface where users can enter student information and generate predictions.

### Prediction Flow

```text
User Input
    ↓
Pandas DataFrame
    ↓
Saved ML Pipeline
    ↓
Preprocessing
    ↓
Linear Regression
    ↓
Predicted Final Marks
    ↓
Performance / Risk / Recommendations
```

---

# 🖥️ Application Modules

## 🏠 Home

Provides an introduction to the project and explains the purpose of the system.

## 🎯 Student Prediction

Allows users to enter student information and generate predicted final marks.

## 📊 Dataset Insights

Provides data exploration and visualizations to understand patterns in the dataset.

## 📈 Feature Analysis

Displays information about the relationships learned by the regression model.

## 🎯 Goal Planner

Helps students define an academic target and provides goal-oriented guidance.

## 📋 Model Performance

Displays information about the trained model and its evaluation.

## ℹ️ About

Provides information about the project, technologies and development approach.

---

# 📁 Project Structure

```text
Student-Performance-Prediction-Analytics/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
│   └── student_performance_dataset1_corrected.csv
│
├── models/
│   └── student_performance_model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Machine_Learning.ipynb
│
├── src/
│   ├── predict.py
│   ├── utils.py
│   └── pdf_generator.py
│
└── images/
    ├── assignment_score_distribution.png
    ├── attendance_distribution.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── final_marks_distribution.png
    └── ...
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn
* Linear Regression

### Model Management

* Joblib

### Web Application

* Streamlit

### Development Environment

* Jupyter Notebook
* VS Code / PowerShell
* Git & GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/bodaleprathmesh-code/Student-Performance-Prediction-Analytics.git
```

## 2. Navigate to the Project

```bash
cd Student-Performance-Prediction-Analytics
```

## 3. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

## 4. Activate the Environment

```powershell
venv\Scripts\activate
```

## 5. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application using:

```powershell
python -m streamlit run app.py
```

The application will open in your browser.

---

# 📓 Run the Notebooks

The project contains separate notebooks for:

### Exploratory Data Analysis

```text
notebooks/EDA.ipynb
```

### Machine Learning

```text
notebooks/Machine_Learning.ipynb
```

These notebooks contain the data analysis, visualization, preprocessing, model training and evaluation workflow.

---

# 🔐 Project Security

The repository uses a `.gitignore` file to prevent unnecessary or sensitive files from being committed.

Examples include:

```text
venv/
.env
*.env
*.key
*.pem
secrets/
credentials/
__pycache__/
.ipynb_checkpoints/
```

No API keys, passwords or private credentials should be stored directly in the repository.

---

# 🚀 Future Improvements

The project can be further improved by adding:

* Advanced regression algorithms
* Hyperparameter optimization
* Model comparison dashboard
* Cross-validation reporting
* More detailed student analytics
* Explainable AI techniques
* SHAP-based feature explanations
* Historical performance tracking
* Student progress monitoring
* Database integration
* Authentication and user management
* Cloud-based deployment
* Automated model retraining
* Improved recommendation system

---

# ⚠️ Limitations

* Predictions depend on the quality and distribution of the training data.
* The model provides estimates rather than guaranteed future marks.
* Relationships learned by the model should not automatically be interpreted as causal relationships.
* Model performance may change when applied to substantially different datasets.
* Student lifestyle features should be interpreted carefully and in context.

---

# 🎯 Project Outcome

The project demonstrates a complete Machine Learning lifecycle:

```text
Data
 ↓
EDA
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Evaluation
 ↓
Model Saving
 ↓
Web Deployment
 ↓
Prediction
 ↓
Academic Insights
```

The final system combines **Machine Learning + Data Analytics + Web Application Development** into a single interactive platform.

---

# 🔗 Project Links

### GitHub Repository

https://github.com/bodaleprathmesh-code/Student-Performance-Prediction-Analytics

### Live Demo

https://student-performance-prediction-analytics-system.streamlit.app/

---

# 📜 License

This project is developed for **academic and educational purposes**.

---

# ⭐ Acknowledgement

This project was developed as an academic Machine Learning and Data Analytics project to demonstrate the practical application of data preprocessing, regression modelling, model evaluation and web-based deployment.

---

## 👨‍💻 Developed With Python & Machine Learning

**Student Performance Prediction and Analytics System**
*Turning student data into meaningful academic insights.*
