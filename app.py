import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from src.pdf_generator import generate_pdf
from src.predict import predict_student_performance
from src.utils import (
    get_performance_category,
    get_risk_level,
    get_recommendations
)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Load Data & Model
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "student_performance_dataset1_corrected.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_performance_model.pkl"
)

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

df = load_dataset(DATASET_PATH)
model = load_pipeline(MODEL_PATH)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "predicted_marks" not in st.session_state:
    st.session_state.predicted_marks = None

if "performance" not in st.session_state:
    st.session_state.performance = None

if "risk" not in st.session_state:
    st.session_state.risk = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ---------------------------------------------------
# Navigation
# ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Student Prediction",
        "📈 Dataset Insights",
        "📉 Feature Importance",
        "📋 Model Performance",
        "🎯 Goal Planner",
        "ℹ️ About"
    ]
)

# ---------------------------------------------------
# Page 1: Home
# ---------------------------------------------------
if page == "🏠 Home":

    # =========================================================
    # HOME PAGE - PROFESSIONAL LANDING PAGE
    # =========================================================

    # ---------- HERO SECTION ----------
    hero_left, hero_right = st.columns([1.7, 1])

    with hero_left:

        st.markdown("""
        <div style="
            padding: 25px 0 10px 0;
        ">

        <div style="
            color:#2563EB;
            font-size:15px;
            font-weight:700;
            letter-spacing:1.5px;
            margin-bottom:15px;
        ">
        🎓 AI-POWERED ACADEMIC ANALYTICS
        </div>

        <div style="
            font-size:48px;
            font-weight:800;
            line-height:1.12;
            color:#172033;
            margin-bottom:20px;
        ">
        Student Performance<br>
        Prediction & Analytics
        </div>

        <div style="
            font-size:21px;
            font-weight:600;
            color:#334155;
            line-height:1.5;
            margin-bottom:18px;
        ">
        Smarter insights for better academic outcomes.
        </div>

        <div style="
            font-size:16px;
            color:#64748B;
            line-height:1.7;
            max-width:700px;
        ">
        An interactive Machine Learning system designed to
        analyze student academic data, predict performance,
        identify important factors, and provide personalized
        academic recommendations.
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 Start Student Prediction",
                use_container_width=True
            ):
                st.session_state["home_navigation"] = "📊 Student Prediction"

        with col2:
            if st.button(
                "📈 Explore Dataset",
                use_container_width=True
            ):
                st.session_state["home_navigation"] = "📈 Dataset Insights"


    with hero_right:

        st.markdown("""
        <div style="
            background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
            border-radius:24px;
            padding:45px 30px;
            text-align:center;
            min-height:320px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            box-shadow:0 10px 30px rgba(37,99,235,0.08);
        ">

        <div style="
            font-size:90px;
            margin-bottom:20px;
        ">
        🎓
        </div>

        <div style="
            font-size:24px;
            font-weight:700;
            color:#1E3A8A;
            margin-bottom:12px;
        ">
        Smart Academic Insights
        </div>

        <div style="
            font-size:15px;
            color:#475569;
            line-height:1.6;
        ">
        Analyze • Predict • Understand • Improve
        </div>

        </div>
        """, unsafe_allow_html=True)


    st.divider()


    # =========================================================
    # WHAT THIS SYSTEM DOES
    # =========================================================

    st.markdown("""
    <div style="margin-top:10px;">

    <h2 style="color:#172033;">
    💡 What Can You Do With This System?
    </h2>

    <p style="color:#64748B;">
    Explore academic performance through data-driven Machine Learning.
    </p>

    </div>
    """, unsafe_allow_html=True)


    feature1, feature2, feature3 = st.columns(3)


    with feature1:

        st.markdown("""
        <div style="
            padding:25px;
            border:1px solid #E2E8F0;
            border-radius:18px;
            background:#FFFFFF;
            min-height:190px;
        ">

        <div style="font-size:35px;">🎯</div>

        <h3 style="color:#172033;">
        Performance Prediction
        </h3>

        <p style="color:#64748B; line-height:1.6;">
        Enter student academic information and estimate
        the expected final marks using the trained
        Machine Learning model.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with feature2:

        st.markdown("""
        <div style="
            padding:25px;
            border:1px solid #E2E8F0;
            border-radius:18px;
            background:#FFFFFF;
            min-height:190px;
        ">

        <div style="font-size:35px;">📊</div>

        <h3 style="color:#172033;">
        Data Analytics
        </h3>

        <p style="color:#64748B; line-height:1.6;">
        Explore student data through statistics,
        distributions, interactive charts and
        academic performance patterns.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with feature3:

        st.markdown("""
        <div style="
            padding:25px;
            border:1px solid #E2E8F0;
            border-radius:18px;
            background:#FFFFFF;
            min-height:190px;
        ">

        <div style="font-size:35px;">💡</div>

        <h3 style="color:#172033;">
        Academic Insights
        </h3>

        <p style="color:#64748B; line-height:1.6;">
        Understand important academic factors and
        receive personalized recommendations to
        support better performance.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # =========================================================
    # HOW THE SYSTEM WORKS
    # =========================================================

    st.markdown("""
    <h2 style="color:#172033;">
    ⚙️ How the Prediction System Works
    </h2>

    <p style="color:#64748B;">
    A simple Machine Learning workflow converts student data
    into meaningful academic insights.
    </p>
    """, unsafe_allow_html=True)


    step1, step2, step3, step4 = st.columns(4)


    with step1:

        st.markdown("""
        <div style="
            text-align:center;
            padding:20px;
            background:#F8FAFC;
            border-radius:16px;
            min-height:160px;
        ">

        <div style="font-size:35px;">📝</div>

        <h4 style="color:#172033;">
        1. Student Data
        </h4>

        <p style="color:#64748B;">
        Academic and personal
        learning-related inputs
        </p>

        </div>
        """, unsafe_allow_html=True)


    with step2:

        st.markdown("""
        <div style="
            text-align:center;
            padding:20px;
            background:#F8FAFC;
            border-radius:16px;
            min-height:160px;
        ">

        <div style="font-size:35px;">⚙️</div>

        <h4 style="color:#172033;">
        2. Preprocessing
        </h4>

        <p style="color:#64748B;">
        Encoding, scaling and
        feature transformation
        </p>

        </div>
        """, unsafe_allow_html=True)


    with step3:

        st.markdown("""
        <div style="
            text-align:center;
            padding:20px;
            background:#F8FAFC;
            border-radius:16px;
            min-height:160px;
        ">

        <div style="font-size:35px;">🤖</div>

        <h4 style="color:#172033;">
        3. ML Model
        </h4>

        <p style="color:#64748B;">
        Trained Linear Regression
        model generates prediction
        </p>

        </div>
        """, unsafe_allow_html=True)


    with step4:

        st.markdown("""
        <div style="
            text-align:center;
            padding:20px;
            background:#F8FAFC;
            border-radius:16px;
            min-height:160px;
        ">

        <div style="font-size:35px;">💡</div>

        <h4 style="color:#172033;">
        4. Insights
        </h4>

        <p style="color:#64748B;">
        Performance category,
        risk level and recommendations
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.divider()


    # =========================================================
    # KEY INPUT FACTORS
    # =========================================================

    st.markdown("""
    <h2 style="color:#172033;">
    📚 Key Academic Factors
    </h2>

    <p style="color:#64748B;">
    The prediction system considers multiple student-related
    features while generating the expected performance.
    </p>
    """, unsafe_allow_html=True)


    factor1, factor2, factor3, factor4 = st.columns(4)


    with factor1:
        st.info("📖 **Study Hours**\n\nDaily study time")


    with factor2:
        st.info("📅 **Attendance**\n\nAttendance percentage")


    with factor3:
        st.info("📝 **Previous Marks**\n\nPrevious academic performance")


    with factor4:
        st.info("📊 **Assessment Scores**\n\nAssignment, quiz and internal marks")


    factor5, factor6, factor7 = st.columns(3)


    with factor5:
        st.info("🌐 **Internet Access**\n\nAvailability of learning resources")


    with factor6:
        st.info("🏃 **Extracurricular Activities**\n\nStudent participation")


    with factor7:
        st.info("😴 **Sleep Hours**\n\nIncluded as a student-related input")


    st.divider()


    # =========================================================
    # TECHNOLOGY STACK
    # =========================================================

    st.markdown("""
    <h2 style="color:#172033;">
    🛠️ Technology Stack
    </h2>

    <p style="color:#64748B;">
    Technologies used to build the Student Performance
    Prediction & Analytics System.
    </p>
    """, unsafe_allow_html=True)


    tech1, tech2, tech3, tech4, tech5 = st.columns(5)


    with tech1:
        st.markdown("### 🐍 Python")


    with tech2:
        st.markdown("### 🐼 Pandas")


    with tech3:
        st.markdown("### 🔢 NumPy")


    with tech4:
        st.markdown("### 🤖 Scikit-Learn")


    with tech5:
        st.markdown("### 📊 Plotly")


    st.markdown("<br>", unsafe_allow_html=True)


    tech6, tech7 = st.columns(2)


    with tech6:
        st.markdown("### 🌐 Streamlit")


    with tech7:
        st.markdown("### 💾 Joblib")


    st.divider()


    # =========================================================
    # PROJECT PURPOSE
    # =========================================================

    purpose_left, purpose_right = st.columns([1.4, 1])


    with purpose_left:

        st.markdown("""
        <h2 style="color:#172033;">
        🎯 Project Purpose
        </h2>

        <p style="
            color:#475569;
            font-size:16px;
            line-height:1.8;
        ">
        This project demonstrates how Machine Learning and
        data analytics can be applied to academic performance
        analysis.
        </p>

        <p style="
            color:#475569;
            font-size:16px;
            line-height:1.8;
        ">
        The goal is not only to generate a predicted mark,
        but also to help understand student performance
        patterns and provide useful academic insights.
        </p>
        """, unsafe_allow_html=True)


    with purpose_right:

        st.markdown("""
        <div style="
            background:#EFF6FF;
            border-radius:20px;
            padding:30px;
        ">

        <h3 style="color:#1E3A8A;">
        🚀 Built for Academic Analytics
        </h3>

        <p style="
            color:#475569;
            line-height:1.7;
        ">
        Analyze student data.<br>
        Build Machine Learning models.<br>
        Generate predictions.<br>
        Discover meaningful insights.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br><br>", unsafe_allow_html=True)


    # =========================================================
    # FOOTER
    # =========================================================

    st.markdown("""
    <div style="
        text-align:center;
        padding:25px;
        border-top:1px solid #E2E8F0;
        color:#64748B;
    ">

    <b style="color:#334155;">
    🎓 Student Performance Prediction & Analytics System
    </b>

    <br><br>

    Machine Learning • Data Analytics • Academic Insights

    </div>
    """, unsafe_allow_html=True)
# ---------------------------------------------------
# Page 2: Student Prediction
# ---------------------------------------------------
elif page == "📊 Student Prediction":
    st.title("📊 Student Performance Prediction")
    st.markdown("""
Predict a student's expected final marks using the trained Machine Learning model.
Please fill all required details below.
""")
    st.divider()

    left_col, right_col = st.columns([2, 1])

    with left_col:
        with st.container(border=True):
            st.subheader("📝 Student Information")

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )
            study_hours = st.slider(
                "Study Hours (per day)",
                0.0, 12.0, 6.0, 0.5
            )
            attendance = st.slider(
                "Attendance (%)",
                50, 100, 80
            )
            previous_marks = st.slider(
                "Previous Marks",
                30, 95, 70
            )
            assignment = st.slider(
                "Assignment Score",
                40, 100, 75
            )
            quiz = st.slider(
                "Quiz Score",
                30, 100, 70
            )
            internal = st.slider(
                "Internal Marks",
                10, 30, 22
            )
            internet = st.selectbox(
                "Internet Access",
                ["Yes", "No"]
            )
            extra = st.selectbox(
                "Extracurricular Activities",
                ["Yes", "No"]
            )
            sleep = st.slider(
                "Sleep Hours",
                4.0, 10.0, 7.0, 0.5
            )

            predict_btn = st.button(
                "🚀 Predict Final Marks",
                use_container_width=True
            )

            if predict_btn:
                student_data = pd.DataFrame({
                    "Gender": [gender],
                    "Study_Hours": [study_hours],
                    "Attendance_Percent": [attendance],
                    "Previous_Marks": [previous_marks],
                    "Assignment_Score": [assignment],
                    "Quiz_Score": [quiz],
                    "Internal_Marks": [internal],
                    "Internet_Access": [internet],
                    "Extracurricular": [extra],
                    "Sleep_Hours": [sleep]
                })

                predicted_marks = predict_student_performance(student_data)
                
                # Format prediction value safely
                if isinstance(predicted_marks, (np.ndarray, list)):
                    val = float(predicted_marks[0])
                else:
                    val = float(predicted_marks)

                st.session_state.predicted_marks = round(val, 2)
                st.session_state.performance = get_performance_category(st.session_state.predicted_marks)
                st.session_state.risk = get_risk_level(st.session_state.predicted_marks)
                st.session_state.recommendations = get_recommendations(st.session_state.predicted_marks)

    with right_col:
        st.info("""
### Prediction Tips

✔ Enter realistic values

✔ All fields are required

✔ Click Predict button

✔ Receive AI insights
""")

    if st.session_state.predicted_marks is not None:
        st.divider()
        st.subheader("🎯 Prediction Result")

        result1, result2, result3 = st.columns(3)
        with result1:
            st.metric("Predicted Marks", f"{st.session_state.predicted_marks}")
        with result2:
            st.metric("Performance", st.session_state.performance)
        with result3:
            st.metric("Risk Level", st.session_state.risk)

        st.success("Prediction completed successfully.")

        st.subheader("📌 Recommendations")
        for rec in st.session_state.recommendations:
            st.write("✅", rec)

# ---------------------------------------------------
# Page 3: Dataset Insights (All Graphs Preserved)
# ---------------------------------------------------
elif page == "📈 Dataset Insights":
    st.title("📈 Dataset Insights")
    st.markdown("Explore the student performance dataset using interactive visualizations and statistics.")
    st.divider()

    # Metrics Overview
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📊 Total Records", df.shape[0])
    with c2:
        st.metric("📑 Total Features", df.shape[1])
    with c3:
        st.metric("❌ Missing Values", df.isnull().sum().sum())
    with c4:
        st.metric("🗂 Duplicate Rows", df.duplicated().sum())

    st.divider()

    # Data Preview & Info
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    st.subheader("📋 Dataset Information")
    info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(info, use_container_width=True)

    st.divider()

    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.divider()

    # ------------------ Visualizations ------------------
    st.header("📊 Exploratory Data Analysis")

    # Row 1: Gender & Attendance
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👨‍🎓 Gender Distribution")
        gender_count = df["Gender"].value_counts().reset_index()
        gender_count.columns = ["Gender", "Count"]
        fig = px.bar(
            gender_count,
            x="Gender",
            y="Count",
            color="Gender",
            text="Count",
            title="Gender Distribution"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Attendance Distribution")
        fig = px.histogram(
            df,
            x="Attendance_Percent",
            nbins=20,
            title="Attendance Percentage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Study Hours & Previous Marks
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📚 Study Hours Distribution")
        fig = px.histogram(
            df,
            x="Study_Hours",
            nbins=15,
            title="Study Hours Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("📝 Previous Marks Distribution")
        fig = px.histogram(
            df,
            x="Previous_Marks",
            nbins=20,
            title="Previous Marks Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Final Marks & Sleep Hours
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🎯 Final Marks Distribution")
        fig = px.histogram(
            df,
            x="Final_Marks",
            nbins=20,
            title="Final Marks Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("😴 Sleep Hours Distribution")
        fig = px.histogram(
            df,
            x="Sleep_Hours",
            nbins=15,
            title="Sleep Hours Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Internet Access & Extracurriculars
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("🌐 Internet Access")
        fig = px.bar(
            df["Internet_Access"].value_counts().reset_index(),
            x="Internet_Access",
            y="count",
            color="Internet_Access",
            title="Internet Access Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        st.subheader("🏃 Extracurricular Activities")
        fig = px.bar(
            df["Extracurricular"].value_counts().reset_index(),
            x="Extracurricular",
            y="count",
            color="Extracurricular",
            title="Extracurricular Activities"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 5: Scatter Plots (Relationships with Final Marks)
    st.divider()
    st.header("📈 Feature Relationships with Final Marks")

    col9, col10 = st.columns(2)
    with col9:
        st.subheader("📈 Attendance vs Final Marks")
        fig = px.scatter(
            df,
            x="Attendance_Percent",
            y="Final_Marks",
            color="Gender",
            title="Attendance vs Final Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col10:
        st.subheader("📚 Study Hours vs Final Marks")
        fig = px.scatter(
            df,
            x="Study_Hours",
            y="Final_Marks",
            color="Gender",
            title="Study Hours vs Final Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

    col11, col12 = st.columns(2)
    with col11:
        st.subheader("📝 Previous Marks vs Final Marks")
        fig = px.scatter(
            df,
            x="Previous_Marks",
            y="Final_Marks",
            color="Gender",
            title="Previous Marks vs Final Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col12:
        st.subheader("📄 Assignment Score vs Final Marks")
        fig = px.scatter(
            df,
            x="Assignment_Score",
            y="Final_Marks",
            color="Gender",
            title="Assignment Score vs Final Marks"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quiz Score Scatter Plot
    st.subheader("🧪 Quiz Score vs Final Marks")
    fig = px.scatter(
        df,
        x="Quiz_Score",
        y="Final_Marks",
        color="Gender",
        title="Quiz Score vs Final Marks"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Debugging / Model Structure Info (Shown neatly inside code containers)
    st.divider()
    with st.expander("🛠️ View Pipeline Debug Diagnostics"):
        lr_model = model.named_steps["model"]
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        st.write("**Model Type:**", type(lr_model))
        st.write("**Number of Features:**", len(feature_names))
        st.write("**Number of Coefficients:**", len(lr_model.coef_))

        st.write("**Feature Names:**")
        st.json(list(feature_names))

        st.write("**Coefficients:**")
        st.json(list(lr_model.coef_))

# ---------------------------------------------------
# Page 4: Feature Importance
# ---------------------------------------------------
elif page == "📉 Feature Importance":
    st.title("📉 Feature Importance")
    st.markdown("Feature importance based on the trained Linear Regression model.")
    st.divider()

    # Extract pipeline features & model safely
    lr_model = model.named_steps["model"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    st.write("Number of Features:", len(feature_names))
    st.write("Number of Coefficients:", len(lr_model.coef_))

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr_model.coef_
    })

    importance["Absolute"] = importance["Coefficient"].abs()
    importance = importance.sort_values(
        by="Absolute",
        ascending=False
    )

    fig = px.bar(
        importance,
        x="Absolute",
        y="Feature",
        orientation="h",
        color="Coefficient",
        title="Feature Importance (Absolute Coefficients)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Importance Table")
    st.dataframe(
        importance,
        use_container_width=True
    )

# ---------------------------------------------------
# Page 5: Model Performance
# ---------------------------------------------------
elif page == "📋 Model Performance":

    st.title("📋 Model Performance & Evaluation")
    st.markdown("""
    Evaluate the accuracy and metrics of the trained **Linear Regression Model** on the student performance dataset.
    """)
    st.divider()

    # 1. Prepare Features (X) and Target (y)
    X = df.drop(columns=["Final_Marks"])
    y_true = df["Final_Marks"]

    # 2. Make Predictions using the loaded pipeline
    y_pred = model.predict(X)

    # 3. Calculate Evaluation Metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # ---------------------------------------------------
    # Key Performance Indicators (KPI Metrics)
    # ---------------------------------------------------
    st.subheader("🎯 Key Evaluation Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric(label="📈 R² Score", value=f"{r2:.4f}", help="Proportion of variance explained by the model (closer to 1.0 is better).")
    with m2:
        st.metric(label="📉 MAE (Mean Absolute Error)", value=f"{mae:.2f}", help="Average magnitude of errors in predictions.")
    with m3:
        st.metric(label="📐 MSE (Mean Squared Error)", value=f"{mse:.2f}")
    with m4:
        st.metric(label="📊 RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}", help="Standard deviation of residuals (in the same units as Final Marks).")

    st.divider()

    # ---------------------------------------------------
    # Performance Visualizations
    # ---------------------------------------------------
    st.subheader("📊 Performance Visualizations")

    col_plot1, col_plot2 = st.columns(2)

    # Plot 1: Actual vs Predicted Marks
    with col_plot1:
        eval_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        
        fig_actual_vs_pred = px.scatter(
            eval_df,
            x="Actual",
            y="Predicted",
            opacity=0.6,
            title="🎯 Actual vs. Predicted Final Marks",
            labels={"Actual": "Actual Final Marks", "Predicted": "Predicted Final Marks"}
        )
        # Ideal prediction diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig_actual_vs_pred.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="Red", dash="dash", width=2)
        )
        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

    # Plot 2: Residual Plot (Error Distribution)
    with col_plot2:
        eval_df["Residuals"] = y_true - y_pred
        
        fig_residuals = px.histogram(
            eval_df,
            x="Residuals",
            nbins=30,
            title="📉 Residuals Distribution (Errors)",
            labels={"Residuals": "Error (Actual - Predicted)"},
            color_discrete_sequence=["#FF6B6B"]
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # Model Pipeline Summary
    # ---------------------------------------------------
    st.subheader("⚙️ Pipeline Architecture")
    st.markdown("Below is the structure of the Scikit-Learn pipeline used for preprocessing and prediction:")
    st.code(str(model), language="text")
# ---------------------------------------------------
# Page 6: Goal Planner
# ---------------------------------------------------
elif page == "🎯 Goal Planner":

    st.title("🎯 Goal Planner")

    st.markdown("""
Set your target marks and receive realistic improvement suggestions.
""")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="goal_gender"
        )

        study_hours = st.slider(
            "Study Hours",
            0.0,
            12.0,
            6.0,
            0.5,
            key="goal_study"
        )

        attendance = st.slider(
            "Attendance %",
            50,
            100,
            80,
            key="goal_att"
        )

        previous_marks = st.slider(
            "Previous Marks",
            30,
            95,
            70,
            key="goal_prev"
        )

        assignment = st.slider(
            "Assignment Score",
            40,
            100,
            75,
            key="goal_assign"
        )

    with col2:

        quiz = st.slider(
            "Quiz Score",
            30,
            100,
            70,
            key="goal_quiz"
        )

        internal = st.slider(
            "Internal Marks",
            10,
            30,
            22,
            key="goal_internal"
        )

        internet = st.selectbox(
            "Internet Access",
            ["Yes","No"],
            key="goal_net"
        )

        extra = st.selectbox(
            "Extracurricular",
            ["Yes","No"],
            key="goal_extra"
        )

        sleep = st.slider(
            "Sleep Hours",
            4.0,
            10.0,
            7.0,
            0.5,
            key="goal_sleep"
        )

    target = st.slider(
        "🎯 Target Marks",
        40,
        100,
        85
    )

    if st.button("Generate Goal Plan"):

        student = pd.DataFrame({

            "Gender":[gender],
            "Study_Hours":[study_hours],
            "Attendance_Percent":[attendance],
            "Previous_Marks":[previous_marks],
            "Assignment_Score":[assignment],
            "Quiz_Score":[quiz],
            "Internal_Marks":[internal],
            "Internet_Access":[internet],
            "Extracurricular":[extra],
            "Sleep_Hours":[sleep]

        })

        current = float(model.predict(student)[0])

        current = max(0, min(100, current))

        st.subheader("Current Prediction")

        st.metric(
            "Predicted Marks",
            f"{current:.2f}"
        )

        if current >= target:

            st.success("🎉 Congratulations! You are already achieving your target.")

        else:

            gap = target - current

            st.warning(f"You need approximately **{gap:.1f}** more marks.")

            improvements = []

            if study_hours < 8:
                improvements.append(
                    f"📚 Increase Study Hours from **{study_hours:.1f} → {min(8,study_hours+2):.1f} hrs/day**"
                )

            if attendance < 95:
                improvements.append(
                    f"🏫 Increase Attendance from **{attendance}% → {min(95,attendance+10)}%**"
                )

            if assignment < 95:
                improvements.append(
                    f"📝 Improve Assignment Score from **{assignment} → {min(95,assignment+10)}**"
                )

            if quiz < 95:
                improvements.append(
                    f"🧪 Improve Quiz Score from **{quiz} → {min(95,quiz+10)}**"
                )

            if internal < 28:
                improvements.append(
                    f"📖 Improve Internal Marks from **{internal} → {min(28,internal+4)}**"
                )

            # Sleep recommendation (realistic)
            if sleep < 7:
                improvements.append(
                    f"😴 Maintain **7–8 hours** sleep daily."
                )
            elif sleep > 8.5:
                improvements.append(
                    "😴 Sleeping more than 8.5 hours usually doesn't improve marks significantly. Maintain around **7–8 hours**."
                )

            if internet == "No":
                improvements.append(
                    "🌐 Use online learning resources whenever possible."
                )

            if extra == "Yes":
                improvements.append(
                    "⚖ Balance extracurricular activities with study time."
                )

            st.subheader("Recommended Improvements")

            for item in improvements:
                st.write("✅", item)

            # Estimated realistic score
            estimated = current

            estimated += min(2, max(0, 8-study_hours))*2
            estimated += min(5, (95-attendance)/10)
            estimated += min(3, (95-assignment)/10)
            estimated += min(3, (95-quiz)/10)
            estimated += min(2, (28-internal)/4)

            estimated = min(100, estimated)

            st.divider()

            st.metric(
                "Estimated Achievable Score",
                f"{estimated:.2f}"
            )

            if estimated >= target:
                st.success("🎯 Target looks achievable with these improvements.")
            else:
                st.info("💡 Target is ambitious. Continue improving consistently over time.")
        
# ---------------------------------------------------
# Page 7: About
elif page == "ℹ️ About":

    # =========================================================
    # ABOUT PROJECT
    # =========================================================

    st.title("ℹ️ About Project")

    st.markdown("""
    ## 🎓 Student Performance Prediction & Analytics System

    An **AI-powered Machine Learning dashboard** designed to analyze
    student academic data, predict final marks, and provide useful
    performance insights and recommendations.
    """)

    st.divider()

    # =========================================================
    # PROJECT OVERVIEW
    # =========================================================

    st.subheader("📌 Project Overview")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
        This project uses **Machine Learning and Data Analytics** to
        understand the factors that influence student academic performance.

        The system allows users to:

        - 📊 Explore student datasets
        - 🔍 Analyze academic performance
        - 🤖 Predict final marks using Machine Learning
        - 📈 Visualize important patterns and relationships
        - 📉 Analyze feature coefficients
        - 🎯 Set academic goals
        - 💡 Receive personalized recommendations
        """)

    with col2:

        st.info("""
        ### 🎯 Project Goal

        Build an easy-to-use analytics system that helps understand
        student performance and provides data-driven academic insights.
        """)

    st.divider()

    # =========================================================
    # TECHNOLOGY STACK
    # =========================================================

    st.subheader("🛠️ Technology Stack")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("""
        ### 🐍 Python

        Used for data processing, machine learning,
        prediction logic and application development.
        """)

    with tech2:
        st.markdown("""
        ### 🤖 Scikit-Learn

        Used for preprocessing, pipelines,
        model training and prediction.
        """)

    with tech3:
        st.markdown("""
        ### 📊 Pandas

        Used for dataset loading,
        cleaning and data analysis.
        """)

    with tech4:
        st.markdown("""
        ### 📈 Plotly

        Used to create interactive
        charts and visualizations.
        """)

    st.divider()

    # =========================================================
    # MACHINE LEARNING PIPELINE
    # =========================================================

    st.subheader("🤖 Machine Learning Pipeline")

    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:

        st.markdown("""
        **1️⃣ Data Collection**

        Student academic and personal performance data.

        **2️⃣ Data Preprocessing**

        Numerical features are scaled using `StandardScaler`
        and categorical features are encoded using
        `OneHotEncoder`.

        **3️⃣ Feature Processing**

        Important academic features such as:

        - Previous Marks
        - Attendance
        - Study Hours
        - Assignment Score
        - Quiz Score
        - Internal Marks
        - Sleep Hours
        """)

    with pipeline_col2:

        st.markdown("""
        **4️⃣ Machine Learning Model**

        The trained model uses a Scikit-Learn pipeline
        with **Linear Regression**.

        **5️⃣ Prediction**

        The system predicts the student's expected
        final marks.

        **6️⃣ Insights**

        The prediction is converted into:

        - Performance Category
        - Risk Level
        - Personalized Recommendations
        """)

    st.divider()

    # =========================================================
    # DATASET INFORMATION
    # =========================================================

    st.subheader("📊 Dataset Information")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "👨‍🎓 Student Records",
            f"{len(df):,}"
        )

    with d2:
        st.metric(
            "📋 Features",
            df.shape[1]
        )

    with d3:
        st.metric(
            "🎯 Target",
            "Final Marks"
        )

    with d4:
        st.metric(
            "🤖 Model",
            "Linear Regression"
        )

    st.divider()

    # =========================================================
    # KEY FEATURES
    # =========================================================

    st.subheader("✨ Key Features")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        st.markdown("""
        ### 📊 Dataset Insights

        Explore the dataset through:

        - Statistical summaries
        - Distribution charts
        - Interactive scatter plots
        - Dataset information
        """)

    with feature2:

        st.markdown("""
        ### 🎯 Student Prediction

        Enter student information and generate
        an estimated final marks prediction using
        the trained Machine Learning pipeline.
        """)

    with feature3:

        st.markdown("""
        ### 📉 Feature Analysis

        Analyze model coefficients to understand
        which processed features have greater
        influence on the prediction.
        """)

    st.divider()

    # =========================================================
    # PROJECT OBJECTIVE
    # =========================================================

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Use Machine Learning to predict student final marks.",
        "Understand the relationship between academic factors and performance.",
        "Provide interactive data visualization and analytics.",
        "Identify important features affecting predictions.",
        "Provide performance and risk-level insights.",
        "Create a simple and user-friendly academic analytics dashboard."
    ]

    for objective in objectives:
        st.markdown(f"✅ {objective}")

    st.divider()

    # =========================================================
    # APPLICATION MODULES
    # =========================================================

    st.subheader("🧩 Application Modules")

    modules = pd.DataFrame({
        "Module": [
            "🏠 Home",
            "📊 Student Prediction",
            "📈 Dataset Insights",
            "📉 Feature Importance",
            "📋 Model Performance",
            "🎯 Goal Planner",
            "ℹ️ About"
        ],
        "Purpose": [
            "Project dashboard and summary",
            "Predict student's final marks",
            "Explore and visualize dataset",
            "Analyze model coefficients",
            "View model evaluation results",
            "Set and analyze academic goals",
            "Project information and technology"
        ]
    })

    st.dataframe(
        modules,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================================================
    # DISCLAIMER
    # =========================================================

    st.subheader("⚠️ Important Note")

    st.warning("""
    The predicted marks are generated using a trained Machine Learning
    model and should be considered an **estimated result**, not an
    official academic grade.

    Actual student performance may depend on several additional factors
    that are not included in the dataset.
    """)

    st.divider()

    # =========================================================
    # PROJECT FOOTER
    # =========================================================

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:20px;
            border-radius:10px;
            background-color:rgba(128,128,128,0.08);
        ">

        <h3>🎓 Student Performance Prediction System</h3>

        <p>
        AI • Machine Learning • Data Analytics • Streamlit
        </p>

        <p>
        Built as an academic Machine Learning project.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )