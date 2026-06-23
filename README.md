# 🎓 UNI-BRIDGE: AI Study Abroad Recommender

> An AI-powered study abroad recommendation system that combines Machine Learning and Generative AI to provide personalized country recommendations, university insights, cost analysis, and admission guidance.

---

## 📌 Overview

UNI-BRIDGE helps students identify the most suitable study-abroad destinations based on their academic profile, budget, and preferences.

The system uses Machine Learning models to predict the best-fit countries and Large Language Models (LLMs) to generate detailed explanations and guidance for each recommendation.

---

## ✨ Features

- 🎯 Personalized country recommendations
- 🤖 Machine Learning-based prediction engine
- 📊 Confidence-based country ranking
- 💰 Budget-aware recommendations
- 🌍 Region preference matching
- 🏫 University and admission insights
- 🧠 AI-generated explanations using Hugging Face LLMs
- 📈 Interactive visualizations
- 🌙 Dark/Light Mode support
- 💻 Streamlit-based web application

---

## 🏗️ Project Architecture

```text
User Input
    │
    ▼
Student Profile
(GPA, IELTS, Budget, Region, Degree)
    │
    ▼
Feature Engineering
    │
    ▼
Machine Learning Model
(Random Forest)
    │
    ▼
Top Country Recommendations
    │
    ▼
Hugging Face LLM
    │
    ▼
Detailed AI Insights
    │
    ▼
Streamlit Web Application
```

---

## 📂 Dataset Sources

### QS World University Rankings Dataset
- University rankings
- Academic reputation
- Country-wise education quality

### Cost of International Education Dataset
- Tuition fees
- Living expenses
- Affordability metrics

### Synthetic Student Dataset
- GPA
- IELTS Score
- Budget
- Preferred Region
- Degree Level
- Field of Study

---

## ⚙️ Methodology

1. Collected QS World University Rankings and Cost of International Education datasets from Kaggle.
2. Cleaned and preprocessed both datasets by handling missing values and standardizing data.
3. Created a realistic synthetic student dataset containing academic, financial, and preference-related attributes.
4. Merged all datasets into a unified recommendation dataset.
5. Performed feature engineering and data preprocessing.
6. Trained and evaluated multiple machine learning models.
7. Selected Random Forest as the final model based on performance metrics.
8. Integrated Hugging Face LLM API to generate personalized explanations.
9. Built and deployed the application using Streamlit.

---

## 🛠️ Tech Stack

### Programming & Data Science
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

### AI & Machine Learning
- Random Forest
- Hugging Face Inference API
- Large Language Models (LLMs)

### Visualization & Deployment
- Plotly
- Streamlit
- Custom CSS

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/UNI-BRIDGE-AI-Study-Abroad-Recommender.git

cd UNI-BRIDGE-AI-Study-Abroad-Recommender
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_api_key
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📊 Model Information

| Feature | Value |
|----------|----------|
| Final Model | Random Forest |
| Evaluation Metric | F1 Score |
| Validation Method | Cross Validation |
| Countries Supported | 58+ |
| Frontend | Streamlit |
| AI Integration | Hugging Face LLM |

---

# 📸 Application Screenshots

> Store screenshots inside a folder named **screenshots/**

## 🏠 Home Page

```markdown
![Home Page](screenshots/Hpage%20dark.png)
```

## 📋 Student Profile Form

```markdown
![Student Profile](screenshots/student_profile.png)
```

## 🌍 Country Recommendations

```markdown
![Recommendations](screenshots/recommendations.png)
```

## 🤖 AI Insights

```markdown
![AI Insights](screenshots/ai_insights.png)
```

## 🌙 Dark Mode

```markdown
![Dark Mode](screenshots/dark_mode.png)
```

## 📊 Analytics Dashboard

```markdown
![Analytics](screenshots/analytics.png)
```

---

## 🔮 Future Scope

- Real-world student datasets
- University-level recommendations
- Scholarship recommendations
- Visa success prediction
- Job market insights
- Mobile application development
- User feedback-based learning

---

## 👨‍💻 Contributors

- Durva Palkar

---

## 📜 License

This project is developed for educational and research purposes.

---

⭐ If you found this project useful, please consider giving it a star!
