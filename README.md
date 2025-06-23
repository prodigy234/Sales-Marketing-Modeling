
# 🛒 Supermarket Sales Analytics Dashboard

This project presents an end-to-end **interactive analytics dashboard** built using **Streamlit** to analyze supermarket sales data. It includes **exploratory analysis**, **customer segmentation**, **churn prediction**, and **feature importance insights**, all aimed at driving informed retail business decisions.

---

This well detailed Sales Modeling Analytics Dashboard which I developed for a supermarket can be accessed live on streamlit [Here](https://salesmodelling.streamlit.app/)

---

## 📬 Author

**Gbenga Kajola**
🎓 Certified Data Analyst | 👨‍💻 Certified Data Scientist | 🧠 AI/ML Engineer | 📱 Mobile App Developer 

[LinkedIn](https://www.linkedin.com/in/kajolagbenga)

[Portfolio](https://kajolagbenga.netlify.app)

[Certified_Data_Scientist](https://www.datacamp.com/certificate/DSA0012312825030)

[Certified_Data_Analyst](https://www.datacamp.com/certificate/DAA0018583322187)

[Certified_SQL_Database_Programmer](https://www.datacamp.com/certificate/SQA0019722049554)


---

## 📁 Dataset Overview

- **Rows**: 1000
- **Columns**: 17
- **Missing Values**: 0
- **Data Types**: Categorical and Numerical
- **Source**: `supermarket_sales.csv` (uploaded)

### Sample Columns:
- `Invoice ID`, `Branch`, `City`, `Customer type`, `Gender`, `Product line`, `Total`, `Payment`, `Date`, `gross income`

---

## 🧭 Navigation Sections

### 1. 📄 Dataset Overview
- Preview the first few rows
- Data types and shape
- Missing value analysis
- Descriptive statistics summary

### 2. 🔍 Exploratory Data Analysis
- Gender distribution and purchase behavior
- Product line analysis
- Total purchase distribution across different segments

### 3. 🧮 Customer Segmentation
- **K-Means Clustering** used to segment customers based on:
  - `Total`
  - `Quantity`
  - `Gross Income`
- **Silhouette Score** helps validate cluster cohesion
- Segments:
  - **Cluster 0**: Low spenders (discount-driven)
  - **Cluster 1**: Mid-level buyers (potential upsell)
  - **Cluster 2**: High-value customers (priority)

### 4. 📉 Churn Prediction
- Used **Random Forest Classifier** with hyperparameter tuning via **GridSearchCV**
- Predicts customer types (proxy for churn) using features like `Total`, `COGS`, `Gross Income`, etc.
- Outputs:
  - Classification report
  - Confusion matrix heatmap
  - ROC AUC score

### 5. 📌 Feature Importance
- Identifies the top influential features for predicting customer type
- Visualized with bar plots for intuitive interpretation

### 6. 👨‍💻 About Developer
- Name: **Kajola Gbenga**
- Roles: Certified Data Analyst, Data Scientist, SQL Programmer, App Developer
- Links:
  - [LinkedIn](https://www.linkedin.com/in/kajolagbenga)
  - [GitHub](https://github.com/prodigy234)
  - [Portfolio](https://kajolagbenga.netlify.app/)

---

## 📥 Downloadable Report

- **Format**: Microsoft Word (`.docx`)
- **Content Includes**:
  - Key Insights from each module
  - Cluster analysis
  - Feature insights
  - Business recommendations
  - Visual explanations

---

## 📦 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Word Report (Python-docx)

---

## 🧠 Business Recommendations

- **Target Cluster 2**: Drive loyalty with premium offers
- **Offer Discounts to Cluster 0**: Price-sensitive audience
- **Upsell Cluster 1**: Moderate spenders
- **Marketing on High-performing Lines**: Focus on "Food & Beverages", "Health & Beauty"
- **Leverage Churn Prediction**: Prioritize retention for member-type customers

---

## 🏁 How to Run This App

```bash
pip install -r requirements.txt
streamlit run supermarket_dashboard.py
```

---

## 📃 License

This project is open-source and free to use. For contributions or inquiries, contact the developer.

