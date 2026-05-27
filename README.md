#  **Black Friday Sales Analysis & Data Preprocessing**


<p align="center">
  <img src="https://media.licdn.com/dms/image/v2/D5612AQGCK2JQa876Ig/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1726987846519?e=2147483647&v=beta&t=ZI1sigAfuGbYCJNpwi1G3SG5dlMLQ-B_0NHVMbEffY0" 
       alt="Black Friday Sales Analysis" 
       width="800"/>
</p>


##  Project Overview

Black Friday marks the beginning of the major holiday shopping season, where retailers experience extremely high sales volumes. Understanding customer purchasing behavior during this period helps businesses optimize pricing strategies, inventory planning, and marketing efforts.

This project focuses on performing detailed **Exploratory Data Analysis (EDA)** and **data preprocessing** on historical Black Friday sales data. The main objective of this phase is to clean, transform, and prepare the dataset for future machine learning modeling.

This project is part of my **100 Days of Machine Learning** journey.



##  Dataset Description

The dataset contains transactional records from a retail store during Black Friday sales.

### Key Features:

- `User_ID`
- `Product_ID`
- `Gender`
- `Age`
- `Occupation`
- `City_Category`
- `Stay_In_Current_City_Years`
- `Marital_Status`
- `Product_Category_1`
- `Product_Category_2`
- `Product_Category_3`
- `Purchase` (Target Variable)

- Total Records: 500,000+  
- Total Columns: 12  
- Target Variable: **Purchase**


##  **Exploratory Data Analysis (EDA)**

### 1️⃣ Data Understanding
- Checked dataset shape and structure  
- Reviewed data types  
- Identified missing values  
- Checked duplicate records  

### 2️⃣ Missing Value Analysis
- Identified null values in `Product_Category_2` and `Product_Category_3`  
- Visualized missing data using heatmaps  
- Calculated percentage of missing values  

### 3️⃣ Target Variable Analysis
- Analyzed distribution of `Purchase`  
- Checked skewness and kurtosis  
- Visualized outliers using boxplots  

### 4️⃣ Customer Behavior Insights

####  Gender
- Male customers contributed a larger share of total purchases.  
- Average spending differed between genders.

#### Age Group
- Customers aged 26–35 showed higher purchasing activity.  
- Other age groups showed comparatively lower spending patterns.

####  Occupation
- Certain occupations demonstrated higher average purchase amounts.

####  City Category
- Spending behavior varied across city categories.

---

##  **Data Preprocessing**

### 1️⃣ Handling Missing Values
- Filled missing values in `Product_Category_2` and `Product_Category_3`  
- Ensured dataset contains no null values before modeling  

### 2️⃣ Removing Irrelevant Columns
- Dropped `User_ID` and `Product_ID` as they do not contribute to prediction  

### 3️⃣ Outlier Treatment
- Used the **IQR (Interquartile Range)** method to detect outliers  
- Applied capping technique to reduce the impact of extreme values



## **Feature Engineering**

### 🔢 Encoding Categorical Variables

- **Label Encoding** applied to:
  - `Gender`

- **Ordinal Mapping** applied to:
  - `Age`

- **One-Hot Encoding** applied to:
  - `City_Category`

- Cleaned and converted:
  - `Stay_In_Current_City_Years` into numeric format



##  **Feature Scaling**

- Applied **StandardScaler** to normalize numerical features  
- Ensured features are centered and scaled  
- Prepared dataset for future regression modeling



## **Technologies Used**

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook


##  Project Status

✅ Data Cleaning Completed  
✅ Exploratory Data Analysis Completed  
✅ Missing Value Treatment Completed  
✅ Feature Engineering Completed  
✅ Feature Scaling Completed  

🔜 Next Step: Applying Regression Models to Predict Purchase Amount  


## 📂 Repository Structure 
Black-Friday-Sales/
│
├── notebook.ipynb
├── dataset.csv
└── README.md
