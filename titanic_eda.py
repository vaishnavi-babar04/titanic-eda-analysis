# Exploratory Data Analysis (EDA) on Titanic Dataset

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")
# Display First 5 Rows
print("First 5 Rows:")
print(df.head())
# Dataset Information
print("\nDataset Information:")
print(df.info())
# Dataset Shape
print("\nDataset Shape:", df.shape)
# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())
# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())
# -----------------------------
# Data Cleaning
# -----------------------------
# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())
# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Drop Cabin column (too many missing values)
df.drop("Cabin", axis=1, inplace=True)
# Check after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# -----------------------------
# Visualization 1: Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# -----------------------------
# Visualization 2: Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# Visualization 3: Survival by Gender
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# -----------------------------
# Visualization 4: Passenger Class
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# -----------------------------
# Visualization 5: Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------

# Convert Gender into Numeric Values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Select Numeric Columns
numeric_data = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Insights
# -----------------------------
print("\n----- Insights -----")
print("1. Female passengers had a higher survival rate.")
print("2. First-class passengers survived more often.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Passenger class influenced survival chances.")
print("5. Gender was an important factor affecting survival.")