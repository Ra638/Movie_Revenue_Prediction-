# Movie Data Analysis

# Problem Statement:
# This script analyzes a dataset of movies to understand relationships between budget, revenue, genre, and other factors.
# It explores correlations, visualizes key metrics, and identifies trends in movie performance.

# Importing necessary libraries
# pandas - for data manipulation
# numpy - for numerical operations
# seaborn - for data visualization
# matplotlib - for plotting graphs

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

# Load the dataset
df = pd.read_csv(r"C:\Users\Rayan\OneDrive\Documents\movies.csv")

# Display basic information about the dataset
df.head()
df.describe()

# Check for missing values in each column
for col in df.columns:
    pct_missing = round(np.mean(df[col].isnull()), 2)
    print(f'{col} - {pct_missing}%')

# Fill missing values in budget and gross columns with 0 and convert to integer
df['budget'] = df['budget'].fillna(0).astype('int64')
df['gross'] = df['gross'].fillna(0).astype('int64')

# Extract year from the released column
df['Correct_year'] = df['released'].astype(str).str.extract(r'(\d{4})')
print(df[['Correct_year', 'released']].head())

# Sort dataset by gross revenue
df_sort = df.sort_values(by=['gross'], ascending=False)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Scatter plot: Budget vs Gross Earnings
plt.figure(figsize=(10,6))
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget Vs Gross Earnings')
plt.xlabel('Movie Budget')
plt.ylabel('Total Gross Revenue')
plt.show()

# Regression plot: Budget vs Gross Earnings
sns.regplot(x='budget', y='gross', data=df_sort, 
            scatter_kws={"color": "red", "s": 50, "marker": "o"}, 
            line_kws={"color": "blue", "linewidth": 2})
plt.show()

# Convert year columns to numeric type
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['Correct_year'] = pd.to_numeric(df['Correct_year'], errors='coerce')

# Compute correlation matrix and plot heatmap
correlation_matrix = df.corr(numeric_only=True, method='pearson')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Correlation analysis by genre
df_genre_test = df.groupby('genre')[['budget', 'gross']].corr().unstack().iloc[:, [1, 2]]
df_genre_test.dropna(inplace=True)
df_genre_test_sorted = df_genre_test.sort_values(by=('budget', 'gross'), ascending=False)
print(df_genre_test_sorted)

# Compute ROI (Return on Investment) and analyze by genre
df = df[df['budget'] > 0]  # Remove movies with zero budget
df['ROI'] = (df['gross'] - df['budget']) / df['budget']
df_roi = df.groupby('genre')['ROI'].mean().sort_values(ascending=False)
print(df_roi)

# Bar plot: Average ROI per Genre
plt.figure(figsize=(12,6))
sns.barplot(x=df_roi.values, y=df_roi.index, palette="coolwarm")
plt.xlabel("Average ROI")
plt.ylabel("Genre")
plt.title("Average ROI per Genre")
plt.show()

# Heatmap: Budget vs Gross by Genre
plt.figure(figsize=(12,6))
sns.heatmap(df_genre_test, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Budget Vs Gross By Genre")
plt.show()

# Convert categorical columns to numeric values for correlation analysis
df_numerized = df.copy()
for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].fillna('Unknown').astype('category').cat.codes

# Compute and visualize correlation matrix of numeric data
correlation_matrix_ = df_numerized.corr()
plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix_, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xlabel("Movie Feature")
plt.ylabel("Movie Feature")
plt.title("Feature Correlation Matrix")
plt.show()

# Find and display top correlations
correlation_unstacked = correlation_matrix_.unstack().sort_values(ascending=False)
correlation_unstacked = correlation_unstacked[correlation_unstacked < 1]  # Remove self-correlations
print("Top 10 Strongest Positive Correlations")
print(correlation_unstacked.head(12).iloc[2:])
print("\nTop 10 Strongest Negative Correlations")
print(correlation_unstacked.tail(10))

# Boxplot: Budget, Gross, and ROI
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['budget', 'gross', 'ROI']])
plt.title("Boxplot of Budget, Gross, and ROI")
plt.show()

# Log-scale Regression plot: Budget vs Gross
plt.figure(figsize=(12,6))
sns.regplot(x=np.log1p(df['budget']), y=np.log1p(df['gross']), data=df,
            scatter_kws={"color":"red"}, line_kws={"color":"blue"})
plt.xlabel('Log(Budget)')
plt.ylabel('Log(Gross Revenue)')
plt.title("Budget vs Gross Earnings (Log Scale)")
plt.show()

# Bar plot: ROI by Movie Rating
plt.figure(figsize=(12,6))
sns.barplot(x=df['rating'], y=df['ROI'])
plt.title("ROI by Movie Rating")
plt.xlabel("Movie Rating")
plt.ylabel("Return on Investment (ROI)")
plt.xticks(rotation=45)
plt.show()

# Line plot: Average Gross Revenue Over Time
df_year = df.groupby('Correct_year')['gross'].mean()
plt.figure(figsize=(12,6))
sns.lineplot(x=df_year.index, y=df_year.values)
plt.title("Average Movie Gross Revenue Over Time")
plt.xlabel("Year")
plt.ylabel("Average Gross Revenue")
plt.show()
