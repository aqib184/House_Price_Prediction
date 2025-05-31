import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data.csv")

# Feature engineering
df['year_sold'] = pd.to_datetime(df['date']).dt.year
df['house_age'] = df['year_sold'] - df['yr_built']
df['has_been_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df.drop(['date', 'yr_renovated', 'yr_built', 'street', 'country'], axis=1, inplace=True)

# Remove outliers
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

columns_to_clean = ['price', 'sqft_living', 'bathrooms', 'bedrooms', 'sqft_above', 'sqft_basement', 'sqft_lot']
for col in columns_to_clean:
    df = remove_outliers_iqr(df, col)

# Train/test split
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'house_age']
categorical_features = ['waterfront', 'view', 'condition', 'city', 'statezip', 'has_been_renovated']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)

# Prediction page
def page_predict():
    st.header("House Price Prediction")
    user_input = {}
    for feature in numeric_features:
        user_input[feature] = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
    for feature in categorical_features:
        options = list(df[feature].unique())
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")

# Data overview page
def page_data():
    st.header("Data Overview")
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.subheader("Dataset Info")
    import io  # Add at the top of your script

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Dataset Shape")
    st.write(df.shape)

# Visualization page
def page_visuals():
    st.header("Data Visualizations")

    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, bins=50, color='skyblue', ax=ax)
    st.pyplot(fig)

    st.subheader("Bedrooms Count")
    fig, ax = plt.subplots()
    sns.countplot(x='bedrooms', data=df, palette='pastel', ax=ax)
    st.pyplot(fig)

    st.subheader("Bathrooms Count")
    fig, ax = plt.subplots()
    sns.countplot(x='bathrooms', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("Price vs. Sqft Living")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sqft_living', y='price', data=df, alpha=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Cities with Most Listings")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_cities = df['city'].value_counts().nlargest(10).index
    sns.countplot(data=df[df['city'].isin(top_cities)], x='city', order=top_cities, palette="crest", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr(numeric_only=True)
    top_corr = corr_matrix['price'].abs().sort_values(ascending=False).head(11)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[top_corr.index].corr(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Main app
st.sidebar.title("Navigation")
pages = {
    "Predict Price": page_predict,
    "Data Overview": page_data,
    "Visualizations": page_visuals
}
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
