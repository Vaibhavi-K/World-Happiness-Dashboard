import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="World Happiness Dashboard", layout="wide")

# Load the uploaded dataset
@st.cache_data
def load_data():
    df = pd.read_csv("world_happiness.csv")
    return df

df = load_data()

# Handle column renaming if needed
df.columns = df.columns.str.strip()

st.title("World Happiness Report Dashboard")

# Show dataset preview
st.subheader(" Dataset Preview")
st.dataframe(df.head())

# 1. Top 10 Happiest Countries (Bar Chart)
st.subheader(" Top 10 Happiest Countries")
top_10 = df.sort_values(by="Score", ascending=False).head(10)
fig1, ax1 = plt.subplots()
sns.barplot(x="Score", y="Country or region", data=top_10)
ax1.set_title("Top 10 Happiest Countries")
st.pyplot(fig1)

# 2. GDP vs Happiness Score (Scatter Plot)
st.subheader("GDP per Capita vs Happiness Score")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="GDP per capita", y="Score")
ax2.set_title("GDP vs Happiness Score")
st.pyplot(fig2)

# 3. Happiness Contribution by Region (Pie Chart)

st.subheader("Top 10 Regions Contributing to Global Happiness")

# Group by region, sum scores, sort and select top 10
region_scores = df.groupby("Country or region")["Score"].sum().sort_values(ascending=False).head(10)

# Plot pie chart
fig3, ax3 = plt.subplots()
ax3.pie(region_scores, labels=region_scores.index, autopct="%1.1f%%", startangle=140)
ax3.axis("equal")
ax3.set_title("Top 10 Regions by Total Happiness Score")
st.pyplot(fig3)


# 4. Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_cols = ['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
             'Freedom to make life choices', 'Generosity']
corr = df[corr_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="YlGnBu")
ax4.set_title("Correlation Between Variables")
st.pyplot(fig4)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.subheader(" Predict Happiness Score using GDP & Social support")

# Features and target
X = df[['GDP per capita', 'Social support']]  # Make sure both are in a list
y = df['Score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot actual vs predicted
fig5, ax5 = plt.subplots()
ax5.scatter(y_test, y_pred, color='darkgreen', alpha=0.7)
ax5.set_xlabel("Actual Happiness Score")
ax5.set_ylabel("Predicted Happiness Score")
ax5.set_title("Actual vs Predicted Happiness Score")
st.pyplot(fig5)


# Footer
st.markdown("---")
st.caption("Streamlit Assignment 2 – World Happiness Dashboard")
