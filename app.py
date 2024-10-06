import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Display the image in the app
#st.image('pic1.jpg', caption="Heart Surgeon", use_column_width=True)

# Page Configuration
st.set_page_config(
    page_title="Heart Attack Prediction App",
    page_icon="❤️",
    layout="centered"
)

# Load your dataset from the local file
@st.cache_data
def load_data():
    data = pd.read_csv('heart_attack.csv')  # Assuming the file is in the same folder as app.py
    #st.write("Available columns:", data.columns)  # Display the columns in the dataset
    return data

# Train the model
def train_model(df):
    # Exclude 'output' and use the other columns as features
    X = df.drop(columns=['output'])
    y = df['output']  # Target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X.columns  # Return the model and the feature names

# Custom CSS to style the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    .stApp h1 {
        color: #e63946;
        font-size: 2.5rem;
    }
    .stApp .stButton button {
        background-color: #457b9d;
        color: white;
        border-radius: 5px;
        font-size: 1.2rem;
    }
    .stApp input {
        border-radius: 5px;
        padding: 5px;
        border: 1px solid #a8dadc;
    }
    .stApp h2 {
        color: #1d3557;
    }
    </style>
    """, unsafe_allow_html=True
)

# Main function for Streamlit app
def main():
    st.title("Heart Attack Prediction App")
    st.markdown("### Predict your heart attack risk with simple inputs!")
    
    # Load the dataset and train the model
    df = load_data()
    model, feature_columns = train_model(df)

    # Use columns to organize input fields in two columns to save space
    left_column, right_column = st.columns(2)
    
    # Create input fields dynamically for each feature except 'output'
    input_data = {}
    
    for i, column in enumerate(feature_columns):
        # Alternating input fields between the two columns
        with left_column if i % 2 == 0 else right_column:
            if df[column].dtype == 'object':
                input_data[column] = st.text_input(f"{column} (optional)", value="")
            elif df[column].dtype == 'int64' or df[column].dtype == 'float64':
                input_data[column] = st.number_input(f"{column} (optional)", value=float(df[column].median()))

    # Convert input data to a NumPy array, using default values if not provided
    input_values = np.array([[input_data[column] for column in feature_columns]])

    # Styled Button
    if st.button("Predict"):
        prediction = model.predict(input_values)
        result = "High Risk" if prediction == 1 else "Low Risk"
        st.success(f"The prediction is: {result}")

if __name__ == '__main__':
    main()
