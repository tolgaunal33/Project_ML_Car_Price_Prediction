import streamlit as st
import numpy as np
import pickle
from PIL import Image
import time
import matplotlib.pyplot as plt

# Load a car image for the header
header_image = Image.open("car_pic.png")  

# Set Streamlit Page Config
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Feature Inputs Function
def get_user_inputs():
    st.sidebar.header("Select Car Features")
    
    # Organize inputs into categories with icons
    with st.sidebar.expander("🕰️ Basic Information"):
        year = st.selectbox("Year", [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    
    with st.sidebar.expander("💪 Performance"):
        horsepower = st.number_input("Horsepower", min_value=0, max_value=1500, step=1, value=150)
        torque = st.number_input("Torque", min_value=0, max_value=1500, step=1, value=150)
        cylinders = st.selectbox("Cylinder Configuration", ["Flat-4", "Flat-6", "I3", "I4", "I5", "I6", "V6", "V8", "V10", "V12", "W12", "Unknown"])
        
    with st.sidebar.expander("🏭 Vehicle Details"):    
        makes = st.selectbox('Make', ["Acura", "Alfa Romeo", "Aston Martin", "Audi", "BMW", "Bentley", "Buick", "Cadillac", "Chevrolet", 
                "Chrysler", "Dodge", "FIAT", "Ferrari", "Fisker", "Ford", "GMC", "Genesis", "Honda", "Hyundai", "INEOS", 
                "INFINITI", "Jaguar", "Jeep", "Karma", "Kia", "Lamborghini", "Land Rover", "Lexus", "Lincoln", "Lotus", 
                "Lucid", "MINI", "Maserati", "Mazda", "McLaren", "Mercedes-Benz", "Mitsubishi", "Nissan", "Polestar", 
                "Porsche", "Ram", "Rivian", "Rolls-Royce", "Scion", "Subaru", "Tesla", "Toyota", "VinFast", "Volkswagen", "Volvo", "smart"])
    
        body_sizes = st.selectbox('Body Size', ["Compact", "Large", "Midsize"])
    
        body_styles = st.selectbox('Body Style', ["Cargo Minivan", "Cargo Van", "Convertible", "Convertible SUV", "Coupe", "Hatchback", 
                    "Passenger Minivan", "Passenger Van", "Pickup Truck", "SUV", "Sedan", "Wagon"])

    with st.sidebar.expander("⚙️ Technical Specifications"):    
        engine_aspirations = st.selectbox('Engine Aspiration', ["Electric Motor", "Naturally Aspirated", "Supercharged", "Turbocharged", "Twin-Turbo", "Twincharged"])
    
        drivetrains = st.selectbox('Drivetrain', ["4WD", "AWD", "FWD", "RWD"])

        transmissions = st.selectbox('Transmission', ["Automatic", "Manual"])
    
    # Combine all inputs into a dictionary
    user_inputs = {
        "Year": year,
        "Horsepower": horsepower,
        "Torque": torque,
        'Cylinder': cylinders,
        "Make": makes,
        "Body Size": body_sizes,
        "Body Style": body_styles,
        "Engine Aspiration": engine_aspirations,
        "Drivetrain": drivetrains,
        "Transmission": transmissions,
    }
    
    return user_inputs

# Transforming the user input
def transform_user_inputs(data, feature_list):
    input_data = {feature: 0 for feature in feature_list}

    # Numeric Features
    input_data['Year'] = data['Year']
    input_data['Horsepower'] = data['Horsepower']
    input_data['Torque'] = data['Torque']

    # Categorical One-Hot Encoding
    for cat in ['Cylinder', 'Make', 'Body Size', 'Body Style', 'Engine Aspiration', 'Drivetrain', 'Transmission']:
        feature_name = f"{cat}_{data[cat]}"
        if feature_name in input_data:
            input_data[feature_name] = 1

    return np.array([list(input_data.values())])

# List of all possible categorical features and their options
features = [
    'Year', 'Horsepower', 'Torque',
    'Cylinders_Flat-4', 'Cylinders_Flat-6', 'Cylinders_I3', 'Cylinders_I4', 'Cylinders_I5', 'Cylinders_I6', 
    'Cylinders_Unknown', 'Cylinders_V10', 'Cylinders_V12', 'Cylinders_V6', 'Cylinders_V8', 'Cylinders_W12',
    'Make_Acura', 'Make_Alfa Romeo', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley', 
    'Make_Buick', 'Make_Cadillac', 'Make_Chevrolet', 'Make_Chrysler', 'Make_Dodge', 'Make_FIAT', 
    'Make_Ferrari', 'Make_Fisker', 'Make_Ford', 'Make_GMC', 'Make_Genesis', 'Make_Honda', 'Make_Hyundai', 
    'Make_INEOS', 'Make_INFINITI', 'Make_Jaguar', 'Make_Jeep', 'Make_Karma', 'Make_Kia', 'Make_Lamborghini', 
    'Make_Land Rover', 'Make_Lexus', 'Make_Lincoln', 'Make_Lotus', 'Make_Lucid', 'Make_MINI', 'Make_Maserati', 
    'Make_Mazda', 'Make_McLaren', 'Make_Mercedes-Benz', 'Make_Mitsubishi', 'Make_Nissan', 'Make_Polestar', 
    'Make_Porsche', 'Make_Ram', 'Make_Rivian', 'Make_Rolls-Royce', 'Make_Scion', 'Make_Subaru', 'Make_Tesla', 
    'Make_Toyota', 'Make_VinFast', 'Make_Volkswagen', 'Make_Volvo', 'Make_smart',
    'Body Size_Compact', 'Body Size_Large', 'Body Size_Midsize',
    'Body Style_Cargo Minivan', 'Body Style_Cargo Van', 'Body Style_Convertible', 'Body Style_Convertible SUV', 
    'Body Style_Coupe', 'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van', 
    'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
    'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated', 'Engine Aspiration_Supercharged', 
    'Engine Aspiration_Turbocharged', 'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged',
    'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD',
    'Transmission_Automatic', 'Transmission_Manual'
]

# Main Streamlit app
def main():
    # Header Image
    st.image(header_image, use_column_width=True)

    # App Title and Description
    st.title("🚗 Car Price Prediction App")
    st.markdown("""
        Welcome to the **Car Price Prediction App**! This app predicts the price of a car based on its features.
        Simply select the car's features in the sidebar and click **Predict** to get the estimated price.
    """)

    # Get user inputs
    user_inputs = get_user_inputs()

    # Transform user inputs
    transformed_inputs = transform_user_inputs(user_inputs, features)

    # Load the saved Linear Regression model
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make a prediction
    if st.button("🚀 Predict Price"):
        with st.spinner("Calculating price..."):
            time.sleep(1)
            prediction = model.predict(transformed_inputs)
            st.success(f"Estimated Price: **${prediction[0]:,.2f}**")

    # Reset Inputs Button 
    if st.sidebar.button("Reset Inputs"):
        st.session_state.clear()
        st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ by Tolga Unal")    
    st.markdown("---")
    st.markdown("""
    **Follow me on:**
    - [GitHub](https://github.com/tolgaunal33)
    - [LinkedIn](https://linkedin.com/in/tolgaunal33)
    - [Twitter](https://twitter.com/tolgaunal33)
""")

    # Dark Mode Toggle 
    dark_mode = st.sidebar.checkbox("Dark Mode")
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stSidebar {
                background-color: #2e2e2e;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    main()