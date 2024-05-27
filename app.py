import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model/best_model_bank_load_defaulter_RandomForestClassifier.joblib')

# Define the mappings for categorical columns
mapping_ms = {'single': 1, 'married': 2}
mapping_ho = {'rented': 1, 'owned': 2, 'norent_noown': 3}
mapping_co = {'no': 1, 'yes': 2}

# Initialize session state if not already set
if 'income' not in st.session_state:
    st.session_state['income'] = ''
if 'age' not in st.session_state:
    st.session_state['age'] = ''
if 'experience' not in st.session_state:
    st.session_state['experience'] = ''
if 'married_single' not in st.session_state:
    st.session_state['married_single'] = 'Select an option'
if 'house_ownership' not in st.session_state:
    st.session_state['house_ownership'] = 'Select an option'
if 'car_ownership' not in st.session_state:
    st.session_state['car_ownership'] = 'Select an option'
if 'current_job_yrs' not in st.session_state:
    st.session_state['current_job_yrs'] = ''
if 'current_house_yrs' not in st.session_state:
    st.session_state['current_house_yrs'] = ''

def main():
    st.title("Bank Loan Defaulter Prediction")

    # Input fields for the user to enter data
    st.text_input('Income', key='income')
    st.text_input('Age', key='age')
    st.text_input('Experience', key='experience')
    st.selectbox('Married/Single', ['Select an option', 'single', 'married'], key='married_single')
    st.selectbox('House Ownership', ['Select an option', 'rented', 'owned', 'norent_noown'], key='house_ownership')
    st.selectbox('Car Ownership', ['Select an option', 'no', 'yes'], key='car_ownership')
    st.text_input('Current Job Years', key='current_job_yrs')
    st.text_input('Current House Years', key='current_house_yrs')

    if st.button('Predict'):
        # Create a DataFrame from the input data
        new_data = pd.DataFrame({
            'Income': [st.session_state['income']],
            'Age': [st.session_state['age']],
            'Experience': [st.session_state['experience']],
            'Married/Single': [st.session_state['married_single'] if st.session_state['married_single'] != 'Select an option' else None],
            'House_Ownership': [st.session_state['house_ownership'] if st.session_state['house_ownership'] != 'Select an option' else None],
            'Car_Ownership': [st.session_state['car_ownership'] if st.session_state['car_ownership'] != 'Select an option' else None],
            'CURRENT_JOB_YRS': [st.session_state['current_job_yrs']],
            'CURRENT_HOUSE_YRS': [st.session_state['current_house_yrs']]
        })

        # Ensure that all inputs are provided and not empty
        if new_data.isnull().values.any() or new_data.applymap(lambda x: x is None or x == '').any().any():
            st.error("Error in data input. Please ensure all fields are filled.")
        else:
            # Convert numerical columns to appropriate types
            try:
                new_data = new_data.astype({
                    'Income': float,
                    'Age': int,
                    'Experience': int,
                    'CURRENT_JOB_YRS': int,
                    'CURRENT_HOUSE_YRS': int
                })
            except ValueError as e:
                st.error(f"Error in data input: {e}")
                return
            
            # Map the categorical columns using the provided mappings
            new_data['Married/Single'] = new_data['Married/Single'].map(mapping_ms)
            new_data['House_Ownership'] = new_data['House_Ownership'].map(mapping_ho)
            new_data['Car_Ownership'] = new_data['Car_Ownership'].map(mapping_co)
            
            # Check for NaN values after mapping
            if new_data.isnull().values.any():
                st.error("Error in data input. Please check all fields and try again.")
            else:
                # Scale the new data using the max values from the new data itself
                new_data_scaled = new_data / new_data.max()
                
                # Make predictions on the preprocessed new data
                try:
                    predictions = model.predict(new_data_scaled)
                    
                    # Convert numeric predictions to labels
                    predicted_label = 'non_default' if predictions[0] == 0 else 'defaulter'
                    
                    # Display the prediction
                    st.success(f'Prediction for the new data point: {predicted_label}')
                except ValueError as e:
                    st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()