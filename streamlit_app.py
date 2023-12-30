import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("Employee Performance Analysis")

    # Define your list of columns
    columns_for_df = ['Age', 'Gender', 'EducationBackground', 'MaritalStatus',
                       'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
                       'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
                       'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel',
                       'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime',
                       'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
                       'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
                       'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
                       'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
                       'YearsWithCurrManager', 'Attrition']

    # Create a dictionary to store the selected inputs
    selected_data = {col: 0 for col in columns_for_df}

    # Create a dictionary mapping each categorical feature to its options
    categorical_options = {
        'Gender': ['male', 'female'],
        'EducationBackground': ['Marketing', 'Life Sciences', 'Human Resources', 'Medical','Other', 'Technical Degree'],  # Add options for EducationBackground
        'MaritalStatus': ['Single', 'Married', 'Divorced'],  # Add options for MaritalStatus
        'EmpDepartment': ['Sales', 'Human Resources', 'Development', 'Data Science','Research & Development', 'Finance'],  # Add options for EmpDepartment
        'EmpJobRole': ['Sales Executive', 'Manager', 'Developer', 'Sales Representative','Human Resources', 'Senior Developer', 'Data Scientist','Senior Manager R&D', 'Laboratory Technician','Manufacturing Director', 'Research Scientist',
       'Healthcare Representative', 'Research Director', 'Manager R&D','Finance Manager', 'Technical Architect', 'Business Analyst','Technical Lead', 'Delivery Manager'],  # Add options for EmpJobRole
        'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],  # Add options for BusinessTravelFrequency
        'OverTime': ['No', 'Yes']  # Add options for OverTime
    }

    # Create select boxes for categorical features
    for feature, options in categorical_options.items():
        selected_value = st.sidebar.selectbox(f"Select {feature}:", options)
        selected_data[feature] = selected_value

    # Define numerical features and their corresponding slider ranges
    numerical_features = {
        'Age': (18, 100, 30),
        'DistanceFromHome': (1, 30, 5),  # Adjust the range for DistanceFromHome
        'EmpHourlyRate': (30, 100, 50),  # Adjust the range for EmpHourlyRate
        'TotalWorkExperienceInYears': (0, 40, 10),  # Adjust the range for TotalWorkExperienceInYears
        'YearsSinceLastPromotion': (0, 15, 0)  # Adjust the range for YearsSinceLastPromotion
    }

    # Create sliders for numerical features
    for feature, (min_val, max_val, default_val) in numerical_features.items():
        selected_value = st.sidebar.slider(f"Select {feature}:", min_val, max_val, default_val)
        selected_data[feature] = selected_value

    # Create a DataFrame with the selected inputs
    data_df = pd.DataFrame([selected_data])

    # Display the selected inputs
    st.write("Selected Inputs:")
    st.write(data_df)

    if st.button("Predict"):
        # Load the prediction service
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )

        # If service is not available, run the pipeline to create it
        if service is None:
            st.write("No service could be found. The pipeline will be run first to create a service.")
            main()

        # Prepare the input data for prediction
        json_list = json.loads(json.dumps(list(data_df.T.to_dict().values())))
        data = np.array(json_list)

        # Make predictions using the service
        pred = service.predict(data)

        # Display the predicted result
        st.success("Predicted Employee Attrition Probability (0 - 1): {:.2f}".format(pred[0]))

if __name__ == "__main__":
    main()
