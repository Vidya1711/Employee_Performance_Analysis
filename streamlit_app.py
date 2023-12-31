import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("Employee Performance Analysis")

    # Define your list of columns
    columns_for_df = ['Gender','EmpDepartment', 'EmpJobRole', 'EmpEnvironmentSatisfaction',
       'EmpLastSalaryHikePercent', 'EmpWorkLifeBalance',
       'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # Create a dictionary to store the selected inputs
    selected_data = {col: 0 for col in columns_for_df}

    # Create a dictionary mapping each categorical feature to its options
    categorical_options = {
        'Gender': ['male', 'female'],
        # 'EducationBackground': ['Marketing', 'Life Sciences', 'Human Resources', 'Medical','Other', 'Technical Degree'],  # Add options for EducationBackground
        # 'MaritalStatus': ['Single', 'Married', 'Divorced'],  # Add options for MaritalStatus
        'EmpDepartment': ['Sales', 'Human Resources', 'Development', 'Data Science','Research & Development', 'Finance'],  # Add options for EmpDepartment
        'EmpJobRole': ['Sales Executive', 'Manager', 'Developer', 'Sales Representative','Human Resources', 'Senior Developer', 'Data Scientist','Senior Manager R&D', 'Laboratory Technician','Manufacturing Director', 'Research Scientist',
        'Healthcare Representative', 'Research Director', 'Manager R&D','Finance Manager', 'Technical Architect', 'Business Analyst','Technical Lead', 'Delivery Manager'],  # Add options for EmpJobRole
        # 'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],  # Add options for BusinessTravelFrequency
        # 'OverTime': ['No', 'Yes']  # Add options for OverTime
    }

    # Create select boxes for categorical features
    for feature, options in categorical_options.items():
        selected_value = st.sidebar.selectbox(f"Select {feature}:", options)
        selected_data[feature] = selected_value

    # Define numerical features and their corresponding slider ranges
    numerical_features = {
        # 'Age': (18, 100, 30),
        # 'DistanceFromHome': (1, 30, 5),  # Adjust the range for DistanceFromHome
        # 'EmpHourlyRate': (30, 100, 50),  # Adjust the range for EmpHourlyRate
        # 'TotalWorkExperienceInYears': (0, 40, 10),  # Adjust the range for TotalWorkExperienceInYears
        # 'YearsSinceLastPromotion': (0, 15, 0),  # Adjust the range for YearsSinceLastPromotion
        'EmpEnvironmentSatisfaction' : (0,15,1),
        'EmpLastSalaryHikePercent': (10,30,11),
        'EmpWorkLifeBalance' : (0,5,1),
        'ExperienceYearsAtThisCompany': (0,40,0),
        'ExperienceYearsInCurrentRole': (0,20,0),
        'YearsSinceLastPromotion' : (0,20,0),
        'YearsWithCurrManager': (0,20,0)
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


# import streamlit as st
# import pandas as pd 
# import joblib
# import json 
# import numpy as np
# from pipelines.deployment_pipeline import prediction_service_loader



# def prediction():
#     st.title('Sales Conversion Optimization Model')
    
#     try: 
#         gender = st.selectbox('Gender',['Male', 'Female'])
#         education_background = st.selectbox('EducationBackground', ['Marketing', 'Life Sciences', 'Human Resources', 'Medical','Other', 'Technical Degree'])
#         marital_status = st.selectbox('MaritalStatus', ['Single', 'Married', 'Divorced'])
#         empdepartment = st.selectbox('EmpDepartment', ['Sales', 'Human Resources', 'Development', 'Data Science','Research & Development', 'Finance'])
#         job_role = st.selectbox('EmpJobRole', ['Sales Executive', 'Manager', 'Developer', 'Sales Representative','Human Resources', 'Senior Developer', 'Data Scientist','Senior Manager R&D', 'Laboratory Technician','Manufacturing Director', 'Research Scientist',
#        'Healthcare Representative', 'Research Director', 'Manager R&D','Finance Manager', 'Technical Architect', 'Business Analyst','Technical Lead', 'Delivery Manager'])
#         travel_frequency = st.selectbox('BusinessTravelFrequency', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
#         overtime = st.selectbox('OverTime', ['No', 'Yes'])
#         age = st.number_input('Age' )
#         distance = st.number_input('DistanceFromHome' )
#         hourlyrate = st.number_input('EmpHourlyRate' )
#         workexperience = st.number_input('TotalWorkExperienceInYears')
#         yearsincepromotion = st.number_input('YearsSinceLastPromotion')
     
#         gender_map = {'Male': 1, 'Female': 0}
#         education_map = {'Marketing':2, 'Life Sciences' :1, 'Human Resources':0, 'Medical':3,'Other':4, 'Technical Degree':5}
#         marital_map = {'Single':2, 'Married':1, 'Divorced':0}
#         empdepartment_map = {'Sales':5, 'Human Resources':3, 'Development':1, 'Data Science':0,'Research & Development':4, 'Finance':2}
#         job_role_map = {'Sales Executive':13, 'Manager':8, 'Developer':3, 'Sales Representative':14,'Human Resources':6, 'Senior Developer':15, 'Data Scientist':1,'Senior Manager R&D':16, 'Laboratory Technician':7,'Manufacturing Director':10, 'Research Scientist':12,
#        'Healthcare Representative':5, 'Research Director':11, 'Manager R&D':9,'Finance Manager':4, 'Technical Architect':17, 'Business Analyst':0,'Technical Lead':18, 'Delivery Manager':2}
#         travel_map = {'Travel_Rarely':2, 'Travel_Frequently':1, 'Non-Travel':0}
#         overtime_map = {'No':0, 'Yes':1}
        
#         gender_feat = gender_map[gender]
#         education_feat = education_map[education_background]
#         marital_feat = marital_map[marital_status]
#         empdepartment_feat = empdepartment_map[empdepartment]
#         job_role_feat = job_role_map[job_role]
#         travel_frequency_feat = travel_map[travel_frequency]
#         overtime_feat = overtime_map[overtime]

#         df = {
#             'gender': gender_feat,
#              'age' :age,
#              'education': education_feat,
#              'marital_status':marital_feat,
#              'emp_department': empdepartment_feat,
#              'job_role': job_role_feat,
#              'travel_frequency': travel_frequency_feat,
#              'overtime': overtime_feat,
#              'distance': distance,
#              'hourlyrate': hourlyrate,
#              'work_experience': workexperience,
#              'yearssincelastpromotion': yearsincepromotion
#          }
    
        
#         data = pd.Series([gender_feat,age,education_feat,marital_feat,empdepartment_feat,job_role_feat,travel_frequency_feat,overtime_feat,distance,hourlyrate,workexperience,yearsincepromotion])
#         if st.button('Predict'):
#             service = prediction_service_loader(
#         pipeline_name="continuous_deployment_pipeline",
#         pipeline_step_name="mlflow_model_deployer_step",
#         running=False,
#         )
#         df = pd.DataFrame(
#             {
#              'gender': [gender_feat],
#              'age' :[age],
#              'education': [education_feat],
#              'marital_status':[marital_feat],
#              'emp_department': [empdepartment_feat],
#              'job_role': [job_role_feat],
#              'travel_frequency': [travel_frequency_feat],
#              'overtime': [overtime_feat],
#              'distance': [distance],
#              'hourlyrate': [hourlyrate],
#              'work_experience': [workexperience],
#              'yearssincelastpromotion': [yearsincepromotion] 
#             }
#         )
#         json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#         data = np.array(json_list)
#         pred = service.predict(data)
#         st.success(f"Employee Performance :{pred}") 
            
#     except Exception as e:
#         st.error(e)
        
        
        
        
# if __name__ == '__main__':
#     prediction()
    
    
    
