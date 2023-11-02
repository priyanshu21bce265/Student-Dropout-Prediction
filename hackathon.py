# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression,Perceptron
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


# df = pd.read_csv("data.csv",sep=";")
# # df

# df['Target'] = LabelEncoder().fit_transform(df['Target'])
# # df['Target'].value_counts()
# df.drop(df[df['Target'] == 1].index, inplace = True)
# # df
# df['Dropout'] = df['Target'].apply(lambda x: 1 if x==0 else 0)
# # df
# x = df.iloc[:, :36].values
# #x = df[["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]].values
# print(x)
# x = StandardScaler().fit_transform(x)
# # x
# y = df['Dropout'].values
# # y
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
# def perform(y_pred):
#     print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
#     print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
#     print("Accuracy : ", accuracy_score(y_test, y_pred))
#     print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
#     cm = confusion_matrix(y_test, y_pred)
#     print("\n", cm)
#     print("\n")
#     print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
#     print(classification_report(y_test, y_pred))
#     print("**"*27+"\n")
    
#     cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Non-Dropout', 'Dropout'])
#     cm.plot()
    
    
# model_nb = GaussianNB()
# model_nb.fit(x_train, y_train)
# y_pred_nb = model_nb.predict(x_test)
# perform(y_pred_nb)

# import streamlit as st

# # Your existing code here

# # Add a Streamlit app
# st.title("Student Dropout Prediction")

# # Create input elements for user interaction
# st.sidebar.header("Input Features")
# # Add input elements for all the attributes you want the user to input
# # For example:
# tuition_fees = st.sidebar.number_input("Tuition Fees up to Date", min_value=0, step=1000)
# curricular_units_1st_sem_approved = st.sidebar.number_input("Curricular Units 1st Sem (Approved)", min_value=0, step=1)
# # Add more input elements for other attributes...

# # Create a button for the user to trigger predictions
# if st.sidebar.button("Predict"):
#     # Combine user inputs into a feature array
#     user_inputs = np.array([tuition_fees, curricular_units_1st_sem_approved, ...])  # Add all the inputs here

#     # Perform the same preprocessing on user_inputs as you did on x
#     user_inputs = StandardScaler().transform(user_inputs.reshape(1, -1))

#     # Make predictions using your model
#     prediction = model_nb.predict(user_inputs)

#     # Display the prediction result
#     if prediction[0] == 0:
#         st.write("Prediction: Non-Dropout")
#     else:
#         st.write("Prediction: Dropout")

# # Display the classification report and confusion matrix as you did before
# # Optionally, you can add a section to display plots or visualizations of your choice.

# # Run the Streamlit app
# if __name__ == "__main__":
#     st.sidebar.text("Streamlit web app for student dropout prediction")
#     st.sidebar.text("Input your features and click 'Predict'")












# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.naive_bayes import GaussianNB

# # Load the dataset
# df = pd.read_csv("data.csv", sep=";")
# df['Target'] = LabelEncoder().fit_transform(df['Target'])
# df.drop(df[df['Target'] == 1].index, inplace=True)
# df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# # Preprocess the data
# x = df.iloc[:, :36].values
# x = StandardScaler().fit_transform(x)
# y = df['Dropout'].values

# # Load the trained model
# model_nb = GaussianNB()
# model_nb.fit(x, y)

# # Streamlit app
# st.title("Student Dropout Prediction")

# # Create input elements for user interaction
# st.sidebar.header("Input Features")

# # Add input elements for all the features
# input_features = [
#     "Marital status", "Application mode", "Application order", "Course",
#     "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
#     "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
#     "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor",
#     "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
#     "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
#     "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
# ]

# user_inputs = []

# for feature in input_features:
#     cleaned_feature_name = feature.lower().replace(" ", "_").replace("/", "_")
#     user_input = st.sidebar.text_input(feature, key=cleaned_feature_name)
#     user_inputs.append(user_input)

# # Combine user inputs into a feature array
# user_inputs = np.array(user_inputs, dtype=float)

# # Perform preprocessing on user_inputs
# user_inputs = np.expand_dims(user_inputs, axis=0)
# user_inputs = StandardScaler().transform(user_inputs)

# # Make predictions using the model
# prediction = model_nb.predict(user_inputs)

# # Display the prediction result
# st.write(f"Prediction: {'Dropout' if prediction[0] == 1 else 'Non-Dropout'}")












# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.naive_bayes import GaussianNB

# # Load the dataset
# df = pd.read_csv("data.csv", sep=";")
# df['Target'] = LabelEncoder().fit_transform(df['Target'])
# df.drop(df[df['Target'] == 1].index, inplace=True)
# df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# # Preprocess the data
# x = df.iloc[:, :36].values
# x = StandardScaler().fit_transform(x)
# y = df['Dropout'].values

# # Load the trained model
# model_nb = GaussianNB()
# model_nb.fit(x, y)

# # Streamlit app
# st.title("Student Dropout Prediction")

# # Create input elements for user interaction
# st.sidebar.header("Input Features")

# # Add input elements for all the features
# input_features = [
#     "Marital status", "Application mode", "Application order", "Course",
#     "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
#     "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
#     "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor",
#     "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
#     "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
#     "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
# ]

# user_inputs = []

# for feature in input_features:
#     cleaned_feature_name = feature.lower().replace(" ", "_").replace("/", "_")
#     user_input = st.sidebar.text_input(feature, key=cleaned_feature_name)
#     user_inputs.append(user_input)

# # Validate user inputs and convert them to float
# valid_inputs = []
# for user_input in user_inputs:
#     if user_input.strip() == '':
#         valid_inputs.append(np.nan)
#     else:
#         valid_inputs.append(float(user_input))

# # Combine user inputs into a feature array
# user_inputs = np.array(valid_inputs)

# # Check if there are any NaN values in user_inputs
# if np.isnan(user_inputs).any():
#     st.write("Please fill in all input fields with valid numeric values.")
# else:
#     # Perform preprocessing on user_inputs
#     user_inputs = np.expand_dims(user_inputs, axis=0)
#     user_inputs = StandardScaler().transform(user_inputs)

#     # Make predictions using the model
#     prediction = model_nb.predict(user_inputs)

#     # Display the prediction result
#     st.write(f"Prediction: {'Dropout' if prediction[0] == 1 else 'Non-Dropout'}")

# without button










# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.naive_bayes import GaussianNB

# # Load the dataset
# df = pd.read_csv("data.csv", sep=";")
# df['Target'] = LabelEncoder().fit_transform(df['Target'])
# df.drop(df[df['Target'] == 1].index, inplace=True)
# df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# # Preprocess the data
# x = df.iloc[:, :36].values
# x = StandardScaler().fit_transform(x)
# y = df['Dropout'].values

# # Load the trained model
# model_nb = GaussianNB()
# model_nb.fit(x, y)

# # Streamlit app
# st.title("Student Dropout Prediction")

# # Create input elements for user interaction
# st.sidebar.header("Input Features")

# # Add input elements for all the features
# input_features = [
#     "Marital status", "Application mode", "Application order", "Course",
#     "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
#     "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
#     "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor",
#     "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
#     "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
#     "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
# ]

# user_inputs = []

# for feature in input_features:
#     cleaned_feature_name = feature.lower().replace(" ", "_").replace("/", "_")
#     user_input = st.sidebar.text_input(feature, key=cleaned_feature_name)
#     user_inputs.append(user_input)

# # Add a "Predict" button
# if st.sidebar.button("Predict"):
#     # Validate user inputs and convert them to float
#     valid_inputs = []
#     for user_input in user_inputs:
#         if user_input.strip() == '':
#             valid_inputs.append(np.nan)
#         else:
#             valid_inputs.append(float(user_input))

#     # Combine user inputs into a feature array
#     user_inputs = np.array(valid_inputs)

#     # Check if there are any NaN values in user_inputs
#     if np.isnan(user_inputs).any():
#         st.write("Please fill in all input fields with valid numeric values.")
#     else:
#         # Perform preprocessing on user_inputs
#         user_inputs = np.expand_dims(user_inputs, axis=0)
#         user_inputs = StandardScaler().transform(user_inputs)

#         # Make predictions using the model
#         prediction = model_nb.predict(user_inputs)

#         # Display the prediction result
#         st.write(f"Prediction: {'Dropout' if prediction[0] == 1 else 'Non-Dropout'}")
# hale chhe but not correct




# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# # Load the dataset
# df = pd.read_csv("data.csv", sep=";")
# df['Target'] = LabelEncoder().fit_transform(df['Target'])
# df.drop(df[df['Target'] == 1].index, inplace=True)
# df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# # Define the list of features (including the corrected "Daytime/evening attendance" feature)
# input_features = [
#     "Marital status", "Application mode", "Application order", "Course",
#     "Daytime/evening attendance\t",  # Corrected feature name
#     "Previous qualification", "Previous qualification (grade)",
#     "Nacionality", "Mother's qualification", "Father's qualification",
#     "Mother's occupation", "Father's occupation", "Admission grade", "Displaced",
#     "Educational special needs", "Debtor", "Tuition fees up to date", "Gender",
#     "Scholarship holder", "Age at enrollment", "International",
#     "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
#     "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
#     "Curricular units 1st sem (grade)",
#     "Curricular units 1st sem (without evaluations)",
#     "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
#     "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
#     "Curricular units 2nd sem (grade)",
#     "Curricular units 2nd sem (without evaluations)",
#     "Unemployment rate", "Inflation rate", "GDP"
# ]

# # Preprocess the data
# X = df[input_features].values  # Include only the selected features
# y = df['Dropout'].values

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # Create a StandardScaler and fit it using the training data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Create and train a Support Vector Classifier (SVC)
# svc = SVC(kernel='linear', C=1)
# svc.fit(X_train_scaled, y_train)

# # Streamlit app
# st.title("Student Dropout Prediction")

# # Create input elements for user interaction
# st.sidebar.header("Input Features")

# user_inputs = []

# for feature in input_features:
#     cleaned_feature_name = feature.lower().replace(" ", "_").replace("/", "_")
#     user_input = st.sidebar.text_input(feature, key=cleaned_feature_name)
#     user_inputs.append(user_input)

# # Add a "Predict" button
# if st.sidebar.button("Predict"):
#     # Validate user inputs and convert them to float
#     valid_inputs = []
#     for user_input in user_inputs:
#         if user_input.strip() == '':
#             valid_inputs.append(np.nan)
#         else:
#             valid_inputs.append(float(user_input))

#     # Combine user inputs into a feature array
#     user_inputs = np.array(valid_inputs)

#     # Check if there are any NaN values in user_inputs
#     if np.isnan(user_inputs).any():
#         st.write("Please fill in all input fields with valid numeric values.")
#     else:
#         # Scale the user inputs using the same scaler used for training data
#         user_inputs_scaled = scaler.transform(user_inputs.reshape(1, -1))

#         # Make predictions using the trained SVC model
#         prediction = svc.predict(user_inputs_scaled)

#         # Display the prediction result
#         st.write(f"Prediction: {'Dropout' if prediction[0] == 1 else 'Non-Dropout'}")





























import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv", sep=";")
df['Target'] = LabelEncoder().fit_transform(df['Target'])
df.drop(df[df['Target'] == 1].index, inplace=True)
df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# Define the list of features (including the corrected "Daytime/evening attendance" feature)
input_features = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance\t",  # Corrected feature name
    "Previous qualification", "Previous qualification (grade)",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date", "Gender",
    "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Preprocess the data
X = df[input_features].values  # Include only the selected features
y = df['Dropout'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a StandardScaler and fit it using the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a Support Vector Classifier (SVC)
svc = SVC(kernel='linear', C=1)
svc.fit(X_train_scaled, y_train)

# Streamlit app
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="✅",
    layout="wide",
)

st.title("Student Dropout Prediction")

# Create a form for user interaction
with st.form("user_input_form"):
    user_inputs = []

    for feature in input_features:
        cleaned_feature_name = feature.lower().replace(" ", "_").replace("/", "_")
        with st.container():
            user_input = st.text_input(feature, key=cleaned_feature_name)
            user_inputs.append(user_input)

    # Add a "Predict" button at the end
    predict_button = st.form_submit_button("Predict")

# Prevent Enter key from triggering form submission
if predict_button:
    st.markdown('<style>div.row-widget.stButton > button {width: 100%;}</style>', unsafe_allow_html=True)

    # Validate user inputs and convert them to float
    valid_inputs = []
    for user_input in user_inputs:
        if user_input.strip() == "":
            valid_inputs.append(np.nan)
        else:
            valid_inputs.append(float(user_input))

    # Combine user inputs into a feature array
    user_inputs = np.array(valid_inputs)

    # Check if there are any NaN values in user_inputs
    if np.isnan(user_inputs).any():
        st.warning("Please fill in all input fields with valid numeric values.")
    else:
        # Scale the user inputs using the same scaler used for training data
        user_inputs_scaled = scaler.transform(user_inputs.reshape(1, -1))

        # Make predictions using the trained SVC model
        prediction = svc.predict(user_inputs_scaled)

        # Display the prediction result
        result_text = "Dropout" if prediction[0] == 1 else "Non-Dropout"
        st.success(f"Prediction: {result_text}")
