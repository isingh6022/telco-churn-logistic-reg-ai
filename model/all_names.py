from typing import Literal, get_args

data_file  = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
id_col     = 'customerID'
target_col = 'Churn'

target_encoding = {'No': 0, 'Yes': 1}

# Allowed values
Gender        = Literal['Female', 'Male']
YesNo         = Literal['No', 'Yes']
SeniorFlag    = Literal[0, 1]
PhoneAddOn    = Literal['No', 'Yes', 'No phone service']
InternetType  = Literal['DSL', 'Fiber optic', 'No']
InternetAddOn = Literal['No', 'Yes', 'No internet service']
ContractType  = Literal['Month-to-month', 'One year', 'Two year']
PaymentType   = Literal['Electronic check', 'Mailed check',
                        'Bank transfer (automatic)', 'Credit card (automatic)']

# convert values of a Literal type to a list
def values(choice):
    return list(get_args(choice))


# strictly binary columns -> Binary/Label encoding (first value = 0, second = 1)
binary_cols = {
    'gender'          : values(Gender),
    'Partner'         : values(YesNo),
    'Dependents'      : values(YesNo),
    'PhoneService'    : values(YesNo),
    'PaperlessBilling': values(YesNo),
}

# already stored as 0/1 in the csv, used as such
passthrough_cols = ['SeniorCitizen']

# multi categorical columns -> One-Hot encoding (first value = baseline, dropped)
multi_cols = {
    # phone services
    'MultipleLines'   : values(PhoneAddOn),
    # internet and add on services
    'InternetService' : values(InternetType),
    'OnlineSecurity'  : values(InternetAddOn),
    'OnlineBackup'    : values(InternetAddOn),
    'DeviceProtection': values(InternetAddOn),
    'TechSupport'     : values(InternetAddOn),
    'StreamingTV'     : values(InternetAddOn),
    'StreamingMovies' : values(InternetAddOn),
    # account info
    'Contract'        : values(ContractType),
    'PaymentMethod'   : values(PaymentType),
}

# numerical columns -> scaled to mean 0, std 1 (from the training rows)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


# all the input features, in the order the model receives them
feature_cols = list(binary_cols) + passthrough_cols + list(multi_cols) + numeric_cols
