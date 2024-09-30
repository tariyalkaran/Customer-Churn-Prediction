
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Fetch input data from form
    inputQuery1 = request.form.get('query1', None)
    inputQuery2 = request.form.get('query2', None)
    inputQuery3 = request.form.get('query3', None)
    inputQuery4 = request.form.get('query4', None)
    inputQuery5 = request.form.get('query5', None)
    inputQuery6 = request.form.get('query6', None)
    inputQuery7 = request.form.get('query7', None)
    inputQuery8 = request.form.get('query8', None)
    inputQuery9 = request.form.get('query9', None)
    inputQuery10 = request.form.get('query10', None)
    inputQuery11 = request.form.get('query11', None)
    inputQuery12 = request.form.get('query12', None)
    inputQuery13 = request.form.get('query13', None)
    inputQuery14 = request.form.get('query14', None)
    inputQuery15 = request.form.get('query15', None)
    inputQuery16 = request.form.get('query16', None)
    inputQuery17 = request.form.get('query17', None)
    inputQuery18 = request.form.get('query18', None)
    inputQuery19 = request.form.get('query19', None)

    # Collect the inputs into a list
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    # Debugging: Check if all inputs are captured
    print("Data received from form:", data)  # You can check this in the console or logs
    
    # Create DataFrame with 19 columns
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])

    
    # Feature engineering (e.g., tenure grouping)
    new_df['tenure_group'] = pd.cut(new_df.tenure.astype(int), range(0, 80, 12), right=False, labels=[
        "{0} - {1}".format(i, i + 11) for i in range(0, 72, 12)
    ])
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encoding for categorical columns
    numerical_cols = new_df[['MonthlyCharges', 'TotalCharges']]
    categorical_cols = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                              'PhoneService', 'MultipleLines', 'InternetService', 
                                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                              'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                              'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                              'tenure_group']])
    
    new_df_dummies = pd.concat([numerical_cols, categorical_cols], axis=1)

    # Ensure new_df_dummies has the same columns as the training data
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    missing_cols = set(model_columns) - set(new_df_dummies.columns)
    
    for col in missing_cols:
        new_df_dummies[col] = 0
    
    new_df_dummies = new_df_dummies[model_columns]

    # Load the model and predict
    model = pickle.load(open("model.sav", "rb"))
    single = model.predict(new_df_dummies)
    probability = model.predict_proba(new_df_dummies)[:, 1]

    # Return result
    if single == 1:
        output1 = "This customer is likely to churn!"
        output2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        output1 = "This customer is likely to continue!"
        output2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    return render_template('home.html', output1=output1, output2=output2)

app.run()


