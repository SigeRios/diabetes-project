import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')

# Optionally print descriptions to inspect the data
print(reference_data.describe())
print(new_data.describe())

# Remove columns with zero variance
reference_data = reference_data.loc[:, (reference_data != 0).any(axis=0)]
new_data = new_data.loc[:, (new_data != 0).any(axis=0)]

# Apply smoothing to avoid divide by zero issues
reference_data += 1e-10
new_data += 1e-10

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference_data.drop('Outcome', axis=1), 
                      current_data=new_data.drop('Outcome', axis=1), 
                      column_mapping=None)

report_json = data_drift_report.as_dict()
drift_detected = report_json['metrics'][0]['result']['dataset_drift']

if drift_detected:
    print("Data drift detected. Retraining the model.")
    with open('drift_detected.txt', 'w') as f:
        f.write('drift_detected')
else:
    print("No data drift detected.")
    with open('drift_detected.txt', 'w') as f:
        f.write('no_drift')
