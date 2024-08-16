import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import joblib

# Load and prepare the data
data = pd.read_csv('Placement.csv')
warnings.filterwarnings('ignore')

# Check for and display non-numeric values
print(data.head())
print(data.dtypes)

# Drop unnecessary columns and handle categorical encoding
data = data.drop(['sl_no', 'salary'], axis=1)

# Convert categorical columns to numeric
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['ssc_b'] = data['ssc_b'].map({'Central': 1, 'Others': 0})
data['hsc_b'] = data['hsc_b'].map({'Central': 1, 'Others': 0})
data['hsc_s'] = data['hsc_s'].map({'Science': 2, 'Commerce': 1, 'Arts': 0})
data['degree_t'] = data['degree_t'].map({'Sci&Tech': 2, 'Comm&Mgmt': 1, 'Others': 0})
data['specialisation'] = data['specialisation'].map({'Mkt&HR': 1, 'Mkt&Fin': 0})
data['workex'] = data['workex'].map({'Yes': 1, 'No': 0})
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Check for any remaining non-numeric values
print(data.head())
print(data.dtypes)

# Ensure all columns are numeric
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Non-numeric values found in column: {col}")
        print(data[col].unique())

# Handle missing values (impute with mean for simplicity)
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split data into features and target variable
X = data.drop('status', axis=1)
y = data['status']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train models
lr = LogisticRegression()
lr.fit(X_train, y_train)
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Make predictions
y_pred1 = lr.predict(X_test)
y_pred2 = svm_model.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = gb.predict(X_test)

# Evaluate model performance
score1 = accuracy_score(y_test, y_pred1)
score2 = accuracy_score(y_test, y_pred2)
score3 = accuracy_score(y_test, y_pred3)
score4 = accuracy_score(y_test, y_pred4)
score5 = accuracy_score(y_test, y_pred5)
score6 = accuracy_score(y_test, y_pred6)

print(score1, score2, score3, score4, score5, score6)

# Create DataFrame for model accuracy
final_data = pd.DataFrame({
    'Models': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GB'],
    'ACC': [score1 * 100, score2 * 100, score3 * 100, score4 * 100, score5 * 100, score6 * 100]
})

# Plot model performance
sns.barplot(x=final_data['Models'], y=final_data['ACC'])

# Predict on new data
new_data = pd.DataFrame({
    'gender': [0],
    'ssc_p': [67.0],
    'ssc_b': [0],
    'hsc_p': [91.0],
    'hsc_b': [0],
    'hsc_s': [1],
    'degree_p': [58.0],
    'degree_t': [2],
    'workex': [0],
    'etest_p': [55.0],
    'specialisation': [1],
    'mba_p': [58.8],
})

# Ensure new data has same columns as X
new_data = new_data[X.columns]

# Check new data for non-numeric values
print(new_data.head())
print(new_data.dtypes)

# Load and use the saved model
joblib.dump(lr, 'model_campus_placement')
model = joblib.load('model_campus_placement')
p = model.predict(new_data)
prob = model.predict_proba(new_data)

# Print prediction
if p[0] == 1:
    print('Placed')
    print(f"You will be placed with probability of {prob[0][1]:.2f}")
else:
    print("Not Placed")

print(prob)
