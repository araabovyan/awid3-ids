from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from utils import read_csvs

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score


DATA_DIR="/data/ara/processed/awid.csv"


df = pd.read_csv(DATA_DIR)
print(df.describe(include='object'))

dummies = []
cols = ['radiotap.present.tsft', 'wlan.fc.ds']
for col in cols:
   dummies.append(pd.get_dummies(df[col]))
df.drop(cols, axis=1, inplace=True)

dummies =  pd.concat(dummies, axis=1)
df = pd.concat((df, dummies), axis=1)

y = df['label']
X = df.drop(['label'], axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)


sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y_encoded)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression()
# models['Support Vector Machines linear'] = LinearSVC()
# models['Support Vector Machines plonomial'] = SVC(kernel='poly')
# models['Support Vector Machines RBf'] =  SVC(C=100.0)
# models['Decision Trees'] = DecisionTreeClassifier(max_depth=3)
# models['Random Forest'] = RandomForestClassifier()
# models['Naive Bayes'] = GaussianNB()
# models['K-Nearest Neighbor'] = KNeighborsClassifier(n_neighbors=20)


accuracy, precision, recall = {}, {}, {}
confusion_matrix = {}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, y_train)
    
    # Make predictions
    predictions = models[key].predict(X_test)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    confusion_matrix[key] = confusion_matrix(y_test, predictions)


df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)