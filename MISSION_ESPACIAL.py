import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df_missions = pd.read_csv('space_missions.csv', encoding='ISO-8859-1')
df_dict = pd.read_csv('space_missions_data_dictionary.csv', encoding='ISO-8859-1')

print("Diccionario de datos:")
print(df_dict)

print("\n Datos de misiones espaciales:")
print(df_missions.head())

print("\n Descripción de las columnas:")
for field, description in zip(df_dict['Field'], df_dict['Description']):
    print(f"{field}: {description}")

print(df_missions.shape)
print(df_missions.info())
print(df_missions.describe())
print(df_missions.isnull().sum())
df_missions = df_missions.dropna()

if 'Price' in df_missions.columns:
    df_missions['Price'] = df_missions['Price'].str.replace(',', '').astype(float)

categorical_columns = ['Company', 'Location', 'Date', 'Time', 'Rocket', 'Mission', 'RocketStatus']
df_missions = pd.get_dummies(df_missions, columns=categorical_columns)

numeric_columns = df_missions.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df_missions[numeric_columns] = scaler.fit_transform(df_missions[numeric_columns])

if 'MissionStatus' in df_missions.columns:
    x = df_missions.drop('MissionStatus', axis=1)
    y = df_missions['MissionStatus']
else:
    print("La columna 'MissionStatus' no existe en el DataFrame.")
    x, y = None, None

if x is not None and y is not None:
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    precision = accuracy_score(y_test, y_pred)
    print('Precisión:', precision)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.countplot(y_pred)
    plt.xlabel('Predicted Mission Status')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Mission Status')
    plt.show()
    
else:
    print("No se puede continuar sin la columna 'MissionStatus'.")