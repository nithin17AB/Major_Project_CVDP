import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Cardio_dataset.csv", sep=';')
print(df.head())
print(df.columns)

print(df.isnull().sum())  # check missing values
print(df.describe())      # statistics
print(df.info())          # data types

sns.countplot(x='cardio', data=df)
plt.title("Heart Disease Count")
plt.show()

sns.histplot(df['age'], bins=20)
plt.show()

sns.countplot(x='gender', hue='cardio', data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


from sklearn.model_selection import train_test_split
X = df.drop('cardio', axis=1)
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
acc_lr = model_lr.score(X_test, y_test)
print("LR Accuracy:", acc_lr)


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
acc_knn = model_knn.score(X_test, y_test)
print("KNN Accuracy:", acc_knn)



from sklearn.svm import SVC
model_svm = SVC()
model_svm.fit(X_train, y_train)
acc_svm = model_svm.score(X_test, y_test)
print("SVM Accuracy:", acc_svm)


from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
acc_dt = model_dt.score(X_test, y_test)
print("DT Accuracy:", acc_dt)


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
acc_rf = model_rf.score(X_test, y_test)
print("RF Accuracy:", acc_rf)


models = ['LR','KNN','SVM','DT','RF']
accuracy = [acc_lr, acc_knn, acc_svm, acc_dt, acc_rf]

sns.barplot(x=models, y=accuracy)
plt.title("Model Comparison")
plt.show()

final_model = model_rf