#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
for dirname, _, filenames in os.walk('/home/andrea/Documents'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.simplefilter("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:



df = pd.read_csv('/home/andrea/Documents/water_potability.csv')
df.head()


# In[7]:


sns.countplot(x="Potability",data=df, palette={0:'blue', 1:'green'})
plt.xlabel('Potability: 0 is not potable, 1 is potable')
plt.ylabel('Amount of water')
porc = (len(df[df.Potability==1]) / len(df.Potability)) * 100
print('The percentage of waters that are potable is: {:.2f}%'.format(porc))


# In[8]:


#Visualizzazione dei valori nulli
df.isnull().sum()


# In[9]:


#è possibile utilizzare il KNN impute per i valori nulli
def fill_nan(df):
    for index, column in enumerate(df.columns[:9]): #vede tra le feature eliminando potability,perciò aono 9
        #print(index, column)
        df[column] = df[column].fillna(df.groupby('Potability')[column].transform('mean'))
    return df
        
df = fill_nan(df)

df.isna().sum() #Verifica che non ci siano più valori nulli


# In[10]:


from sklearn.model_selection import train_test_split

X = df.drop(['Potability'],axis=1) #serve per utilizzare tutti i dati del dataset tranne la potabilità che diventa la y, ovvero la predizione
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=18) #stratify=y


# In[11]:


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier(),
         "Decision Tree": DecisionTreeClassifier(),
         "Naive Bayes": GaussianNB()}


# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
   
    """
   Fits and evaluates given machine learning models.
   models: a dict of different Scikit_Learn machine learning models
   X_train: training data (no labels)
   X_test: testing data (no labels)
   y_train: training labels
   y_test: test labels
   """ 
    # Set random seed
    np.random.seed(18)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit model to data
        model.fit(X_train, y_train)
        # Evaluate model and append its score to model_scores
        model_scores[name] = cross_val_score(model,
                                             X_test,
                                             y_test,
                                            scoring='accuracy',
                                            cv=5
                                            ).mean()

    return model_scores


# In[12]:


model_scores = fit_and_score(models,X_train,X_test,y_train,y_test)

model_scores


# In[13]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar(color="pink");


# In[14]:


#Final Model, avendo accuratezza maggiore 79%
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[15]:


# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Pastel1")


# In[16]:


# Classification report
print(classification_report(y_test, y_pred))


# In[17]:


# Mette in ordine le feature
def plot_features(columns, importances,n=20):
    df = (pd.DataFrame({"features": columns,
                       "feature_importances": importances})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    # Plot dataframe
    fix, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature Importance")
    ax.invert_yaxis()
    
plot_features(df.drop(['Potability'],axis=1).columns, model.feature_importances_)


# In[18]:


#RANDOM FOREST RISTRETTO
forest= RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train[['Sulfate','ph']],y_train)


# In[20]:


#Essendo le feature più importanti, l'utente può inserire i valori e farne una predizione
Sulfate= input("Indica il valore di solfato: ")
ph= input("Indica il valore di ph: ")
forest.predict([[ Sulfate,ph]])


# In[21]:


#Precisione della predizione
forest.predict_proba([[800,9]])
