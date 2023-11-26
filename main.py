# %%
"""
# Import required libraries
"""

# %%
import os
# libraries for data reading and modification
import numpy as np
import pandas as pd
# libraries to visualize data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import random
import warnings
warnings.filterwarnings("ignore")

# libraries required to build ML and DNN models
from sklearn.preprocessing import PolynomialFeatures, normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# %%
"""
# Preprocessing Dataset
"""

# %%
spotify = pd.read_csv(input("Enter Dataset path: "), delimiter = ';')
spotify.head()

# %%
spotify.shape

# %%
spotify.describe()

# %%
spotify.nunique() #2208 unique artists from 74 unique nationalities. One song at max played by 9 artists, 9161 unique songs

# %%
"""
## PCA
"""

# %%
#Removing the unwanted rows and columns
spotify.drop("id" ,axis =1 ,inplace= True)
spotify.drop("Song URL" ,axis =1 ,inplace= True)
#Creating dataset of song attributes only
spotify_songatts = spotify.iloc[:,4:10]
spotify_songatts["Rank"] = spotify.iloc[:,0]
# spotify_songatts.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(spotify_songatts.corr(),
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    annot=True,
    fmt=".2f",
    square=True, ax=ax)

# %%
#Normalizing the values in dataset to between 0 and 1
spotify_songatts1 = StandardScaler().fit_transform(spotify_songatts)
#Finding the covariance of new dataset. spotify_songatts.corr() can also be used
cov = (spotify_songatts1.T @ spotify_songatts1) / (spotify_songatts1.shape[0] - 1)

# %%
#Finding the eigen values from covariance matrix. Performing eigen decomposition. Refer: https://towardsdatascience.com/a-step-by-step-introduction-to-pca-c0d78e26a0dd
eig_values, eig_vectors = np.linalg.eig(cov)
#Determining which princial components to select
idx = np.argsort(eig_values, axis=0)[::-1]
sorted_eig_vectors = eig_vectors[:, idx]
cumsum = np.cumsum(eig_values[idx]) / np.sum(eig_values[idx])
xint = range(1, len(cumsum) + 1)
plt.plot(xint, cumsum)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.xticks(xint)
#PCA not possible with song attributes. Maximum we can do is 6 to 5. Since each attribute adding something to the fuct

# %%
"""
## **Outlier Analysis**
"""

# %%
outlier= list(spotify.select_dtypes('float64').columns)
fig = px.box(spotify, y=outlier)
spotify.plot(kind = "box" , subplots = True , figsize = (50,50) , layout = (20,20))

# %%
"""
## **Other Analysis of data**
"""

# %%
"""
**Which nations produce most songs ?**
"""

# %%
nationality_counts = spotify['Nationality'].value_counts()
continent_order = spotify['Continent'].unique()
# nationality_counts

# %%
nationality_df = pd.DataFrame({'Nationality': nationality_counts.index, 'Count': nationality_counts.values})
fig_nationality = px.bar(nationality_df, x='Nationality', y='Count', color='Nationality',
                         title='Distribution of Nationalities and Continents')
fig_nationality.update_layout(xaxis_title='Nationality', yaxis_title='Count', xaxis_tickangle=-45)
fig_nationality.show()

# %%
"""
**Which artist produces most songs ?**
"""

# %%
spotify_artist = spotify.iloc[:,12:15]
spotify_artist = spotify_artist.drop(['# of Nationality'], axis=1)
spotify_artist["title"] = spotify.iloc[:,1]
spotify_artistv1 = spotify_artist.drop_duplicates(subset=None, keep="first", inplace=False)
# spotify_artistv1

# %%
artist_song_counts = spotify_artistv1.groupby('Artist (Ind.)')['title'].count().reset_index()
top_10 = artist_song_counts.sort_values(by=['title'], ascending=False).iloc[0:10,:]

# %%
fig = px.bar(top_10, x='Artist (Ind.)', y='title')
fig.update_xaxes(title_text='Artist Name')
fig.update_yaxes(title_text='Number of Songs')
fig.update_layout(title='Number of unique Songs per Artist')
fig.update_layout(xaxis_tickangle=-45)

fig.show()

# %%
"""
**How song attributes change of the years ?**
"""

# %%
spotify['Date'] = pd.to_datetime(spotify['Date'],errors='ignore')
spotify['Year'] = spotify['Date'].dt.year
grouped_data = spotify.groupby('Year').agg({'Danceability': 'mean', 'Energy': 'mean','Loudness': 'mean','Speechiness': 'mean','Acousticness': 'mean','Instrumentalness': 'mean'}).reset_index()
fig = px.scatter(grouped_data, x='Year', y=['Danceability', 'Energy','Speechiness','Acousticness','Instrumentalness'],
                 labels={'Danceability': 'Danceability', 'Energy': 'Energy','Speechiness': 'Speechiness','Acousticness': 'Acousticness','Instrumentalness': 'Instrumentalness' ,'Year': 'Year'},
                 title='Song Attributes Over the Years')

fig.update_traces(marker=dict(size=8))

fig.show()

# %%
new_df = pd.DataFrame([],columns=spotify.columns)

# %%
"""
# Hit song detection
"""

# %%
best_songs={}
if 'req_met_songs.txt' in os.listdir():
    c=True
    with open('req_met_songs.txt','r') as f:
        str_song = f.read()
        str_song = str_song.split(',')
        for i in str_song:
            if '{' in i:
                i=i.replace('{','')
            if  "}" in i:
                i=i.replace('}','')
            if c:
                best_songs[i.split(':')[0][1:-1]]=int(i.split(':')[1])
                c=False
            else:
                best_songs[i.split(':')[0][2:-1]]=int(i.split(':')[1])
else:
    for i in spotify['id'].unique():
        fdf = spotify[spotify['id']==i]
        fdf['Rank Diff'] = fdf['Rank'].diff()
        if fdf['Rank'].mean()<70 and fdf['Rank'].corr(fdf['Rank Diff'])<0.1:
            best_songs[i]=1
        else:
            best_songs[i]=0
    with open('req_met_songs.txt','w') as f:
        f.write(str(best_songs))
        


# %%
result_best_song=[]
spotify = pd.read_csv(input("Enter Dataset path: "), delimiter = ';')
for i in spotify['id']:
    result_best_song.append([best_songs[i]]) 
spotify['results'] = pd.DataFrame(result_best_song)

new_df = spotify.drop_duplicates(subset='id', keep='first')


# %%
"""
# Resampling dataset
"""

# %%
process_df = spotify.drop(['Rank','Title','Date','Artists','# of Artist','Artist (Ind.)','# of Nationality','Nationality','Continent','Points (Total)','Points (Ind for each Artist/Nat)','id','Song URL'],axis=1)
true_data = process_df[process_df['results']==1]
false_data = process_df[process_df['results']==0]
true_data_upsampled = resample(true_data, replace=True, n_samples=len(false_data), random_state=42)
balanced_data = pd.concat([false_data, true_data_upsampled])
balanced_data = balanced_data.sample(frac=1, random_state=42)
balanced_data.head()

# %%
"""
# Evaluation metrics
"""

# %%
def eval_metric(Y_test,X_test,model):
    accuracy = model.score(X_test, Y_test)
    print("Accuracy:", accuracy)
    res = model.predict(X_test)
    sns.heatmap(confusion_matrix(Y_test,res),annot=True)
    print('F1-score is: ',f1_score(Y_test,res))
    print(classification_report(Y_test,res))
    fpr, tpr, thresholds = roc_curve(Y_test, res)
    print(f"FPR (False Positive Rate): {fpr}")
    print(f"TPR (True Positive Rate): {tpr}")
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

# %%
"""
# Model initialization and execution
"""

# %%
X = balanced_data.drop('results',axis=1)
y = balanced_data['results']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


LR_model = LogisticRegression()
LR_model.fit(X_train, Y_train)
print("Logistic Regression ")
eval_metric(Y_test,X_test,LR_model)

RF_model = RandomForestClassifier(max_depth=5)
RF_model.fit(X_train, Y_train)
print('Random Forest')
eval_metric(Y_test,X_test,RF_model)

NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)
print('Naive Bayes')
eval_metric(Y_test,X_test,NB_model)

# %%
# Plots the feature importances for each feature. This is a bar plot and will be plotted on the subplots
importances = RF_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF_model.estimators_], axis=0)
forest_importances = pd.Series(importances)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
custom_optimizer = Adam(lr=0.001)
model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))  
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])
model.summary()

# %%

history = model.fit(
    x_train, Y_train, 
    epochs=50, 
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)
model.save('final_NNmodel.h5')

# %%
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss for Trial 3')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy for Trial 3')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
res = []
for i in model.predict(X_test):
    res.append(int(i[0]))
res=np.array(res)
accuracy = accuracy_score(Y_test,res)
print("Accuracy:", accuracy)
print("F1-score",f1_score(Y_test,res))
print(classification_report(Y_test,res))
sns.heatmap(confusion_matrix(Y_test,res),annot=True)
fpr, tpr, thresholds = roc_curve(Y_test, model.predict(X_test))
print(f"FPR (False Positive Rate): {fpr}")
print(f"TPR (True Positive Rate): {tpr}")
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# %%
process_df = new_df.drop(['Rank','Date','Title','Artists','# of Artist','Artist (Ind.)','# of Nationality','Nationality','Continent','Points (Total)','Points (Ind for each Artist/Nat)','Song URL','results','id'],axis=1)
X=process_df.to_numpy()
y = new_df['Rank'].to_numpy()

rank_vs_count={}
for i in new_df['Rank'].unique():
    rank_vs_count[i]=new_df[new_df['Rank']==i]['Rank'].to_numpy().sum()

# %%
wcss = [] 
for i in range(1, 41):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 41), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# %%
kmeNS = KMeans(n_clusters=5,random_state=0,n_init='auto').fit(X)
kmeNS.cluster_centers_

# %%
cluster_dist={}
for i in kmeNS.labels_:
    if i in cluster_dist.keys():
        cluster_dist[i]+=1
    else:
        cluster_dist[i]=1
plt.bar(cluster_dist.keys() ,cluster_dist.values(), color='blue')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Graph')
plt.show()

# %%
print(cluster_dist)

# %%
cluster_performance={}
for i in range(len(kmeNS.labels_)):
    if kmeNS.labels_[i] in cluster_performance.keys():
        cluster_performance[kmeNS.labels_[i]]+=spotify[spotify['id']==new_df['id'].iloc[i]]['results'].to_numpy()[0]
    else:
        cluster_performance[kmeNS.labels_[i]]=0
    
print(cluster_performance)

# %%
plt.bar(cluster_performance.keys() ,cluster_performance.values(), color='blue')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Graph')

plt.show()

# %%
cluster_centers = np.array([[ 7.36164732e-01,  7.40245003e-01, 9.40435203e-02, 1.73951644e-01,  2.74306899e-03,  6.98611863e-01],
 [ 7.48773531e-01,  6.06065031e-01, 3.17455220e-01, 1.86485454e-01,  1.88819167e-03,  4.74222476e-01],
 [ 5.48552673e-01,  3.84758648e-01,   7.08490566e-02, 6.75198113e-01,  1.17963836e-02,  3.65423742e-01],
 [ 5.73269504e-01,  5.81758865e-01,  8.60070922e-02, 3.12957447e-01,  6.12588652e-01,  3.31680851e-01],
 [ 6.22563083e-01,  6.55714138e-01,  7.77321120e-02, 1.29418942e-01,  4.47597649e-03,  3.27174559e-01]])
cluster_centers_transposed = cluster_centers.T

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
bar_positions = np.arange(len(cluster_centers_transposed[0]))
labels = np.array(["Danceability","Energy","Speechiness","Acousticness","Instrumentalness","Valence"])

for i in range(len(cluster_centers_transposed)):
    ax.bar(bar_positions + i * bar_width, cluster_centers_transposed[i], bar_width, label=labels[i])

ax.set_xlabel('Clusters')
ax.set_ylabel('Values')
ax.set_title('Cluster Centers for Each Dimension')
ax.set_xticks(bar_positions + bar_width * (len(cluster_centers_transposed) - 1) / 2)
ax.set_xticklabels([f'Cluster {i+1}' for i in range(len(cluster_centers))])
ax.legend()

# %%
cluster_centers = np.array([[ -4.26171446e+03],
       [-6.48928280e+03],
       [ -9.20883902e+03],
       [ -8.11117135e+03],
       [ -5.47634218e+03]])


num_clusters = len(cluster_centers)

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(range(num_clusters), cluster_centers[:, 0])
ax.set_ylabel('Dimension Value')
ax.set_xlabel('Cluster')
ax.set_title('Bar Plot: Single Dimension across Clusters')

plt.show()