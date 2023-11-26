# Music Hit Prediction Project

## Overview
This study focuses on the analysis of Spotify’s “Top 200” playlist and the development of predictive models on the song attributes to determine if a song would become a hit and remain in the top rank 70 in time based on a dataset from 2017 to 2023. Four supervised learning algorithms were tested: Logistic Regression, Random Forest, Naive Bayes, and Neural Networks. The Random Forest model achieved the highest predictive F1-score at 0.823, while the Naive Bayes model reached 0.615 as its F1-score. Both significantly outperformed Neural Network and Logistic Regression at 0.38 and 0.120 F1-score respectively. Further unsupervised learning via K-means clustering identified 5 distinct groups of songs based on song features. This clustering allows new songs to be categorized with similar existing hit songs. The results demonstrate the feasibility of applying machine learning techniques to predict musical success and understand similarities between hit songs. This hit prediction system could assist music producers and companies in planning promotional strategies.

The key steps include:

1. Exploratory Data Analysis
   - Analyze song attribute trends over time
   - Identify top artists and nationalities 
   - Check for outliers
   - Apply PCA for dimensionality reduction
2. Define "hit" songs based on rank threshold and stability  
3. Train Logistic Regression, Random Forest, Naive Bayes and Neural Network models
4. Evaluate models using accuracy, f1-score, ROC AUC etc.
5. Cluster songs using KMeans and analyze clusters
   
## Files  

**main.ipynb**
- Main Python file containing all code for analysis and modeling

**requirements.txt** 
- Contains Python package dependencies


## Setup

 ```Shell
    pip install -r requirements.txt
 ```

Post the setup execute the cells in Jupyter notebook to visualize Dataset and implement the above-mentioned models.

As an alternative executing main.py will also yield the same analysis results.

 ```Shell
     python main.py
 ```
