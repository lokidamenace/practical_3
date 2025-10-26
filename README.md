# Summary and Analysis

As we want to predict if the client will subscribe to a bank term deposit, we aim to compare and evaluate the performance of various supervised algorithms to the KNN model to see which model gives the highest accuracy. 

There are some binary and categorical columns which are object types, so I need to convert them to numerical format. In addition, there are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label so I think it'll be ok to treat the unknowns as a category. All feature columns need to be encoded into numeric values so they can be fed into the model.

Based on the correlation matrix, I will focus on columns with a higher correlation to subscribed, our target variable. Therefore, I will drop columns with low correlations, including those related to campaign, job, marital, education, housing, loan, and day_of_week. In addition, duration is highly correlated to subscribed at 0.405274 (e.g., if duration=0 then subscribed=no). Because after the end of the call, subscribed is obviously known. Thus, duration should be discarded if the intention is to have a realistic predictive model.

Using KNN, I looped the algorithm through a range of k values and stored the accuracy scores of the various models, then they could be graphically displayed to determine the optimal number of clusters. The next step was to use the K-fold cross-validation technique to reduce variance and compare its performance with other supervised algorithms. In order to improve model performance, I used the StandardScaler function to scale the data. In addition, I used the Pipeline function to ensure all of the data is treated the same way for each algorithm. Finally, I used Ensemble algorithms to increase model performance.

