## P1: Supervised Learning: Finding Donors for CharityML
Task: On a census data set (>450000 records and 13 features), train a supervised machine learning model to predict whether an individual’s income is above $50,000 annually. <br>

**1. Exploratory data analysis and investigation of correlation between label and features**<br>

**2. Data preprocessing**
 - log-transform features of skewed distribution;
 - Normalizing numerical features;
 - one-hot encoding categorical features (dummies);
 - shuffle and split data into training (80%) and test (20%) sets<br>
 
**3. Evaluating model performance**
 - naïve predictor performance; 
 - choose 3 supervised learning ML models, and describe their real-world application, strength, weakness and why they are suitable for this dataset;
 - Create a training and predicting pipeline: for each selected model, run the model on 1%, 10% and 100% of the training model, calculate the accuracy on training and test data set, and the time they take;<br>
 
**4. Improving results**
 - chose the best model based on performance and running time; 
 - describe the model in layman’s term;
 - model tuning with GridSearchCV<br>
 
**5. Feature importance**
 - feature relevance by exploratory investigation on original data set;
 - extract feature importance from the trained model;
 - feature selection and its effect on model performance
