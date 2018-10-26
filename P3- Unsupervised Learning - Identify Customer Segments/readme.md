## P3- Unsupervised Learning – Identify Customer Segments
Task: apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. <br>

Model training is performed on a general population dataset. The general population dataset was cleaned before training. Then the trained model is used to identify customer segments. 
### Step 0: Load and explore training data
-	Load general population data. It has ~ 900,000 records and 85 columns.
### Step 1: Preprocessing training data
#### Step 1.1: Assess missing values
-	Write a function that converts missing data code to numpy NaN
-	Investigate patterns in the amount of missing data in each column.
-	Drop columns that contain exceptionally large amount of missing values (outliers).
-	Assess missing values in each row.
-	Split data into two data frames, one with less than 5 missing values in each row, and the other with more. Clustering model will train on the data frame with less missing values in each row.
#### Step 1.2 Select and re-encode features
-	Re-encode categorical features: convert non-numerical categorical features to numerical; drop categorical features that have more than 3 categories. 
-	Engineer mixed-type features: 2 features contain mixed information. For each feature, I break them into 2 features that encode one information each. The original mixed features are dropped. 
#### Step 1.3 Create cleaning function
A function was created that performs all the above cleaning jobs, including re-encoding missing values, dropping columns, re-encoding categorical features and engineering mixed-type features.  This cleaning function will be used to clean customer data set. 
### Step 2: Feature transformation
#### Step 2.1 Feature scaling
-	Impute missing values with median
-	Apply standard scaling
#### Step 2.2 Dimensionality reduction
-	PCA on the data
-	Plot explained_variance_ratio of individual PC and cumulative variance
-	Decide the optimal number of PC based on the cumulative variance curve
#### Step 2.3 Interpret principal components
-	Define a function that maps feature weights in each PC
-	Sort feature weights in descending order of the absolute value
-	Interpret the top 3 PC on their corresponding top 10 features

### Step 3 Clustering
#### Step 3.1: Apply clustering to general population
-	Select a random subset (10%) dataset of the pca data as sample dataset
-	Run a number of KMeans clustering with k ranging 5-30
-	Plot the scores of different KMeans, and decide on the optimal number of clusters
-	Refit KMeans model on the complete pca data
#### Step 3.2 Applying clustering to customer data
-	Clean customer data with previous defined cleaning function
-	Apply imputation, scaling, PCA transformation, and KMeans to the cleaned data, using the objects resulted from general population data. 
#### Step 3.3 Compare customer data and general population
-	Plot percentage of population of each cluster of customer and general population data
-	Identified the cluster that is overrepresented in customer data set. Customers in this cluster lives in regions that have low movement, high number of small houses, low number of larger houses and low number of business buildings, and are relatively wealthy.  These customers are mostly male, with high combative attitude. They are mostly dominant-minded, critical-minded, not dreamful, not socially-minded, not family-minded, not cultural-minded, and they are very likely an investor.
-	Identified the cluster that is underrepresented in customer data set. The regions these people live in, and their personalities are exactly opposite of those who are overrepresented in customer data. 
