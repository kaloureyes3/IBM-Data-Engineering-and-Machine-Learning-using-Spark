#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMBD0231ENSkillsNetwork26766988-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Machine Learning with Apache Spark ML**
# 

# Estimated time needed: **15** minutes
# 

# This lab goes introduces Machine Learning using Spark ML Lib (sparkml).
# 

# ![](http://spark.apache.org/images/spark-logo.png)
# 

# ## Objectives
# 

# Spark ML Library is also commonly called MLlib and is used to perform machine learning operations using DataFrame-based APIs.
# 
# After completing this lab you will be able to:
# 

# *   Import the Spark ML and Statistics Libraries
# *   Perform basic statistics operations using Spark
# *   Build a simple linear regression model using Spark ML
# *   Train the model and perform evaluation
# 

# ***
# 

# ## Setup
# 

# For this lab, we are going to be using Python and Spark (pyspark). These libraries should be installed in your lab environment or in SN Labs.
# 

# In[3]:


# When you are executing on SN labs please uncomment the below lines and then run all cells.Next again Restart the kernel and run all cells.
get_ipython().system('pip3 install pyspark==3.1.2')
get_ipython().system('pip install findspark')
import findspark
findspark.init()


# In[4]:


# Pandas is a popular data science package for Python. In this lab, we use Pandas to load a CSV file from disc to a pandas dataframe in memory.
import pandas as pd
import matplotlib.pyplot as plt
# pyspark is the Spark API for Python. In this lab, we use pyspark to initialize the spark context. 
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


# ## Exercise 1 -  Spark session
# 

# In this exercise, you will create and initialize the Spark session needed to load the dataframes and operate on it
# 

# #### Task 1: Creating the spark session and context
# 

# In[5]:


# Creating a spark context class
sc = SparkContext()

# Creating a spark session
spark = SparkSession \
    .builder \
    .appName("Python Spark DataFrames basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# #### Task 2: Initialize Spark session
# 
# To work with dataframes we just need to verify that the spark session instance has been created.
# Feel free to click on the "Spark UI" button to explore the Spark UI elements.
# 

# In[6]:


spark


# #### Task 2: Importing Spark ML libraries
# 
# In this exercise we will import 4 SparkML functions.
# 
# 1.  (Feature library) VectorAssembler(): This function is used to create feature vectors from dataframes/raw data. These feature vectors are required to train a ML model or perform any statistical operations.
# 2.  (Stat library) Correlation(): This function is from the statistics library within SparkML. This function is used to calculate correlation between feature vectors.
# 3.  (Feature library) Normalized(): This function is used to normalize features. Normalizing features leads to better ML model convergence and training results.
# 4.  (Regression Library) LinearRegression(): This function is used to create a Linear Regression model and train it.
# 

# In[7]:


from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression


# ## Exercise 2 - Loading the data and Creating Feature Vectors
# 

# In this section, you will first read the CSV file into a pandas dataframe and then read it into a Spark dataframe
# 
# Pandas is a library used for data manipulation and analysis. Pandas offers data structures and operations for creating and manipulating Data Series and DataFrame objects. Data can be imported from various data sources, e.g., Numpy arrays, Python dictionaries and CSV files. Pandas allows you to manipulate, organize and display the data.
# 
# In this example we use a dataset that contains information about cars.
# 

# #### Task 1: Loading data into a Pandas DataFrame
# 

# In[8]:


# Read the file using `read_csv` function in pandas
cars = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0225EN-SkillsNetwork/labs/data/cars.csv')


# In[9]:


# Preview a few records
cars.head()


# For this example, we pre process the data and only use 3 columns. This preprocessed dataset can be found in the `cars2.csv` file.
# 

# In[10]:


cars2 = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0225EN-SkillsNetwork/labs/data/cars2.csv', header=None, names=["mpg", "hp", "weight"])
cars2.head()


# #### Task 2: Loading data into a Spark DataFrame
# 

# In[11]:


# We use the `createDataFrame` function to load the data into a spark dataframe
sdf = spark.createDataFrame(cars2)


# In[12]:


# Let us look at the schema of the loaded spark dataframe
sdf.printSchema()


# #### Task 3: Converting data frame columns into feature vectors
# 
# In this task we use the `VectorAssembler()` function to convert the dataframe columns into feature vectors.
# For our example, we use the horsepower ("hp) and weight of the car as input features and the miles-per-gallon ("mpg") as target labels.
# 

# In[13]:


assembler = VectorAssembler(
    inputCols=["hp", "weight"],
    outputCol="features")

output = assembler.transform(sdf).select('features','mpg')


# We now create a test-train split of 75%-25%
# 

# In[14]:


train, test = output.randomSplit([0.75, 0.25])


# ## Exercise 3 - Basic stats and feature engineering
# 

# In this exercise, we determine the correlation between feature vectors and normalize the features.
# 

# #### Task 1: Correlation
# 
# Spark ML has inbuilt Correlation function as part of the Stat library. We use the correlation function to determine the different types of correlation between the 2 features - "hp" and "weight".
# 

# In[15]:


r1 = Correlation.corr(train, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))


# In[16]:


r2 = Correlation.corr(train, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))


# We can see that there is a 0.86 (or 86%) correlation between the features. That is logical as a car with higher horsepower likely has a bigger engine and thus weighs more. We can also visualize the feature vectors to see that they are indeed correlated.
# 

# In[17]:


plt.figure()
plt.scatter(cars2["hp"], cars2["weight"])
plt.xlabel("horsepower")
plt.ylabel("weight")
plt.title("Correlation between Horsepower and Weight")
plt.show()


# #### Task 2: Normalization
# 
# In order for better model training and convergence, it is a good practice to normalize feature vectors.
# 

# In[18]:


normalizer = Normalizer(inputCol="features", outputCol="features_normalized", p=1.0)
train_norm = normalizer.transform(train)
print("Normalized using L^1 norm")
train_norm.show(5, truncate=False)


# #### Task 2: Standard Scaling
# 
# This is a standard practice to scale the features such that all columns in the features have zero mean and unit variance.
# 

# In[19]:


standard_scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
train_model = standard_scaler.fit(train)
train_scaled = train_model.transform(train)
train_scaled.show(5, truncate=False)


# In[20]:


test_scaled = train_model.transform(test)
test_scaled.show(5, truncate=False)


# ## Exercise 4 - Building and Training a Linear Regression Model
# 

# In this exercise, we train a Linear Regression model `lrModel` on our training dataset. We train the model on the standard scaled version of features.
# We also print the final RMSE and R-Squared metrics.
# 

# #### Task 1: Create and Train model
# 
# We can create the model using the `LinearRegression()` class and train using the `fit()` function.
# 

# In[21]:


# Create a LR model
lr = LinearRegression(featuresCol='features_scaled', labelCol='mpg', maxIter=100)

# Fit the model
lrModel = lr.fit(train_scaled)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
#trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("R-squared: %f" % trainingSummary.r2)


# We see a RMSE (Root mean squared error) of 4.26. This means that our model predicts the `mpg` with an average error of 4.2 units.
# 

# #### Task 2: Predict on new data
# 
# Once a model is trained, we can then `transform()` new unseen data (for eg. the test data) to generate predictions.
# In the below cell, notice the "prediction" column that contains the predicted "mpg".
# 

# In[22]:


lrModel.transform(test_scaled).show(5)


# ### Question 1 - Correlation
# 

# Print the correlation matrix for the test dataset split we created above.
# 

# In[24]:


# Code block for learners to answer
r1 = Correlation.corr(test, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))


# Double-click **here** for the solution.
# 
# <!-- The answer is below:
# 
# r1 = Correlation.corr(test, "features").head()
# print("Pearson correlation matrix:\n" + str(r1[0]))
# 
# -->
# 

# ### Question 2 - Feature Normalization
# 

# Normalize the training features by using the L2 norm of the feature vector.
# 

# In[28]:


# Code block for learners to answer

normalizer_l2 = Normalizer(inputCol="features", outputCol="features_normalized", p=2.0)
train_norm_l2 = normalizer_l2.transform(train)
print("Normalized using L^2 norm\n"+str(train_norm_l2))
train_norm_l2.show(5, truncate=False)


# Double-click **here** for the solution.
# 
# <!-- The answer is below:
# 
# normalizer_l2 = Normalizer(inputCol="features", outputCol="features_normalized", p=2.0)
# train_norm_l2 = normalizer_l2.transform(train)
# rint("Normalized using L^1 norm\n"+str(train_norm_l2))
# train_norm_l2.show(5, truncate=False)
# 
# -->
# 

# ### Question 3 - Train Model
# 

# Repeat the model training shown above for another 100 iterations and report the coefficients.
# 

# In[29]:


# Code block for Question 3
normalizer_l2 = Normalizer(inputCol="features", outputCol="features_normalized", p=100.0)
train_norm_l2 = normalizer_l2.transform(train)
print("Normalized using L^100 norm\n"+str(train_norm_l2))
train_norm_l2.show(5, truncate=False)


# ## Authors
# 

# [Karthik Muthuraman](https://www.linkedin.com/in/karthik-muthuraman/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMBD0231ENSkillsNetwork26766988-2022-01-01)
# 

# ### Other Contributors
# 

# [Jerome Nilmeier](https://github.com/nilmeier/)
# 

# ## Change Log
# 

# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description      |
# | ----------------- | ------- | ------------- | ----------------------- |
# | 2022-07-14        | 0.4     | Lakshmi Holla | Added code for pyspark  |
# | 2021-12-22        | 0.3     | Lakshmi Holla | Made changes in scaling |
# | 2021-08-05        | 0.2     | Azim          | Beta launch             |
# | 2021-07-01        | 0.1     | Karthik       | Initial Draft           |
# 

# Copyright © 2021 IBM Corporation. All rights reserved.
# 
