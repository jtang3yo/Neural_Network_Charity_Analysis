# Neural_Network_Charity_Analysis
Neural Network Analysis for charity 

## Overview of Project 

Bek’s come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME**_AMT—Income classification
* **SPECIAL**_CONSIDERATIONS—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Deliverables: 
1. ***Deliverable 1:*** Preprocessing Data for a Neural Network Model
2. ***Deliverable 2:*** Compile, Train, and Evaluate the Model
3. ***Deliverable 3:*** Optimize the Model

## Resources: 
* Data Source: `charity_data.csv`, `AlphabetSoupCharity.h5` and `AlphabetSoupCharity_Optimzation.h5` 
* Data Tools:  `AlphabetSoupCharity_starter_code.ipynb`, `AlphabetSoupCharity.ipynb` and `AlphabetSoupCharity_Optimzation.ipynb`.
* Software: `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas`

## Analysis 

### Deliverable 1:  
#### Preprocessing Data for a Neural Network Model 
##### Deliverable Requirements: 
      Using your knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.
1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are considered the target(s) for your model?
    - What variable(s) are considered the feature(s) for your model?
2. Drop the `EIN` and `NAME` columns.
<img width="835" alt="Screen Shot 2021-08-24 at 3 48 36 PM" src="https://user-images.githubusercontent.com/82353749/130680410-d11c6dfc-c80b-4d8e-96eb-02421572ccd0.png">

3. Determine the number of unique values for each column.
<img width="835" alt="Screen Shot 2021-08-24 at 3 49 56 PM" src="https://user-images.githubusercontent.com/82353749/130680503-d79dbb0a-0091-4113-a562-7d4af5c94c20.png">

4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Create a density plot to determine the distribution of the column values.
<img width="810" alt="Screen Shot 2021-08-24 at 3 51 03 PM" src="https://user-images.githubusercontent.com/82353749/130680698-6e64f85e-b5cb-44b1-bbae-4d7c924510fd.png">
<img width="816" alt="Screen Shot 2021-08-24 at 3 51 27 PM" src="https://user-images.githubusercontent.com/82353749/130680716-dd5f0c41-d1d7-43f5-9206-03b2cf3e406e.png">
6. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, `Other`, and then check if the binning was successful.
7. Generate a list of categorical variables.
<img width="787" alt="Screen Shot 2021-08-24 at 3 52 16 PM" src="https://user-images.githubusercontent.com/82353749/130680803-e7a9ddf5-6168-478e-854c-1e91f6d5d7a8.png">
8. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
9. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
<img width="818" alt="Screen Shot 2021-08-24 at 3 52 48 PM" src="https://user-images.githubusercontent.com/82353749/130680888-f0e85b7a-1085-4b96-b035-02987b71d799.png">
10. Split the preprocessed data into features and target arrays.
11. Split the preprocessed data into training and testing datasets.
12. Standardize numerical variables using Scikit-Learn’s `StandardScaler` class, then scale the data.
<img width="820" alt="Screen Shot 2021-08-24 at 3 53 51 PM" src="https://user-images.githubusercontent.com/82353749/130681018-2a3587e9-b454-49eb-9818-45fc2ea018c5.png">

### Deliverable 2:  
#### Compile, Train, and Evaluate the Model 
##### Deliverable Requirements:

    Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

* The neural network model using Tensorflow Keras contains working code that performs the following steps:
    * The number of layers, the number of neurons per layer, and activation function are defined 
    * An output layer with an activation function is created 
    * There is an output for the structure of the model 
    * There is an output of the model’s loss and accuracy 
    * The model's weights are saved every 5 epochs 
    * The results are saved to an HDF5 file 
* <img width="819" alt="Screen Shot 2021-08-24 at 3 55 25 PM" src="https://user-images.githubusercontent.com/82353749/130681246-a7c5d454-f53b-479c-b52e-0a7a78b70973.png">
* <img width="818" alt="Screen Shot 2021-08-24 at 3 56 03 PM" src="https://user-images.githubusercontent.com/82353749/130681353-442e1195-1907-41c6-9ebe-9b288c8dd9d1.png">

### Deliverable 3:  
#### Optimize the Model
##### Deliverable Requirements:

    Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.
    
1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import your dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Create a callback that saves the model's weights every 5 epochs.
6. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.

* Attempted to reduce number of features in X variable, or changed numbers of neurons in hidden layers, or activation methods, was not able to render over 75% accuracy rate. Instead, 
* Using `keras-tuner` to generate the best optimizer, 
<img width="823" alt="Screen Shot 2021-08-24 at 4 17 20 PM" src="https://user-images.githubusercontent.com/82353749/130683940-3d3e636a-46fc-4074-94d5-9db15e56800e.png">
<img width="828" alt="Screen Shot 2021-08-24 at 4 18 01 PM" src="https://user-images.githubusercontent.com/82353749/130683999-a7ef2cfa-e438-41cd-9679-e4c648f49baf.png">
<img width="821" alt="Screen Shot 2021-08-24 at 4 18 34 PM" src="https://user-images.githubusercontent.com/82353749/130684051-83965f37-0639-45ab-b720-71ed2c24db89.png">

## Summary 

**Model Configuration:**

* hidden_nodes_layer1 = 80
* hidden_nodes_layer2 = 30
* number_input_features = 43

* Our Analysis and Deep Learning Model Results include a recommendation for how a different model could solve this classification and results.




