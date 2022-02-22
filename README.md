# Churn Prediction with Neural Network

## Project Overview
![](https://atrium.ai/wp-content/uploads/2021/07/What-stops-customer-churn-Having-a-centralized-data-hub-does-and-heres-why.jpeg)

Customers are the lifeblood of any business, so companies need to understand churn if they want to grow and adapt to meet their needs. To meet the need of surviving in the competitive environment, the retention of existing customers has become a huge challenge. It is stated that the cost of acquiring a new customer is far more than that for retaining the existing one. One strategy to reduce churn is looking for trends. Analyze the churn data may help companies to notice that most customers leave within a certain timeframe or, perhaps it seems that when a customer uses a certain feature or service, they are less or more likely to churn. Thus, this project focus on:
- It is a classification problem in order to predict whether a customer will churn or not the company. The dataset contains less examples of churn customers, which means it is a classification on imbalanced data.
- Explore and Analyse the Churn Data to get insights and understand what variables/features are contributing to customer churn.
- Build a model using deep learning with TensorFlow framework with Keras, class weights to help the model learn from the imbalanced data and evaluate its performace.

The Jupyter Notebook can be accessed [here](https://github.com/ThiPauli/Churn_Predicition/blob/main/Churn_Prediction_with_Neural_Network.ipynb).

## Objectives
The task is to create a model which can predict accurately but also we are more interested in **reduce** situtations where:
- Customers left the company and the model ended up predicting that they stayed. For this reason, the goal is to evaluate **Recall** metric in order to achieve it.

## Methodology
- `Data Wrangling` - Dropped misssing or null values in the dataset.
- `Exploratory Data Analysis` - Analyzed the data and summarized the main characteristics.
- `Data Visualization` - Used histograms, bar plots, scatter plots and violin plots to visualize the data and it's characteristics.
- `Data Preprocessing` - Normalization the numerical and One Hot Encoding the non-numerical columns.
- `Feature Engineering` - Many features may be useless or noisy and removing them can result in better scores, the model might be able to classify easier and also speed up the training. Removing some features based on the exploratory data analysis.
- `Neural Network` - Built a baseline model and changed its architecture and tweaked some hyperparameters to achieve better performance.
- `Evaluation Metrics` - Accuracy, Precision, Recall, AUC (Area Under the Curve of a Receiver Operating Characteristic curve) and AUPRC (Area Under the Curve of the Precision-Recall curve).
- `Model Performace Visualizations` - Confusion Matrix, ROC curve and Precision-Recall curve.
- `Final Evaluation` - Made a robust Model Evaluation using the k-fold cross-validation procedure with StratifiedKFold from scikit-learn.
 
## Exploratory Data Analysis
Questions which driven this step:

#### `How many customers continue with the company and didn't continue?`
The data set has 7032 total samples, and 1869 (26.58%) are customers that didn't continue with the company.

#### `Who are the customers more likely to churn?`
`Demographic info about customers:`
- **Gender**: The probability of both male and female to leave the company is basically the same. So, this feature might be redundant for our problem.
- **Partner**: Customers with no partner are more likely to leave the company. It means roughly **1.6x** more likely churn than customers with partner.
- **Dependends**: Customers with no dependents (if the customers live or not with their children, parents, grandparents, etc.) are more likely to leave the company. It means roughly **2x** more likely churn than customers with dependents.
- **Senior citizen (from 65 years old)**: Customers seniors are more likely to churn than younger customers.

#### `Which factores/features might lead to its decision?`
`Services:`
- **Phone Service**: Slightly difference impacting whether the customer has phone service or not in order to leave the company. So, this feature might be redundant for our problem.
- **Multiple Lines**: Basically the same behaviour as the phone service feature.
- **Internet Service**: Customers who opted for Fiber Optic are more likely churn than customers who opted for DSL (**2.3x** more) and/or No Internet (**5.1x** more). The same is noted with customers who opted for DSL compared to No internet (**2.4x** more). Overall, customer who opted for having internet are more likely to churn.

`Positive Services Included:`
- **Online Security**: Customers who opted for having Online Security service are less likely churn (**2.2x** less).
- **Online Backup**: Customers who opted for having Online Backup service are less likely churn (**1.4x** less).
- **Device Protection**: Customers who opted for having Device Protection service are less likely churn (**1.3x** less).
- **Tech Support**: Customers who opted for having Tech Support service are less likely churn (**2.1x** less).

`Negative Services Included:`
- **Streaming TV**: Customers who opted for having Streaming TV service are more likely churn (**1.2x** more).
- **Streaming Movies**: Customers who opted for having Streaming Movies service are more likely churn (**1.2x** more).

`Contract and Payment:`
- **Contract Type**: As we evaluate this feature numerically as well, customers with short contracts are more likely churn than long contracts. Customer with one month contract are **14x** more likely churn than 2 years contract as well as **4x** more than 1 year contract. Overall, the longer the contract, the higher the probability of the customer stay with the company.
- **Paperless Billing**: Customers who opted for having Paperless Billing are more likely churn (**2.1x** more).
- **Payment Method**: Customers who opted for paying with Electronic Check are more likely churn than the other options. Almost 45% of the customers who chose Electronic Check left the company.

`Analysis on Numerical Features:`
- The **Tenure distribution** shows that around half of customers in their first year yet, churn the company. Ternure indicates the total amount of months that the customer has been with the company.
- As a consequence, it reflects that the customers who left earlier had **total charged smaller** as well as a **short period contract**.
- There is a weak correlation between the **numerical features** and the **target (Churn) variable**. 
- There is strong correlation between **tenure** and **total charges**. The more you have been using the company services, the more you have been charged.
- Customer with **higher monthly charges** are more likely churn.

Thus, doing all this analysis we can see the features which have some positive and negative impacts with the target variable as well as some features which might be redudant for this task.

## Conclusions / Results
The baseline model (`model_1`) predicted almost a half of customer who did not churn, but they actually left the company (False Negatives cases). In this specifi classification task, it would be better to have even fewer false negatives despite the cost of increasing the number of false positives. This trade off may be preferable because the costs of keeping a customer is less than trying to attract new customers. Thus, predicting false negatives mean losing even more customers without doing anything to try to prevent their decisions. The false positives may result that the company will give some promotion or benefits without the necessity at the moment. But again, it depends on the business decision in what is more relevant for the problem.

Adding class weights for the `model_2` caused the model to "pay more attention" to examples from an under-represented class and made our model identify more True Positives but also predicted wrongly more False Positives.

After evaluate the models and compare the Precision Recall Curve I could see a linear correlation between our classifier model performace for different thresholds, meaning that the proportion to get a higher recall score affects the precision score in the same magnitude.

The final evaluation showed that the last model (`model_3`) configuration works well for this dataset. The results were an increasement in the recall score from 50% to 80% but decrease in the Precision and the Accuracy scores. Imbalance data classification is a difficult task and we could continue doing more EDA and modeling to find out which features may be relevant so the model could get the most out of your minority class.
