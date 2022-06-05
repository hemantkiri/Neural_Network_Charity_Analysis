# Neural_Network_Charity_Analysis

# Overview of the analysis:
The purpose of this project to predict if the applicants that will be funded by a Charitable organization called Alphabet Soup will be successful. The data “charity_data.csv” file has 34299 rows and 12 columns information. 

# Data Processing

## “IS_SUCCESSFUL” is variables considered the target for this model.

```
# Split our preprocessed data into our features and target arrays (Deleverable1 step#1)
y = application_df["IS_SUCCESSFUL"].values
X = application_df.drop(["IS_SUCCESSFUL"],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```

## Drop the non-beneficial ID columns, 'EIN' and 'NAME'

```
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'. (Deleverable1 step#2)
application_df = application_df.drop(columns = ["EIN", "NAME"])
application_df.head()
```

| APPLICATION_TYPE | AFFILIATION | CLASSIFICATION | USE_CASE | ORGANIZATION | STATUS | INCOME_AMT | SPECIAL_CONSIDERATIONS | ASK_AMT | IS_SUCCESSFUL |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| T10 | Independent | C1000 | ProductDev | Association | 1 | 0 | N | 5000 | 1 |
| T3 | Independent | C2000 | Preservation | Co-operative | 1 | 1-999 | N | 108590 | 1 |
| T5 | CompanySponsored | C3000 | ProductDev | Association | 1 | 0 | N | 5000 | 0 |
| T3 | CompanySponsored | C2000 | Preservation | Trust | 1 | 10000 – 24999 | N | 6692 | 1 |
| T3 | Independent | C1000 | Heathcare | Trust | 1 | 100000 – 499999 | N | 142590 | 1 |


## Merge the one-hot encoding DataFrame

```
# Merge one-hot encoded features and drop the originals (Deleverable1 step#9)
application_df = application_df.merge(encode_df, left_index = True,right_index = True)
application_df = application_df.drop(columns = application_cat)
application_df.head()
```
| STATUS | ASK_AMT | IS_SUCCESSFUL | APPLICATION_TYPE_Other | APPLICATION_TYPE_T10 | APPLICATION_TYPE_T19 | APPLICATION_TYPE_T3 | APPLICATION_TYPE_T4 | APPLICATION_TYPE_T5 | APPLICATION_TYPE_T6 | INCOME_AMT_1-9999 | INCOME_AMT_10000-24999 | INCOME_AMT_100000-499999 | INCOME_AMT_10M-50M | INCOME_AMT_1M-5M | INCOME_AMT_25000-99999 | INCOME_AMT_50M+ | INCOME_AMT_5M-10M | SPECIAL_CONSIDERATIONS_N | SPECIAL_CONSIDERATIONS_Y |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| 1 | 5000 | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 1 | 108590 | 1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 1 | 5000 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 1 | 6692 | 1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 1 | 142590 | 1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |


## Model loss and accuracy

```
# Train the model
fit_model = nn.fit(X_train, y_train,epochs = 100, callbacks = [cp_callback])
```

# Summary:
The model accuracy is 71%. Model successful target was 75%.  I drop two columns ‘EIN’ and ‘Name’, determine the number of unique values for each column. I can get the 75% target if I drop more columns.

My recommendation is to use Random forest classifiers, becasue
(19.5.4)
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model.
Random forest models use a number of weak learner algorithms (decision trees) and combine their output to make a final classification (or regression) decision. 
Both output and feature selection of random forest models are easy to interpret, and they can easily handle outliers and nonlinear data.
