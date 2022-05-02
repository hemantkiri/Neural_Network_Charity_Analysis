# Neural_Network_Charity_Analysis

# Overview of the analysis:
The purpose of this project to predict if the applicants that will be funded by a Charitable organization called Alphabet Soup will be successful. The data “charity_data.csv” file has 34299 rows and 12 columns information. 

# Data Processing

“IS_SUCCESSFUL” is variables considered the target for this model.
![variable considered the target](Deliverable1_step1.png)

Drop the non-beneficial ID columns, 'EIN' and 'NAME'
![Drop ‘EIN’ and ‘NAME’ columns](Deliverable1_step2.png)

Merge the one-hot encoding DataFrame
![Merge the one-hot encoding DataFrame](Deliverable1_step9.png)

Keras model Sequential
![Keras model Sequential](Keras_model_Sequential.png)

Model loss and accuracy
![Model loss and accuracy](Model_loss_and_accuracy.png)


# Summary:
The model accuracy is 71%. Model successful target was 75%.  I drop two columns ‘EIN’ and ‘Name’, determine the number of unique values for each column. I can get the 75% target if I drop more columns.
