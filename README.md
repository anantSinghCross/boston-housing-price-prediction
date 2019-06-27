# Boston Housing Price Prediction

The Boston Housing Dataset consists of price of houses in various places in Boston. Alongside with price, the dataset also provide information such as Crime (CRIM), areas of non-retail business in the town (INDUS), the age of people who own the house (AGE), and there are many other attributes that available here https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names.


You can use the Boston dataset that I have provided in the repository or you can load it from sklearn's library itself.

## Overview of what we have to do

* Loading the dataset
* Evaluating the model using k-fold cross-validation
* Training the model on the dataset
* Saving the model to use it whenever required without the need to retrain it

#### What you see in the repo

* *api.py*- The backend Flask app
* *boston.csv*- Dataset
* *model.h5*- Trained model weights (created by *trainer.py* program)
* *model.json*- Trained model structure in JSON format (created by *trainer.py* program) (I want you to see this file as it really gives a lot of usefull information about the structure and different layers being used.)
* *trainer.py*- The main program you'll be most interested in

#### Check your Libraries

You should have your libraries updated. You will need the following libraries:
* `numpy`
* `keras`
* `pandas`
* `json`
* `sklearn`
* `flask`

### Steps to get it up and running

1. The *trainer.py* loads the dataset available within the repository and first cross-validates it using 10-fold method and then fits the dataset onto the model.
The trained model is then *jsoned* so as to make it easier to load at the time of prediction.

2. The *api.py* makes the server and runs as the backend program. This takes in the values from the form and uses them for prediction.
To run the *api.py* head to the command prompt, change the directory and type `python api.py`.

3. Finally, open the web-browser and go to http://localhost:8001/ to get your *index.html* running.

### What you should see

* **bostonIndex.html**

![bostonIndex.html](bostonIndex.JPG)

* **predictPage.html**

![predictPage.html](predictPage.JPG)

### Brief of what's happening

* The *index.html* takes in values which are sent to the *api.py* running in the background which does all the prediction.
* *api.py* loads the saved model and sends back the prediction in the Jinja variable (present in the HTML file).

***

#### Note

For those who're having any doubts about any aspect of this project, it's literally all over the internet you just need to google it. Since it was one of my first machine learning projects I had a really hard time gathering info about deploying the model. So I thought why not just clump all this info at one place and help some growing minds!
