# Task

See https://uazhlt-ms-program.github.io/ling-582-course-blog/assignments/course-project


Our aim is to investigate how humor is created by making little adjustments to news headlines. Even though they are brief, news headlines offer a special chance to investigate comedy by employing the succinct presentation of significant information. The limitation to allowing just small adjustments enables a targeted study of the specific modifications that turn a serious headline into a lighthearted one.

We utilized the Humicroedit dataset, designed specifically for studies on computational humor. It contains news headlines with highlighted words (formatted as < word />) along with synonyms and assessments of how funny they are. The funniest score is 3, with 0 being not funny and 3 being hilarious. Training a model to forecast the level of funniness based on the headline and its replacement word is the problem at hand.


Files:

```
├── README.md                     <- Introduction of repository
├── model                         <- Trained model
├── requirements.txt              <- Python packages requirement file
├── data                          <- Dataset
|   |____ train.csv               <- Train data
|   |____ test.csv                <- Test data
|   |____ dev.csv                 <- Validation data
|   |____ pre_processed_data.csv  <- pre_processed data
├── src                           <- Source code
|   |____ HumorClassifier.py      <- Neural Network
|____ Main.ipynb                  <- main function 
```

Usage:

```
Install the required libraries listed in requirements.txt by running pip install -r requirements.txt in your terminal.

Open the Main.ipynb notebook in Jupyter or any compatible platform.

The trained model is in the model folder.

The HumorClassifier.py contains the files to train the model.
class HumorClassifier:
functions in the class
  pre_processing  - subsitutes humor word and combines both humor and headline.
  train_test_split- splits the train into train, validation and test(held out data)
  tokenize        - tokenizes data and gives embedding of bert-large-uncased model
  fit             - fits the model
  predict         - predicts labels by loadoing model from model file
  root_mean_square- calulates root mean squared error

To train model:
1. pre_processing
2. tokenize
3. fit
4. predict
5. root_mean_squared_error

To use existing model to predict:
1. pre_processing
2. predict

To make predictions upload the file namming it test.csv.

The test file should contain a column TEXT with the text.

The model is loaded and predicttions are made and the results are saved into predictions.csv file
```
