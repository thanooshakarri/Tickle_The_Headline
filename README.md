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
|   |____ Classifier.py           <- Neural Network
|____ Main.ipynb                  <- main function 
```

Usage:

```
Install the required libraries listed in requirements.txt by running pip install -r requirements.txt in your terminal.

Open the Main.ipynb notebook in Jupyter or any compatible platform.

The trained model is in the model folder.

The Classifier.py contains the files to train the model.

To make predictions upload the file namming it test.csv.

The test file should contain a column TEXT with the text.

The model is loaded and predicttions are made and the results are saved into predictions.csv file
```
