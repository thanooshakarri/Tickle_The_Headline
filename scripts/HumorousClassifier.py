from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
import re
import transformers
import datasets
import tensorflow as tf
from sklearn.metrics import mean_squared_error

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-uncased")

class Humorous_Classifier():
    def __init__(self):
        pass
        
    def pre_processing(self,data_path="data/humorous_headlines.csv"):
        #combining the original headlines with the humorous headlines to get one text vector
        def replace(sentence,word):
            pattern = r'<(.*?)\/>'
            return re.sub(pattern,word,sentence)
        #loading data
        data=pd.read_csv(data_path)
        #using regualr expressions to replace highlighted (<highlighted word/>)words
        pattern = r'<(.*?)\/>'
        #get the replaced headline as a new column
        data["new"]=data.apply(lambda x: replace(x.original,x.edit),axis=1)
        #concatenate original headline with humorous headline(with replaced word) which are separated by [SNIPPET]
        data["TEXT"]=data["original"]+" [SNIPPET] "+data["new"]
        #droped the rest other columns that are unnceceray
        data.drop(["original","edit","new","grades"],axis=1,inplace=True)
        #save this data into a new csv file
        data.to_csv("data/pre_processed_data.csv",index=False)
    
    def train_test_split(self,preprocessed_data_path="data/pre_processed_data.csv"):
        data=pd.read_csv(preprocessed_data_path)
        label=data["meanGrade"].astype("float")
        data.drop(["meanGrade"],inplace=True,axis=1)
        X = data  # Features (text data)
        y = label  # Target variable

        # Split the data into training and testing sets (e.g., 70% train, 15% development, 15% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        pd.concat([X_train,y_train],axis=1).to_csv("data/train.csv",index=False)
        pd.concat([X_test,y_test],axis=1).to_csv("data/test.csv",index=False)
        pd.concat([X_dev,y_dev],axis=1).to_csv("data/dev.csv",index=False)

    def tokenize(self,examples):
        #tokenizing using bert
        return tokenizer(examples["TEXT"], truncation=True, max_length=64,
                        padding="max_length",return_tensors="tf")
    

    def fit(self, train_path="data/train.csv", devlopment_path="data/dev.csv"):
        # load the CSVs into Huggingface datasets
        self.humorous_headlines_dataset = datasets.load_dataset("csv", data_files={
            "train": train_path, "validation": devlopment_path})
        #traines the model using bidirectional neural network layer and saves the model into a model file
        self.humorous_headlines_dataset = self.humorous_headlines_dataset.map(self.tokenize, batched=True)

        train = self.humorous_headlines_dataset["train"].to_tf_dataset(
            columns="input_ids",
            label_cols="meanGrade",
            batch_size=16,
            shuffle=True)
        devlopment = self.humorous_headlines_dataset["validation"].to_tf_dataset(
            columns="input_ids",
            label_cols="meanGrade",
            batch_size=16)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,8))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
                1,
                activation='linear'
                ))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.mean_squared_error,
            metrics=["mean_squared_error"])
        # fit the model to the training data, monitoring F1 on the dev data
        model.fit(
            train,
            epochs=10,
            batch_size=16,
            validation_data=devlopment,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="model",
                    monitor="val_root_mean_squared_error",
                    mode="max",
                    save_best_only=True)])
    
    def predict(self,path="data/test.csv"):
        #takes the data makes predicions and saves them into a predictions.csv file
        # loading saved model
        model = tf.keras.models.load_model("model")

        # load the data for prediction
        test_data = pd.read_csv(path)

        # create input features in the same way as in train()
        self.humorous_headlines_dataset = datasets.Dataset.from_pandas(test_data)
        self.humorous_headlines_dataset = self.humorous_headlines_dataset.map(self.tokenize, batched=True)
        
        tf_dataset = self.humorous_headlines_dataset.to_tf_dataset(
            columns="input_ids",
            batch_size=16)
        
        predictions = model.predict(tf_dataset)
        #rounding of predictions to one decimal values
        formatted_predictions = [float("{:.1f}".format(pred)) for pred in predictions.flatten()]

        pd.concat([test_data[["id"]],pd.DataFrame({"LABEL":formatted_predictions})],axis=1).to_csv("data/predictions.csv",index=False)
    
    def root_mean_square_error(self):
        test_data=pd.read_csv("data/test.csv")
        predictions=pd.read_csv("data/predictions.csv")
        mse=mean_squared_error(test_data[["meanGrade"]],predictions[["LABEL"]])
        print(numpy.sqrt(mse))
        