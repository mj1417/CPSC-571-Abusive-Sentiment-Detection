# **Abusive Sentiment Detection in Social Media Texts using Existing NLP techniques**

AbusiveSentimentDetection is a project aimed at detecting abusive language and sentiment in text data using a combination of natural language processing (NLP) techniques. A comparative analysis is drawn between two models (Naive Bayes and BERT) using dataset that contains social media comments rich in slangs and romanized script.

---

## **Setup**

Follow these steps to set up the project locally:

### **1. Clone the Repository or Download the zip folder**
```bash
git clone https://github.com/mj1417/CPSC-571-Abusive-Sentiment-Detection.git
```

### **2. Navigate to the Poject Directory**

```bash
cd AbusiveSentimentDetection
```

### **3. Install the required dependencies to run the project:**
```bash
pip install torch transformers scikit-learn pandas nltk
```

## **Dataset**

### **1. Original dataset**
The dataset is used in 80-20 split.
**training dataset.csv** contains 80% of comments to train the models and **testing dataset.csv** contains 20% of comments to test the models.

### **2. Processed dataset**
The two datasets, testing and training are processed using NLP techniques. It includes lowercasing, removing special characters, tokenization, stop word removal, lemmatization and handling slangs using a custome slang dictionary created for this project.
**processed_training_dataset.csv** contains the pre-processed training dataset using NLP techniques.
**processed_testing_dataset.csv** contains the pre-processed testing dataset using NLP techniques.

### **3. Results of predicted Naive Bayes model**
The results of the Naive Bayes model on processed testing dataset is stored in **predicted_NaiveBayes.csv**

### **4. Results of predicted BERT model**
The results of the BERT model on processed testing dataset is stored in **predicted_BERT.csv**




