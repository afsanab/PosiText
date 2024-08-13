# PosiText: Happiness Detection in Textual Data

## Overview

**PosiText** is a machine learning project aimed at developing a binary classifier that can accurately identify and classify expressions of happiness within textual data using natural language processing (NLP) techniques. The project utilizes the "Emotion Detection from Text" dataset from Kaggle and leverages Python libraries such as Pandas, Scikit-learn, and NLTK to preprocess the data and build the model.

## Features

- **Text Preprocessing**: Tokenization, stop word removal, and TF-IDF vectorization for feature extraction.
- **Binary Classification**: A Logistic Regression model trained to distinguish between happy and non-happy text.
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, recall, and F1-score.

## Dataset

The dataset used for this project is the ["Emotion Detection from Text"](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) dataset from Kaggle, consisting of tweets labeled with various emotional states. After filtering, 13,847 entries were used for training and testing.

## Project Structure

- **Data Preparation**: 
  - Filtering and preprocessing of the dataset.
  - Feature extraction using TF-IDF vectorization.

- **Model Development**:
  - Model selection and training using Logistic Regression.
  - Data split into training (80%) and testing (20%) sets.

- **Evaluation**:
  - Model performance assessed using accuracy, precision, recall, and F1-score.

## Results

The model achieved an overall accuracy of 72%, with the following performance on key metrics:

- **Happiness Precision**: 0.70
- **Happiness Recall**: 0.42
- **Happiness F1-Score**: 0.52
- **Non-Happiness F1-Score**: 0.81

The model demonstrated higher recall for non-happiness, indicating it is more effective at identifying non-happiness compared to happiness.

## Installation & Usage

1. **Clone the Repository**:
   git clone https://github.com/username/PosiText.git
   cd PosiText
2. **Install Dependencies**:
   Make sure you have Python installed, then run:
   pip install -r requirements.txt
3. **Run the Model**:
   Execute the main script to train and evaluate the model:
   python main.py
4. **Visualize Results**:
   Use the chart script to visualize performance metrics:
   python chart.py
## Tools & Technologies

- **Python**: Pandas, NLTK, Scikit-learn
- **NLP**: TF-IDF Vectorization
- **Model**: Logistic Regression

## Future Work

- Improve happiness detection by exploring more advanced models such as neural networks.
- Expand the dataset to include more balanced classes.
- Explore other emotions beyond happiness.

## Contributors

- Afsana Bhuiyan - [LinkedIn](https://www.linkedin.com/in/afsanabhuiyan/)
