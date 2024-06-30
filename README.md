# Sentiment Analysis with Word Embeddings

This project is an implementation of various machine learning models for sentiment analysis using the Amazon reviews dataset.

## Project Description

This project explores sentiment analysis by building and evaluating models using word embeddings generated through Word2Vec. The steps involved in the project include dataset preparation, feature extraction, and training multiple machine learning models including simple models, feedforward neural networks, and convolutional neural networks.

## Dataset
Used the Amazon reviews dataset, which contains real reviews for kitchen products sold on Amazon. The dataset is downloadable [here](https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz).

The dataset used in this project consists of Amazon reviews. The dataset was balanced to include 250,000 reviews, with 50,000 reviews for each rating score (1 to 5). Reviews were categorized into three sentiment classes:
- Positive (ratings > 3)
- Negative (ratings < 3)
- Neutral (rating = 3)

The dataset was split into training (80%) and testing (20%) sets.

## Word Embedding

### Pretrained Word2Vec
- Loaded the pretrained "word2vec-google-news-300" Word2Vec model.
- Extracted word embeddings and checked semantic similarities using examples like "King - Man + Woman = Queen".

### Custom Word2Vec
- Trained a Word2Vec model using the Amazon reviews dataset.
- Set the embedding size to 300 and the window size to 11, with a minimum word count of 10.
- Compared the semantic similarities with those from the pretrained model.

## Models

### Simple Models
- Used Word2Vec features to train a Perceptron and an SVM model for binary classification (positive vs. negative sentiment).
- Reported accuracy values for models trained using pretrained Word2Vec, custom Word2Vec, and TF-IDF features.

### Feedforward Neural Networks
- Trained a feedforward neural network with two hidden layers (50 and 10 nodes) using average Word2Vec vectors.
- Trained both binary and ternary classification models.
- Concatenated the first 10 Word2Vec vectors for each review and trained the neural network, reporting accuracy for both binary and ternary classification.

### Convolutional Neural Networks
- Trained a two-layer CNN with output channel sizes of 50 and 10.
- Limited maximum review length to 50, padding shorter reviews with zeros.
- Trained CNN for binary and ternary classification, reporting accuracy values for each.

## Results

Reported accuracy values for:
- Perceptron and SVM models with pretrained Word2Vec, custom Word2Vec, and TF-IDF features.
- Feedforward neural networks for binary and ternary classification using average and concatenated Word2Vec vectors.
- Convolutional neural networks for binary and ternary classification.

## Conclusion

The project provided insights into the performance of different models and feature extraction techniques for sentiment analysis. The comparisons between pretrained and custom Word2Vec models, as well as between different machine learning models, highlighted the strengths and weaknesses of each approach.

## Dependencies

- Python
- Jupyter Notebook
- Gensim
- PyTorch or TensorFlow/Keras
- scikit-learn

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the Jupyter Notebook to reproduce the results.
