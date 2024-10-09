# Natural Language Processing (NLP) for Sentiment Analysis

# Movie Review Dataset

This project is part of the AI-ML internship at Skyhighes Technologies. It aims to perform sentiment analysis on movie reviews using Natural Language Processing (NLP) techniques. The model is trained to classify movie reviews as either positive or negative based on the text. The dataset used for this project is the IMDb Movie Review Dataset, which is preprocessed and used to train a neural network model with embedding and LSTM layers.

## Project Structure

The project consists of the following files:

Here's the revised section as you requested:

---

## Project Structure

The project consists of the following files:

* **`Notebook.ipynb`**: This Jupyter notebook contains the complete code for the sentiment analysis project. It walks through the process of loading the IMDb dataset, preprocessing the text data, building and training a neural network model using embedding and LSTM layers, and evaluating the model’s performance. The notebook is structured in a step-by-step manner, allowing users to follow and run the entire sentiment classification workflow.

## Dataset Used

The dataset used is the **IMDb Movie Review Dataset**, which contains 50,000 movie reviews labeled as either positive or negative. This dataset is loaded from the `tensorflow.keras.datasets` module and includes:

- **Training data**: 25,000 labeled movie reviews
- **Test data**: 25,000 labeled movie reviews

Key parameters:
- **Vocabulary size**: 88,584 most frequent words
- **Maximum review length**: 250 words (padded or truncated)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gandharvk422/NLP_Sentiment_Analysis.git
   cd NLP_Sentiment_Analysis
   ```

2. **Install the required libraries:**

   Install the necessary Python libraries (TensorFlow, Keras, etc.) using the following command:
   
   If you don't have these libraries installed, you can manually install them:
   
   ```bash
   pip install tensorflow keras
   ```

3. **Run the Jupyter notebook:**

   Launch the Jupyter notebook to explore and run the sentiment analysis model:

   ```bash
   jupyter notebook Notebook.ipynb
   ```

## Usage

Once the environment is set up, you can open the Jupyter notebook (`Notebook.ipynb`) and run all cells to:

- Load the IMDb dataset using TensorFlow and Keras
- Preprocess the text data (padding sequences, limiting vocabulary size)
- Build a neural network model with embedding and LSTM layers
- Train the model for 10 epochs to classify reviews as positive or negative
- Evaluate the model's accuracy and loss on the test data

Model performance on the test set:
- **Accuracy**: ~86.8%
- **Loss**: ~0.40

## Acknowledgements

I would like to thank **Skyhighes Technologies** for providing the opportunity to work on this project. Special thanks to the developers of **TensorFlow** and **Keras** for their open-source libraries, and to the creators of the **IMDb Movie Review Dataset**.