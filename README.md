# NLP-Crime-Classification


Project Overview
The AI Crime Classification Assistant automates the categorization of crime reports using machine learning and Natural Language Processing (NLP). This project analyzes textual crime descriptions and predicts their respective categories, helping law enforcement and public safety organizations efficiently process and analyze large volumes of crime data.

Features
Automated Crime Categorization: Automatically classifies crime reports into predefined categories.
Text Analysis with NLP: Utilizes Natural Language Processing (NLP) to analyze textual crime descriptions.
TF-IDF Vectorization: Converts crime descriptions into numerical features.
Logistic Regression Classifier: Uses Logistic Regression to accurately categorize crime data.
Scalability: Can be scaled and adapted to different crime datasets or locations.
Usage
Prepare your crime dataset in CSV format, containing columns for crime descriptions and categories.
Load the dataset into the data variable in the crime_classification.py script.
The model will preprocess the data, train on 80% of the dataset, and classify the crime descriptions.
Predictions will be outputted with a new Predicted Category column.
Model Evaluation
The model’s performance is evaluated using standard classification metrics:

Precision
Recall
F1-Score
Accuracy
These metrics help assess how well the model classifies crime data, ensuring high-quality predictions.

Future Improvements
Integration of a web interface for real-time crime data prediction.
Expansion to handle additional crime categories.
Implementation of more advanced machine learning models to further improve accuracy.
License
This project is licensed under the MIT License.
