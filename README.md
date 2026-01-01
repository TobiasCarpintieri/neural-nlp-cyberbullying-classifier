# Neural NLP Cyberbullying Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

This project is a **Natural Language Processing (NLP)** application designed to detect and classify instances of cyberbullying in text. Leveraging a **Deep Learning model (LSTM/Neural Network)** trained on social media data, the application provides real-time classification of comments into specific toxicity categories.

The web interface is built with **Streamlit**, offering an interactive dashboard for analysis, history tracking, and data visualization.

**Live Demo:** [Click here](https://cyberbullyngauditor.streamlit.app/)


## Key Features

* **Multi-Class Classification:** accurately categorizes text into specific types of cyberbullying (e.g., Religion, Gender, Ethnicity, etc.).
* **Real-Time Inference:** Instant analysis of user-inputted text using a trained Keras model.
* **Session History:** Tracks analyzed comments during the active session.
* **Data Export:** Allows users to download the analysis report as a `.csv` file.
* **Visual Analytics:** Integrated **Plotly** charts to visualize the distribution of detected categories.


## Tech Stack

* **Core:** Python 3.11
* **Deep Learning:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Plotly Express
* **Serialization:** Pickle


## Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**

2.  **Install dependencies:**

3.  **Run the application:**


## Project Structure

```text
├── DataSet/                                      # Folder containing raw data
├── Cyberbullying_Classifier_Model_Training.ipynb # Jupyter Notebook used for model training
├── app.py                                        # Main Streamlit application script
├── model_bullying.keras                          # Trained Keras/TensorFlow model
├── encoder.pickle                                # Label encoder object
├── tokenizer.pickle                              # Text tokenizer object
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

<div align="center">
  <sub><a href="https://www.linkedin.com/in/tobiascarpintieri/" target="_blank" style="text-decoration: none; color: inherit;"><strong>Tobias Carpintieri</strong></a><br>
  Data Scientist<br>
  Built in 2026.</sub>
</div>
