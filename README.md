# AI-Powered Text Summarizer

AI-Powered Text Summarizer is an interactive web app built with **Windsurf — the AI agentic IDE** — using Python and Streamlit. It uses NLP techniques like spaCy sentence segmentation and TF-IDF ranking to generate concise extractive summaries from text or documents.

Users can paste text or upload TXT, PDF, or DOCX files, customize summary length, format (paragraph or bullets), and font size. The summary can be edited, copied, or downloaded.

This project showcases practical NLP application with a clean GUI and Docker containerization for easy deployment.

---

## Key Features

* Upload files (`.txt`, `.pdf`, `.docx`) or paste text directly.
* Extractive summarization of text.
* Customizable number of sentences in the summary (1-10).
* Summary output formatting: paragraph or bullet points.
* Font size customization.
* Edit summary inline.
* Copy summary to clipboard or save as text file.
* Uses spaCy (`en_core_web_sm`) and NLTK (`punkt`) models.
* Docker-ready for easy deployment.

---

## Technologies Used:

Python 3.10+

Windsurf IDE

Streamlit for frontend GUI

spaCy for NLP sentence segmentation

scikit-learn TF-IDF vectorizer for sentence scoring

PyMuPDF and python-docx for file text extraction

Docker for containerization

---

## Installation

1. Clone this repository:
 ```bash 
git clone https://github.com/PrathyushaePigilam/textsum.git
cd textsum
```

2. Install dependencies
 ```bash  
 pip install -r requirements.txt
```

## Usage
Run the Streamlit app locally:
 ```bash
streamlit run app.py
```
Open your browser and go to http://localhost:8501.


## Docker
### Build the Docker image
```bash
docker build -t textsum-app .
```
### Run the Docker container
```bash
docker run -p 8501:8501 textsum-app
```
Open your browser and visit:
http://localhost:8501






