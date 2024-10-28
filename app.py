from flask import Flask, render_template, request, redirect, url_for
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
import nltk
import PyPDF2  # For PDFs
from docx import Document  # For Word files
import os
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set up the model and tokenizer for question answering
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def summarize_text(text, word_limit=100, bullet_points=False, bp_limit=None):
    sentences = sent_tokenize(text.strip())
    words = []
    word_count = 0
    summary_sentences = []

    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        if bullet_points and len(summary_sentences) >= bp_limit:
            break
        if word_count + len(sentence_words) <= word_limit:
            words.extend(sentence_words)
            summary_sentences.append(sentence)
            word_count += len(sentence_words)
        else:
            if bullet_points:
                break
            else:
                summary_sentences.append(sentence)
                break

    if bullet_points:
        summary = '<ul>' + ''.join([f"<li>{sentence.strip()}</li>" for sentence in summary_sentences]) + '</ul>'
    else:
        summary = ' '.join(words)

    return summary

def extract_nouns(text, num_keywords=10):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    tagged_words = pos_tag(filtered_words)
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]

    word_freq = Counter(nouns)
    most_common_nouns = word_freq.most_common(num_keywords)
    keywords = [word for word, freq in most_common_nouns]
    return keywords

def summarize_from_url(url, word_limit=100, bullet_points=False, bp_limit=None):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])
    return summarize_text(content, word_limit, bullet_points, bp_limit), extract_nouns(content)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    return content

def extract_text_from_word(file):
    doc = Document(file)
    content = "\n".join([para.text for para in doc.paragraphs])
    return content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    url = request.form.get('url')
    file = request.files.get('file_path')
    word_limit = request.form.get('word_limit', 100)
    bullet_points_limit = request.form.get('bullet_points_limit', 10)
    summary_format = request.form.get('summary_format')

    try:
        word_limit = int(word_limit)
    except ValueError:
        word_limit = 100

    try:
        bullet_points_limit = int(bullet_points_limit)
    except ValueError:
        bullet_points_limit = 10

    bullet_points = summary_format == 'bullet_points'

    summary = ""
    keywords = []
    original_text = ""

    if file:
        filename = file.filename
        if filename.endswith('.pdf'):
            content = extract_text_from_pdf(file)
            original_text = content
            summary = summarize_text(content, word_limit, bullet_points, bullet_points_limit)
            keywords = extract_nouns(content)
        elif filename.endswith('.docx'):
            content = extract_text_from_word(file)
            original_text = content
            summary = summarize_text(content, word_limit, bullet_points, bullet_points_limit)
            keywords = extract_nouns(content)
        else:
            summary = "Unsupported file type. Please upload a PDF or Word document."
    
    elif text and not url:
        original_text = text
        summary = summarize_text(text, word_limit, bullet_points, bullet_points_limit)
        keywords = extract_nouns(text)
    elif url and not text:
        summary, keywords = summarize_from_url(url, word_limit, bullet_points, bullet_points_limit)
    else:
        summary = "Please provide either text, a URL, or upload a file."

    return render_template('result.html', summary=summary, keywords=keywords, original_text=original_text)

@app.route('/ask_chatbot', methods=['GET', 'POST'])
def ask_chatbot():
    if request.method == 'POST':
        original_text = request.form.get('original_text')
        question = request.form.get('question')
        if original_text and question:
            answer = qa_pipeline({'context': original_text, 'question': question})['answer']
            return render_template('ask_bot.html', original_text=original_text, answer=answer, question=question)
        else:
            error = "Question cannot be empty."
            return render_template('ask_bot.html', original_text=original_text, error=error, question=question)
    else:
        original_text = request.args.get('original_text')
        return render_template('ask_bot.html', original_text=original_text)

if __name__ == "__main__":
    app.run(debug=True, port=8000)