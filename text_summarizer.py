import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from flask import Flask, request, render_template_string
import socket

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum') #Load pegasys library using pegasys 
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

def summarize_with_pegasus(text):
    tokens = tokenizer(text, truncation=True, padding='longest', return_tensors="pt") # input text encofing
    summary_ids = model.generate(tokens['input_ids'], max_length=60, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def calculate_top_tfidf_words(text, top_n=5):
    stop_words = stopwords.words('english')
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
    
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    
    word_scores = dict(zip(feature_names, scores))
    top_words = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    
    return top_words

def summarize_text(text):
    # Tokenize the text for stats
    words = word_tokenize(text)
    
    # Count total words, verbs, and spaces
    total_words = len(words)
    total_verbs = len([word for (word, pos) in nltk.pos_tag(words) if pos.startswith('VB')])
    total_spaces = text.count(' ')
    
    summary = summarize_with_pegasus(text) #Summary using pegasys library
    
    top_tfidf_words = calculate_top_tfidf_words(text) #Get top 5 wrods TFFID scores
    
    return summary, total_words, total_verbs, total_spaces, top_tfidf_words

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    total_words = 0
    total_verbs = 0
    total_spaces = 0
    top_tfidf_words = []

    if request.method == 'POST':
        text = request.form['text']
        summary, total_words, total_verbs, total_spaces, top_tfidf_words = summarize_text(text)
    
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat Summarization Agent</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                textarea { width: 100%; height: 200px; }
                button { margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Chat Summarization Agent</h1>
            <form method="post">
                <textarea name="text" placeholder="Enter your text here..."></textarea>
                <br>
                <button type="submit">Summarize</button>
            </form>
            {% if summary %}
                <h2>Summary:</h2>
                <p>{{ summary }}</p>
                <h3>Text Statistics:</h3>
                <ul>
                    <li>Total words: {{ total_words }}</li>
                    <li>Total verbs: {{ total_verbs }}</li>
                    <li>Total spaces: {{ total_spaces }}</li>
                </ul>
                <h3>Top 5 Words by TF-IDF Score:</h3>
                <ul>
                    {% for word, score in top_tfidf_words %}
                        <li>{{ word }}: {{ score }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        </body>
        </html>
    ''', summary=summary, total_words=total_words, total_verbs=total_verbs, total_spaces=total_spaces, top_tfidf_words=top_tfidf_words)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    try:
        port = find_free_port()
        print(f"Starting the Chat Summarization Agent on port {port}...")
        print(f"Please open your web browser and go to http://localhost:{port}")
        app.run(debug=True, port=port, use_reloader=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure all required libraries are installed.")
        print("You can install them using: pip install nltk flask transformers torch scikit-learn")
