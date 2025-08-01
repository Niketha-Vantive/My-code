from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from openai import AzureOpenAI
from docx import Document
from dotenv import load_dotenv
import os
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = 'Vantive'
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sessions.db"
db = SQLAlchemy(app)

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_API_ENDPOINT")
)

model_deployed = "gpt-4o"  # or your current deployed model


# --- UTILS ---

def find_placeholders(doc_path):
    """Find all {placeholder} or {{placeholder}} in paragraphs and tables."""
    doc = Document(doc_path)
    placeholders = set()

    for para in doc.paragraphs:
        matches = re.findall(r"\{+(.+?)\}+", para.text)
        placeholders.update(m.strip() for m in matches)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                matches = re.findall(r"\{+(.+?)\}+", cell.text)
                placeholders.update(m.strip() for m in matches)

    return list(placeholders)


def replace_placeholders_in_docx(input_path, output_path, replacements):
    """Replace placeholders inside paragraphs and table cells."""
    doc = Document(input_path)

    # Replace in paragraphs
    for para in doc.paragraphs:
        for key, val in replacements.items():
            pattern = re.compile(r"\{+ *" + re.escape(key) + r" *\}+")
            para.text = pattern.sub(val, para.text)

    # Replace in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for key, val in replacements.items():
                    pattern = re.compile(r"\{+ *" + re.escape(key) + r" *\}+")
                    cell.text = pattern.sub(val, cell.text)

    doc.save(output_path)


# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/app')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    session['uploaded_file'] = file_path
    session['awaiting_placeholder'] = None

    placeholders = find_placeholders(file_path)
    session['expected_placeholders'] = placeholders
    session['filled_placeholders'] = {}

    return jsonify({
        "message": f"File uploaded successfully. Let's start filling in the placeholders: {placeholders}",
        "placeholders": placeholders
    })


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    expected = session.get('expected_placeholders', [])
    filled = session.get('filled_placeholders', {})
    awaiting = session.get('awaiting_placeholder')

    if awaiting:
        filled[awaiting] = user_input
        session['filled_placeholders'] = filled
        session['awaiting_placeholder'] = None

    if set(filled.keys()) == set(expected):
        input_path = session['uploaded_file']
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], 'output_filled.docx')
        replace_placeholders_in_docx(input_path, output_path, filled)
        return jsonify({"response": "✅ All placeholders filled! The document has been updated."})

    remaining = list(set(expected) - set(filled.keys()))
    if remaining:
        next_placeholder = remaining[0]
        session['awaiting_placeholder'] = next_placeholder
        return jsonify({"response": f"❓ What is the value for '{{{{{next_placeholder}}}}}'?"})
    else:
        return jsonify({"response": "No placeholders found or already filled."})


@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download')
def download_filled():
    filled_path = os.path.join(app.config["UPLOAD_FOLDER"], 'output_filled.docx')
    if not os.path.exists(filled_path):
        return "No filled document found", 404
    return send_from_directory(app.config["UPLOAD_FOLDER"], 'output_filled.docx', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
