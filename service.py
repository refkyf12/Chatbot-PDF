from flask import Flask, request, jsonify, redirect, render_template, url_for
import os
from controller.chatbotController import pdfLoader, storeToVectorDB, askQuestion, loadVectorDB
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

vectorDB = None

## UPLOAD PDF ##

# Cek apakah file memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorDB
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
    try :
        pages = pdfLoader(file_path)
        vectorDB = storeToVectorDB(pages)
        if vectorDB._collection.count() >= 0 :
            return render_template('chatbot.html', filename=filename, count=vectorDB._collection.count(), filepath = file_path)
    except Exception as e:
        return str(e), 500

@app.route('/ask', methods=['POST'])
def ask_question_route():
    global vectorDB
    try :
        if vectorDB is None:
            vectorDB = loadVectorDB()

        if vectorDB is None:
            return "VectorDB is not initialized. Please upload a file first.", 400
        else:
            question = request.form.get('question', '').strip()

            if not question:
                return "Question is required!", 400
            
            result = askQuestion(question, vectorDB)
            
            return render_template('chatbot.html', result=result)
    except Exception as e:
        return str(e), 500

@app.route('/reset', methods=['POST'])
def reset_vectorDB():
    global vectorDB
    try:
        # Hapus direktori tempat menyimpan vector store
        persist_directory = "vector_db"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)  # Menghapus folder dan isinya

        # Reset vectorDB di memori
        vectorDB = None

        return jsonify({"message": "VectorDB has been reset successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=8080)
