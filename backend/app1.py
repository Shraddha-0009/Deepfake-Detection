from flask import Flask, render_template, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from deepfake_model3 import analyze_media  # Import the detection function

app = Flask(__name__)

    # --------------------------- Config ---------------------------
app.config['UPLOAD_FOLDER'] = 'frontend/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'jpg', 'jpeg', 'png'}

    # Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # --------------------------- Helper Functions ---------------------------
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    # --------------------------- Routes ---------------------------
@app.route('/')
def index():
    return render_template('t.html')  # Your frontend HTML file

@app.route('/upload', methods=['POST'])
def upload_file():
    print('[Upload] Request received')
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
        
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[Upload] Saved file as {filepath}")
            
        try:
            # Run deepfake analysis
            analysis_result = analyze_media(filepath)
            print("[Analysis Result]", analysis_result)
                
            return jsonify({
                'status': 'success',
                'filename': filename,
                'result': analysis_result
            })
        except Exception as e:
            print("[Error] Analysis failed:", str(e))
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

    # --------------------------- Main ---------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
