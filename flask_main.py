import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import subprocess

ALLOWED_EXTENSIONS = set(['mpg','avi','mp4'])
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4'))
			p1 = subprocess.Popen(['./movrs.sh'])
			p1.wait();
			flash('File(s) successfully uploaded')
			return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4006, debug=True)
