# from msilib.schema import Class
from abc import abstractmethod
import os
import pandas as pd

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

from abc import ABC, abstractmethod
from flask import request, jsonify, render_template, redirect
import sys

UPLOAD_FOLDER = '../dataset'

app=Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from abc import ABC

class FileFormat(ABC):
	@abstractmethod
	def datasetprocessor(self):
		pass
	def fileUpdate(self):
		pass
	def logic(self):
		self.datasetprocessor()
		self.fileUpdate()

class SaveXLSX(FileFormat):
	def __init__(self, file):
		self.file = file # Get Excel File name 
	def datasetprocessor(self):
		print("===========================================")
		# print("file",self.file)
		print("saving XLSX")
		# Read and store content of an excel file 
		read_file = pd.read_excel(self.file)

		filename = secure_filename(self.file.filename)
		filename = filename.split(".")[0]+".csv"

		# Write the dataframe object into csv file
		read_file.to_csv (os.path.join(app.config['UPLOAD_FOLDER'], filename), 
						index = None,
						header=True)
		print("===========================================")
		print("XLSX File saved successfully")
	def fileUpdate(self):
		pass

class SaveCSV(FileFormat):

	def __init__(self, file):
		self.file = file # Get Excel File name  
	
	def datasetprocessor(self):
		print("===========================================")
		print("saving CSV")
		filename = secure_filename(self.file.filename)
		self.file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print("===========================================")
		print("CSV File saved successfully")
	def fileUpdate(self):
		pass

class FileUpload:
	@app.route('/')
	@app.route('/index')
	def index():
		return render_template("index.html")

	@app.route('/uploader', methods = ['POST', 'GET'], endpoint = 'upload')
	def upload():
		if request.method == 'POST':
			f = request.files['File']

			
			filename = secure_filename(f.filename)
			print("file: =>",filename.split(".")[-1])
			f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			print("===========================================")
			print("File saved successfully")
		return render_template("index.html")


	@app.route('/api/fileUpload', methods=['POST', 'GET'], endpoint = 'datasetupload')
	def datasetupload():
		try:
			# IF THE DATASET File is ended to csv 
			# PLEASE CHECK THE Uploaded dataset Name and.. give it to
			# the if statement
			if request.method == 'POST':
				file = request.files['File']
				filename = secure_filename(file.filename)
				fileformat = filename.split(".")[-1]

				if fileformat == "csv":
					context = SaveCSV(file)
					context.logic()
					print("===========================================")
					print("Client: saved the csv file")
				elif fileformat == "xlsx":
					context = SaveXLSX(file)
					context.logic()
					print("===========================================")
					print("Client: saved the csv file.")
			
			return redirect("http://localhost:5009/", code=302)
		except:
			e = sys.exc_info()[0]
			return jsonify({'error': str(e)})
   
if __name__ == '__main__':
	PORT = os.environ.get('PORT', 5009)
	print("Port Number: ", PORT)
	app.run(debug=True, host='0.0.0.0', port=PORT)
