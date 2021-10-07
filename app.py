from flask import Flask, render_template, request, redirect,url_for
import os

#custom packages import
from data_ingestion.data_import_and_preprocessing import DataImport,Preprocessing
from image_api_functions.api_functions import API_Functions

app=Flask(__name__)

#initializing objects
data_import = DataImport()
preprocessing = Preprocessing()

data_import.make_folders()
images, known_face_names = data_import.read_images()
known_face_encodings = preprocessing.faceEncodings(images)

api_functions = API_Functions(known_face_names,known_face_encodings)


#index page
@app.route('/')
def index():
    return render_template('index.html',image={},style="display:None",alert= "display:None")



@app.route('/update')
def update():
    image_paths = [os.path.join("static","images",i) for i in os.listdir(os.path.join(os.getcwd(),"static","images"))]
    image_list = zip(known_face_names,image_paths)
    return render_template('update.html',image_list= image_list)



@app.route('/upload',methods=['POST'])
def upload():
    if request.files:
        image = request.files["image"]
    path= os.path.join("static","images",image.filename)
    image.save(path)
    global api_functions,images,known_face_names,known_face_encodings
    images, known_face_names = data_import.read_images()
    known_face_encodings = preprocessing.faceEncodings(images)
    api_functions = API_Functions(known_face_names,known_face_encodings)
    return render_template("index.html",image={},style="display:None",alert={})



@app.route("/delete/<string:static>/<string:img_folder>/<string:img_name>",methods=["GET","POST"])
def delete(static,img_folder,img_name):
    path = os.path.join(static,img_folder,img_name)
    print(path)
    os.remove(path)
    
    return redirect(url_for("update"))



@app.route('/image', methods=['POST'])
def prediction_image():
    try:
        if request.files:
            image = request.files["image"]
            # path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            path= image.filename
            image.save(path)
            prediction_img = api_functions.gen_frames(path)
            os.remove(path)
            return render_template('index.html',image=prediction_img,style={},alert= "display:None")
        else:
            return "ERROR! Occurred!!"
    
    except Exception as e:
        raise(Exception(e))
    
      
if __name__=='__main__':
    port = int(os.environ.get("PORT", 5000)) #for using port given by Heroku
    app.run(host="0.0.0.0",port=port)
    # app.run()
