# library imports
import os
import cv2
import face_recognition



class DataImport:
    def make_folders(self,image_folder=None):
        if not image_folder:
            if "images" not in os.listdir(os.path.join(os.getcwd(),"static")):
                os.mkdir(os.path.join(os.getcwd(),"static","images"))
        else:
            if image_folder not in os.listdir(os.path.join(os.getcwd(),"static")):
                os.mkdir(os.path.join(os.getcwd(),"static",image_folder))

    
    def read_images(self, path=None):
        if not path:
            path = os.path.join(os.getcwd(),"static","images")
        images = []
        known_face_names = []
        
        image_list = os.listdir(os.path.join(path))
        print(f"Images Present in {path} are: {image_list}")
        for img in image_list:
            current_Img = cv2.imread(os.path.join(path,img))
            images.append(current_Img)
            known_face_names.append(os.path.splitext(img)[0])
        print(f"Names Extracted from Images: {known_face_names}")

        return images, known_face_names


class Preprocessing:

    def faceEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    