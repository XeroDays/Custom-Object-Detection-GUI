import os
import shutil
import tarfile
import requests

'''
Class to download and evaluate default pretrained tensorflow object detection models.
'''

def downloadmodel(modelurl, download_location):
        '''
        Use this to download model from tensorflow model zoo
        args:
            modelurl, model link address from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
            ensure that you are in correct workspace when downloading model 
        '''
        filename = os.path.basename(modelurl)

        modelname = filename[:filename.index('.')]
        
        cachedir = download_location.replace("\\","/")

        os.makedirs(cachedir, exist_ok=True)

        modelpathname = os.path.join(cachedir, 'trained_models/checkpoints', modelname).replace("\\","/")
        modelpath = modelpathname + '.tar.gz'
        os.makedirs(os.path.join(cachedir, 'trained_models/checkpoints').replace("\\","/"), exist_ok=True)


        filename = modelpath.split("/")

        response = requests.get(modelurl, stream=True)
        if response.status_code == 200:
            with open(modelpath, 'wb') as f:
                f.write(response.raw.read())
        
        try:
            with tarfile.open(modelpath, 'r') as tar:
                tar.extractall(os.path.join(cachedir, 'trained_models/checkpoints').replace("\\","/"))
        #tar = tarfile.open(modelpath)
        #tar.extractall()
                tar.close()
                print("COMPLETE EXTRACTION")

        except Exception as e:
             print(e)

        shutil.move(modelpathname, os.path.join(cachedir, 'trained_models/checkpoints','Detection_Model').replace("\\","/"))
        shutil.move(os.path.join(cachedir, 'trained_models/checkpoints','Detection_Model', 'checkpoint').replace("\\","/"), os.path.join(cachedir, 'trained_models/checkpoints','Detection_Model', 'checkpoint0').replace("\\","/"))


        
            
    