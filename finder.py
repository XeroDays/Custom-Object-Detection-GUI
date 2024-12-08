import os, fnmatch
from PIL import Image

class finder():
    def __init__(self, Search_Folder):
        self.searchpath =Search_Folder
        self.locations = []
        
        # Ensure funciton doesnt error when file is not found
        self.config     = ''
        self.trec       = ''
        self.imgpath    = ''
        self.annotspath = ''
        self.testrec    = ''
        self.results    = ''
        self.lables     = ''
        self.modelname  = 'Detection_Model'
        self.modelpath  = ''
        self.recordpath = ''
        self.width      = ''
        self.height     = ''
        
        self.findmodel()
        #for i in self.filename:
        #    self.locations.append(self.find_files(i, Search_Folder))
            
    def findmodel(self):
        for root, dir, files in os.walk(self.searchpath):
            
            for name in dir:
                if "Images" in name.split():
                    self.imgpath = os.path.join(root, 'Images', 'Resized').replace("\\","/")
                    self.recordpath = root
                    self.origin_Path = os.path.join(root, 'Images', 'Original')

                    if os.path.exists(self.origin_Path):
                        for file in os.listdir(self.origin_Path):
                            if (file.endswith('.JPG') or file.endswith('.jpg') or 
                                file.endswith('.PNG') or file.endswith('.png') or 
                                file.endswith('.BMP') or file.endswith('.bmp')):

                                fullpath = os.path.join(self.origin_Path, file)

                                im = Image.open(fullpath)
                                self.height, self.width = im.size

                elif 'Annotations' in name.split():
                    self.annotspath = os.path.join(root, 'Annotations').replace("\\","/")
                elif 'Detection_Model' in name.split():
                    self.modelpath = os.path.join(root, 'Detection_Model').replace("\\","/")
                    
            for basename in files:
                if fnmatch.fnmatch(basename, '*.config'):
                    self.config = os.path.join(root, basename).replace("\\","/")
                elif fnmatch.fnmatch(basename, 'test.record'):
                    self.trec = os.path.join(root, basename).replace("\\","/")
                elif fnmatch.fnmatch(basename, 'train.record'):
                    self.testrec = os.path.join(root, basename).replace("\\","/")
                elif fnmatch.fnmatch(basename, '*.h5'):
                    self.results = os.path.join(root, basename).replace("\\","/")
                elif fnmatch.fnmatch(basename, '*.pbtxt'):
                    self.lables = os.path.join(root, basename).replace("\\","/")

    def get_locations(self):
        return self.imgpath, self.annotspath, self.config, self.trec, self.testrec, self.results, self.lables, self.modelname, self.modelpath, self.recordpath, str(self.width), str(self.height)
#testing
'''
pp = os.path.join(os.path.expanduser('~'), 'Documents', 'testing')
find = finder(pp)
print(find.get_locations())
'''
    
