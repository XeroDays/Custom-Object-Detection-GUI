'''
Class used to setup the GUI workspace and install
tensorflow and protoc. Functions add protoc to the 
system path and install tensorflow using protoc.
NOTE: Admin mode MUST be used when running this function
If not the path will not be appended with protoc and this 
will not run. 
'''


import os
import logging
import shutil
import winreg as registry
import subprocess


logger = logging.getLogger('Main')
fh = logging.FileHandler('logGUI.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

class path_installer():
    def __init__(self, path):
        self.path = path
        self.complete = False


    def start(self):

        self.complete = False

        file  = open("Setup.txt", 'r')
        if len(file.readlines() <= 0):
            return

        full_path = os.path.join(self.path, "exe.win-amd64-3.8", 'protoc', 'bin;').replace("/",'\\')
        tensorpath = os.path.join(self.path, "tensorflow", "models", "research").replace("\\", "/")

        #gets all current enviroment variables
        env_list = os.environ["Path"]

        env_variables = env_list.split(";")
        i=0
        found = False

        # While loop used to remove enviroment varibles that only exist 
        # in the virtual enviroment and so should not be added to system path
        while i < len(env_variables):
            path = env_variables[i].split('\\')
            for j in path:
                if j == 'PyQt5':
                    del env_variables[i]
                    found  = True
                    break
                elif j == 'cv2':
                    del env_variables[i]
                    found = True
                    break
                elif j == 'mpl_toolkits.libs':
                    del env_variables[i]
                    found = True
                    break
                elif j == 'matplotlib.libs':
                    del env_variables[i]
                    found = True
                    break
                elif j == 'scipy.libs':
                    del env_variables[i]
                    found = True
                    break
            if found:
                found = False
                continue
            i = i + 1

        final_path = full_path + '; '.join(env_variables)

        if len(file.readlines()) == 0:

            logger.info("Adding tensorflow and protoc installs")
    
            try:
                self.update_reg_path_value(final_path)
            except Exception as e:
                logger.error(f"Error adding to path {e}")
                print(f"error {e}")
    
            '''
            try:
                os.system("cd /tensorflow/models/research")
            except Exception as e:
                logger.error(e)

            try:
                os.system("protoc object_detection/protos/*.proto --python_out=.")
            except Exception as e:
                logger.error(e)
            '''
            command  = f'cd {str(tensorpath)} && protoc object_detection/protos/*.proto --python_out=. && python -m pip install object_detection/packages/tf2/setup.py'
            try:
        #os.system("cp object_detection/packages/tf2/setup.py .")
                install = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                logger.debug(install.stdout)
            except Exception as e:
                logger.error(e)
            
            filewrite  = open("Setup.txt", 'w')
            filewrite.write("done")
            

            self.complete = True

    def update_reg_path_value(self, path):

        logger.info("Adding Protoc to Path")
        print("protoc added to path")

        # HKEY_LOCAL_MACHINE used to add to the system path NOT the user path
        SYS_ENV_SUBPATH = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
        env_key = registry.OpenKey(registry.HKEY_LOCAL_MACHINE,SYS_ENV_SUBPATH,
                            0,registry.KEY_ALL_ACCESS)
        registry.SetValueEx(env_key,"Path",0,registry.REG_EXPAND_SZ,path)


    def completed(self):
        return self.complete


