"""
A simple setup script to create an executable using PyQt5. This also
demonstrates the method for creating a Windows executable that does not have
an associated console.

test_pyqt5.py is a very simple type of PyQt5 application

Run the build process by running the command 'python setup.py build'

If everything works well you should find a subdirectory in the build
subdirectory that contains the files needed to run the application
"""

from __future__ import annotations

import sys

from cx_Freeze import Executable, setup

try:
    from cx_Freeze.hooks import get_qt_plugins_paths
except ImportError:
    get_qt_plugins_paths = None


include_files = ['icon.ico', 'Object_Detection_Help_Guide.pdf', 'ObjDet.ui','shuffle.txt', 
                 'help_viewer.ui','eval_dialog.ui', 'createproj.ui','C:/Users/sseaberry/Documents/tensorflow/', 
                 'C:/Users/sseaberry/Documents/protoc/', "Setup.txt", "install.ui","C:/Users/sseaberry/Documents/Gui2/nvvm/",
                  'About.ui' ]
if get_qt_plugins_paths:
    # Inclusion of extra plugins (since cx_Freeze 6.8b2)
    # cx_Freeze imports automatically the following plugins depending of the
    # use of some modules:
    # imageformats, platforms, platformthemes, styles - QtGui
    # mediaservice - QtMultimedia
    # printsupport - QtPrintSupport
    for plugin_name in (
        # "accessible",
        # "iconengines",
        # "platforminputcontexts",
        # "xcbglintegrations",
        # "egldeviceintegrations",
        "wayland-decoration-client",
        "wayland-graphics-integration-client",
        # "wayland-graphics-integration-server",
        "wayland-shell-integration",
    ):
        include_files += get_qt_plugins_paths("PyQt5", plugin_name)

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None
packages = ["PyQt5", "cv2", "os", "sys", "matplotlib.figure", "matplotlib.backends.backend_qt5agg", 
                 "numpy", "PyQt5.QtGui", "PyQt5.QtCore", "PyQt5.QtWidgets", "matplotlib", "pandas", 
                 "sklearn.model_selection", "tensorflow", "queue", "threading", "subprocess", "signal",
                 "time", "io", "tf_slim", "shutil", "PIL", "random", "logging", "object_detection",
                 "csv", "fnmatch", "winreg", "absl", "traceback", "tensorflow_io", 'avro',
                 'apache_beam','lxml', 'Cython', 'contextlib2','six','pycocotools', 'lvis',
                 'scipy','pandas','official', 'orbit', 'tensorflow_models','keras','pyparsing', 
                 'sacrebleu']
build_exe_options = {
    # exclude packages that are not really needed
    "include_files": include_files,
    "zip_include_packages": ["PyQt5"],
    "includes": ["PyQt5", "cv2", "os", "sys", "matplotlib.figure", "matplotlib.backends.backend_qt5agg", 
                 "numpy", "PyQt5.QtGui", "PyQt5.QtCore", "PyQt5.QtWidgets", "matplotlib", "pandas", 
                 "sklearn.model_selection", "tensorflow", "queue", "threading", "subprocess", "signal",
                 "time", "io", "tf_slim", "shutil", "PIL", "random", "logging", "object_detection",
                 "csv", "fnmatch", "winreg", "absl", "traceback", "tensorflow_io",'avro',
                 'apache_beam','lxml', 'Cython', 'contextlib2','six','pycocotools', 'lvis',
                 'scipy','pandas','official', 'orbit', 'tensorflow_models','keras','pyparsing', 
                 'sacrebleu'],
    "packages": packages,
    "include_msvcr": True,
}



bdist_mac_options = {
    "bundle_name": "Test",
}

bdist_dmg_options = {
    "volume_label": "TEST",
}

bdist_msi_options = {
    "add_to_path" : True

}

executables = [Executable("MainGUI.py", base=base, icon="icon.ico"), 
               Executable("C:/Users/sseaberry/Documents/tensorflow/models-master/research/object_detection/model_main_tf2.py", base=base, icon="tensorflow_icon.ico") ]


setup(
    name="simple_PyQt5",
    version="0.4",
    description="Sample cx_Freeze PyQt5 script",
    options={
        "build_exe": build_exe_options,
        "bdist_mac": bdist_mac_options,
        "bdist_dmg": bdist_dmg_options,
        "bdist_msi": bdist_msi_options,
    },
    

    executables=executables,
    
)