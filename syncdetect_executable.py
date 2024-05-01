from PyQt5 import QtWidgets, uic, QtCore,QtGui
from PyQt5.QtWidgets import QLineEdit, QTextBrowser
from PyQt5.QtGui import QRegExpValidator,QPixmap, QCursor
from PyQt5.QtCore import QRegExp, pyqtSignal
import tempfile, os, pathlib

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import shutil,threading
import qdarktheme

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import PIL,PIL.Image
import darkdetect

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class IV_window(QtWidgets.QWidget):                           # <===
    def __init__(self):
        super().__init__()
        Form, Base = uic.loadUiType(resource_path("image_viewer1.ui"))
        self.ui = Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Sync Error Viewer")
        self.setWindowIcon(QtGui.QIcon(resource_path("icon.svg")))
        self.ui.pushButton.clicked.connect(self.browsefiles)
        
    def browsefiles(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Single File',QtCore.QDir.currentPath() , 'Image (*.png *.jpg *jpeg)')
        self.ui.label.clear()
        self.ui.label.setPixmap(QPixmap(fileName))

class MainWindow(QtWidgets.QWidget):
    signal_left = pyqtSignal(str) #сигналы для оповещения функций перематывапния изображений
    signal_right = pyqtSignal(str)
    global tmp
    tmp = None
    global coupling
    coupling = []
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Form, Base = uic.loadUiType(resource_path("interface.ui"))
        self.ui = Form()
        self.ui.setupUi(self)
        #only works after loading ui !!!!
        self.setWindowTitle('SyncDetect')
        self.setWindowIcon(QtGui.QIcon(resource_path("icon.svg")))
        self.ui.stackedWidget.setCurrentWidget(self.ui.page)
        self.ui.amogusbutton.clicked.connect(self.window2)
        self.ui.amogusbutton2.clicked.connect(self.window3)
        self.ui.amogusbutton3.clicked.connect(self.window4)
        
        
    def closeEvent(self, event):#rm temp on window exit
        if tmp is not None:
            shutil.rmtree(tmp)
            self.close()
    
    #window2 = roessler
    #window3 = lorenz
    #window4 = chen
            
    def set_window(self, index, equation_image_name, dark_mode_image, input_fields, calc_func, IV_func):
        self.ui.stackedWidget.setCurrentIndex(index)
        getattr(self.ui, f'backbutton_{index}').clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))

        equation_image = dark_mode_image if darkdetect.isDark() else equation_image_name
        getattr(self.ui, f'eq_{index}').setPixmap(QPixmap(resource_path(equation_image)))

        reg_ex = QRegExp("(^[0-9][.])?[0-9]{1,4}")
        input_validator = QRegExpValidator(reg_ex)
        for field in input_fields:
            field.setValidator(input_validator)

        disabled_widgets = [getattr(self.ui, f'button_imgleft_{index}'), getattr(self.ui, f'button_imgright_{index}'), getattr(self.ui, f'showfolder_{index}')]
        for widget in disabled_widgets:
            widget.setEnabled(False)

        getattr(self.ui, f'calcbtn_{index}').clicked.connect(calc_func)
        getattr(self.ui, f'calcbtn_{index}').clicked.connect(lambda: self.galbutt(index))
        getattr(self.ui, f'btn_{index}').clicked.connect(IV_func)
        
    def window2(self):
        self.set_window(1, 'roessler_black.svg', 'roessler_white.svg', [self.ui.coupparamstart, self.ui.coupparamend, self.ui.param_a, self.ui.param_p, self.ui.param_c, self.ui.param_wr, self.ui.param_wd], self.roessler_func, self.IV_func)

    def window3(self):
        self.set_window(2, 'lorenz_black.svg', 'lorenz_white.svg', [self.ui.coupparamstart_2, self.ui.coupparamend_2, self.ui.param_sigma, self.ui.param_beta, self.ui.param_r1, self.ui.param_r2], self.lorenz_func, self.IV_func)
        
    def window4(self):
        self.set_window(3, 'chen_black.svg', 'chen_white.svg', [self.ui.coupparamstart_3, self.ui.coupparamend_3, self.ui.param_a_3, self.ui.param_b_3, self.ui.param_c_3, self.ui.param_d_3, self.ui.param_e_3, self.ui.param_k1_3, self.ui.param_k2_3], self.chen_func, self.IV_func)
    
    def galbutt(self, index):
        self.setup_widgets(index, getattr(self.ui, f'button_imgleft_{index}'), getattr(self.ui, f'button_imgright_{index}'), getattr(self.ui, f'showfolder_{index}'), self.signal_left, self.signal_right)

    def setup_widgets(self, index, button_left, button_right, button_folder, signal_left, signal_right):
        buttons_and_widgets = [button_left, button_right, button_folder]
        for widget in buttons_and_widgets:
            widget.setEnabled(True)
            
        if index == 1:
            signal_left.connect(self.signalResponse_left)
            signal_right.connect(self.signalResponse_right)
        if index == 2:
            signal_left.connect(self.signalResponse_left2)
            signal_right.connect(self.signalResponse_right2)
        if index == 3:
            signal_left.connect(self.signalResponse_left3)
            signal_right.connect(self.signalResponse_right3)
        

        button_left.clicked.connect(lambda: self.button_clickedleft(index))
        button_right.clicked.connect(lambda: self.button_clickedright(index))
        
        button_folder.clicked.connect(self.folderopen)
        
        
        
        #параметры системы
        
    #для перематывания изображений
           
    def worker_left(self,index): #функция перематывания влево
                total_images = len(self.list_of_images)
                self.i = (self.i - 1) % total_images
                img = self.list_of_images[self.i]

                getattr(self.ui, f'label_pic_{index}').setPixmap(QPixmap('{}\\{}'.format(path_dir, img))) 
                getattr(self.ui, f'subtitle_{index}').setText('система при параметре связи {}'.format(coupling[self.i])) 
                
                self.signal_left.emit("done")
                
    def button_clickedleft(self,index): #нажатие кнопки влево
                getattr(self.ui, f'button_imgleft_{index}').disconnect()
                worker=threading.Thread(target=self.worker_left(index))
                worker.start()
    
    #idk how to pass parameter to a PyQt signal, so i've done a funny:
    def signalResponse_right(self,response): #ответный сигнал, можно продолжать [справа]
         if response=="done":
            self.ui.button_imgright_1.clicked.connect(lambda: self.button_clickedright(1))
            
    def signalResponse_left(self,response): #ответный сигнал, можно продолжать
         if response=="done":
            self.ui.button_imgleft_1.clicked.connect(lambda: self.button_clickedleft(1))
            
    def signalResponse_right2(self,response): 
         if response=="done":
            self.ui.button_imgright_2.clicked.connect(lambda: self.button_clickedright(2))
            
    def signalResponse_left2(self,response): 
         if response=="done":
            self.ui.button_imgleft_2.clicked.connect(lambda: self.button_clickedleft(2))
            
    def signalResponse_right3(self,response): 
         if response=="done":
            self.ui.button_imgright_3.clicked.connect(lambda: self.button_clickedright(3))
            
    def signalResponse_left3(self,response): 
         if response=="done":
            self.ui.button_imgleft_3.clicked.connect(lambda: self.button_clickedleft(3))
            
    def worker_right(self,index): #функция перематывания влево
                total_images = len(self.list_of_images)
                self.i = (self.i + 1) % total_images
                img = self.list_of_images[self.i]

                getattr(self.ui, f'label_pic_{index}').setPixmap(QPixmap('{}\\{}'.format(path_dir, img)))
                getattr(self.ui, f'subtitle_{index}').setText('система при параметре связи {}'.format(coupling[self.i])) 
                
                self.signal_right.emit("done")
                     
    def button_clickedright(self,index):#нажатие кнопки влево
                getattr(self.ui, f'button_imgright_{index}').disconnect()
                worker=threading.Thread(target=self.worker_right(index))
                worker.start()
                
     #loading imgs                 
    def imagesfunc(self,index):
        self.list_of_images = os.listdir(path_dir)
        
        if not snc:
            image_index = 0
            subtitle_text = 'cистема при параметре связи {}'.format(0.00)
        else:
            image_index = snc[0]
            subtitle_text = 'cистема при параметре связи {}'.format(coupling[snc[0]])
        
        input_img_raw_string = '{}\\{}'.format(path_dir, self.list_of_images[image_index])
        getattr(self.ui, f'label_pic_{index}').setPixmap(QPixmap(input_img_raw_string))
        getattr(self.ui, f'subtitle_{index}').setText(subtitle_text)
        while QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not snc:
            self.i = 0
        else:
            self.i = snc[0]
        
    def folderopen(self):#opening directory with all the generated images
        os.startfile(path_dir)
    
    def IV_func(self):
        self.w = IV_window()
        self.w.show()
        
    def neural(self,index):
        self.list_of_images = os.listdir(path_dir)
        model = keras.models.load_model(resource_path('sync_model.keras'))
        class_names = ['nosync', 'sync']
        global snc
        snc = [i for i in range(len(self.list_of_images)) if self.predict_image_sync(i, model, class_names)]
        
        if not snc:
            getattr(self.ui, f'textBrowserUJWFHUKJF_{index}').setText('заданного диапазона не достаточно для достижения системой синхронизации')
        else:
            getattr(self.ui, f'textBrowserUJWFHUKJF_{index}').setText('cистема достигает синхронизации при параметре связи {}'.format(coupling[snc[0]]))

    def predict_image_sync(self, index, model, class_names):
        img = tf.keras.utils.load_img(str(path_dir) + '\plot {:02d}.png'.format(index), target_size=(554, 413))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return class_names[np.argmax(score)] == 'sync'

       
    def roessler_param(self):
        params = {
            'coupparamstart': (0.0, 'sr'),
            'coupparamend': (0.13, 'se'),
            'param_a': (0.15, 'a'),
            'param_p': (0.2, 'p'),
            'param_c': (10.0, 'c'),
            'param_wr': (0.95, 'wr'),
            'param_wd': (0.99, 'wd')
        }
        
        result = {}
        for param_name, (default_value, result_key) in params.items():
            ui_value = getattr(self.ui, param_name).text()
            result[result_key] = float(ui_value) if ui_value else default_value
            
        return result['a'], result['p'], result['c'], result['wr'], result['wd'], result['sr'], result['se']
    
    def roessler_func(self):
        a, p, c, wr, wd, sr, se = self.roessler_param()
        coupling.clear()
        #print(a, '\n', p, '\n', c, '\n', wr, '\n', wd, '\n', sr, '\n', se)  # отладка
        QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        n = 0
        global tmp
        if tmp is not None:
            shutil.rmtree(tmp)
        tmp = tempfile.mkdtemp()
        #print('Имя временного каталога:', tmp)
        global path_dir
        path_dir = pathlib.Path(tmp)

        def rossler(H, t=0):
            return np.array([(-wd * H[1]) - H[2],  # xd 0
                            (wd * H[0]) + (a * H[1]),  # yd 1
                            p + (H[2] * (H[0] - c)),  # zd 2
                            ((-wr * H[4]) - H[5]) + e * (H[0] - H[3]),  # xr 3
                            (wr * H[3]) + (a * H[4]),  # yr 4
                            p + (H[5] * (H[3] - c)),  # zr 5
                            (((-wr * H[7]) - H[8]) + e * (H[0] - H[6])),  # xa 6
                            ((wr * H[6]) + (a * H[7])),  # ya 7
                            p + (H[8] * (H[6] - c))])  # za 8

        for e in np.arange(sr, se, 0.01):
            T = 12000
            T0 = 4800
            t = np.linspace(0, 500, T)
            coupling.append(e)
            H0 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.05, 1.05, 1.05]
            H, infodict = integrate.odeint(rossler, H0, t, full_output=True)

            plt.plot(H[T0:, 3], H[T0:, 6], c='purple')
            plt.axis("off")
            plt.savefig(os.path.join(str(path_dir), 'plot {:02d}.png'.format(n)), bbox_inches='tight', pad_inches=0)
            
            plt.clf()
            n += 1

        self.neural(1)
        self.imagesfunc(1)  
        
    def lorenz_param(self):
        params = {
            'coupparamstart_2': (0.0, 'sr'),
            'coupparamend_2': (13.0, 'se'),
            'param_sigma': (10, 'sigma'),
            'param_beta': (8/3, 'beta'),
            'param_r1': (40.0, 'r1'),
            'param_r2': (35.0, 'r2')
        }
        
        result = {}
        for param_name, (default_value, result_key) in params.items():
            ui_value = getattr(self.ui, param_name).text()
            result[result_key] = float(ui_value) if ui_value else default_value
            
        return result['sigma'], result['beta'], result['r1'], result['r2'], result['sr'], result['se']


    def lorenz_func(self):
            sigma, beta, r1, r2, sr, se = self.lorenz_param()
            coupling.clear()
            #print(sigma, '\n', beta, '\n', r1, '\n', r2, '\n', sr, '\n', se)  # отладка
            QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            n = 0
            global tmp
            if tmp is not None:
                shutil.rmtree(tmp)
            tmp = tempfile.mkdtemp()
            #print('Имя временного каталога:', tmp)
            global path_dir
            path_dir = pathlib.Path(tmp)

            def lorenz(H, t=0):
                return np.array([sigma*(H[1]-H[0]), #H[0]-X1
                        r1*H[0]-H[1]-H[0]*H[2], #H[1]-Y1
                        (-beta*H[2]+H[0]*H[1]), #H[2]-Z1
                            
                        sigma*(H[4]-H[3])+K*(H[0]-H[3]),#H[3]-X2
                        r2*H[3]-H[4]-H[3]*H[5],#H[4]-Y2
                        (-beta*H[5]+H[3]*H[4]),#H[5]-Z2
                        
                        sigma*(H[7]-H[6])+K*(H[0]-H[6]),#H[6]-X3
                        r2*H[6]-H[7]-H[6]*H[8],#H[7]-Y3
                        (-beta*H[8]+H[6]*H[7])])#H[8]-Z3

            for K in np.arange(sr+0.5, se+0.5): #very funny fix
                t = np.linspace(0, 400, 40000)
                coupling.append(K-0.5)
                H0 = [0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 1.05, 1.05, 1.05 ]
                H, infodict = integrate.odeint(lorenz, H0, t, full_output=True)

                plt.plot(H[3500:, 3], H[3500:, 6], c='purple')
                plt.axis("off")
                plt.savefig(os.path.join(str(path_dir), 'plot {:02d}.png'.format(n)), bbox_inches='tight', pad_inches=0)
                
                plt.clf()
                n += 1

            self.neural(2)
            self.imagesfunc(2)
    def chen_param(self):
        params = {
            'coupparamstart_3': (154.0, 'sr'),
            'coupparamend_3': (164.0, 'se'),
            'param_a_3': (35.0, 'a'),
            'param_b_3': (4.9, 'b'),
            'param_c_3': (25.0, 'c'),
            'param_d_3': (5.0, 'd'),
            'param_e_3': (35.0, 'e'),
            'param_k1_3': (110.0, 'k1'),
            'param_k2_3': (190.0, 'k2'),
        }
        
        result = {}
        for param_name, (default_value, result_key) in params.items():
            ui_value = getattr(self.ui, param_name).text()
            result[result_key] = float(ui_value) if ui_value else default_value
            
        return result['a'], result['b'], result['c'], result['d'], result['e'], result['k1'], result['k2'], result['sr'], result['se']
    
    
    def chen_func(self):
                a, b, c, d, e, k1, k2, sr, se = self.chen_param()
                coupling.clear()
                #print(a, '\n', b, '\n', c, '\n', d, '\n',  e, '\n',  k1, '\n',  k2, '\n', sr, '\n', se)  # отладка
                QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
                n = 0
                global tmp
                if tmp is not None:
                    shutil.rmtree(tmp)
                tmp = tempfile.mkdtemp()
                #print('Имя временного каталога:', tmp)
                global path_dir
                path_dir = pathlib.Path(tmp)

                def chen(H, t=0):
                    return np.array([a*(H[1]-H[0])+e*H[1]*H[2], #H[0]-X1
                        c*H[0]-d*H[0]*H[2]+H[1]+H[3], #H[1]-Y1
                        (H[0]*H[1]-b*H[2]), #H[2]-Z1
                        (-k1*H[1]),#H[3]-u1
                            
                        a*(H[5]-H[4])+e*(H[5]*H[6]),#H[4]-X2
                        c*H[4]-d*H[4]*H[6]+H[5]+H[7],#H[5]-Y2
                        (H[4]*H[5]-b*H[6]),#H[6]-Z2
                        (-k2*H[5]+K*(H[0]-H[4])),#H[7]-u2
                        
                        a*(H[9]-H[8])+e*(H[9]*H[10]),#H[8]-X3
                        c*H[8]-d*H[8]*H[10]+H[9]+H[11],#H[9]-Y2
                        (H[8]*H[9]-b*H[10]),#H[10]-Z3
                        (-k2*H[9]+K*(H[0]-H[8]))])#H[11]-u3

                for K in np.arange(sr, se,1): 
                    t = np.linspace(0, 500, 12000)
                    coupling.append(K)
                    H0 = [0.001, 0.001, 0.001,0.001, 0.002, 0.002, 0.002,0.002, 1.05, 1.05, 1.05,1.05]
                    H, infodict = integrate.odeint(chen, H0, t, full_output=True)

                    plt.plot(H[4800:, 5], H[4800:, 9], c='purple')
                    plt.axis("off")
                    plt.savefig(os.path.join(str(path_dir), 'plot {:02d}.png'.format(n)), bbox_inches='tight', pad_inches=0)
                    
                    plt.clf()
                    n += 1

                self.neural(3)
                self.imagesfunc(3)
    
                                    
    
        
        
        
        
       
if __name__ == "__main__":
    import sys
    qdarktheme.enable_hi_dpi()
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme("auto") #theme dependant on OS theme
    #app.setStyle('Fusion') #built-in style, default for Qt
    window = MainWindow()
    window.show()
    sys.exit(app.exec())