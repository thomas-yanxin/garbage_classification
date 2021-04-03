# -*- coding: utf-8 -*-
import datetime
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import paddlehub as hub
import requests
import urllib3
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

tmp = None

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self,parent=None):
        super().__init__(parent) #父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.cap = cv2.VideoCapture()       #视频流
        self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头
        
        self.set_ui()                       #初始化程序界面
        self.slot_init()                    #初始化槽函数
        self.message = " No face_recognition"


    '''程序界面布局'''
    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()    #总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()    #按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()    #数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开摄像头')   #建立用于人脸识别的按键
        self.button_close = QtWidgets.QPushButton('退出')   #建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)    #设置按键大小
        # self.button_recognition .setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        self.setWindowTitle('慧眼识垃圾')
        self.move(200,200)
        self.button_close.move(10,100)                      #移动按键
        self.setWindowIcon(QIcon('/home/thomas/python/The-Eye-Konws-the-Garbage/picture/garbage_icon.png'))
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()   #定义显示视频的Label
        self.label_show_camera.setFixedSize(720,540)    #给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera) #把打开摄像头的按键放到按键布局中

        self.__layout_fun_button.addWidget(self.button_close)       #把退出程序的按键放到按键布局中
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)      #把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)        #把用于显示视频的Label加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main) #到这步才会显示所有控件

    def lable_close(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.label_show_camera.clear()


    def garbage(self, value, percentage):
        '''显示对话框返回值'''
        QMessageBox.information(self, "结果",   "{},置信度为：{}".format(value, percentage), QMessageBox.Yes | QMessageBox.No)
        self.lable_close()
        self.message = " No face_recognition"
        self.button_open_camera.setText('打开摄像头')

    '''初始化所有槽函数'''
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click) #调用face_recognition()
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)#若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText('垃圾识别')
        else:
            self.button_open_camera.setText('打开摄像头')
            self.garbage_recognition()

    def show_camera(self):
        flag, self.image = self.cap.read()
        image = cv2.flip(self.image, 1) # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        show = cv2.resize(self.image, (740, 540))     #把读到的帧的大小重新设置为 640x480
  
        img = cv2.cvtColor(show,cv2.COLOR_BGR2RGB)   #视频色彩转换回RGB，这样才是现实的颜色    
        img = cv2.flip(img,1)                       #视频镜像翻转
        showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_show_camera.setScaledContents(True)

                
    def garbage_recognition(self):

        top_k = 1

        module = hub.Module(name="garbage_classification")
        filepath='/home/thomas/python/The-Eye-Konws-the-Garbage/garbage_classification.json'
        self.picture_file = '/home/thomas/python/The-Eye-Konws-the-Garbage/picture/img.jpg'
        cv2.imwrite(self.picture_file, self.image)
        picture_file = [self.picture_file,]
        res = module.predict(paths=picture_file, top_k=top_k)

        for i, image in enumerate(picture_file):
            print("The returned result of {}: {}".format(image, res[i]))
            category_id = res[i][0][0]
            score = res[i][1][0]
            score = "%.2f%%"%(score*100)
            time = "%.2f"%res[i][2]
            f_obj=open(filepath)
            print(category_id)
            recognition=json.load(f_obj)[str(category_id)]
            # print("The returned result is {}, and the score is {}, the inference time is {}s".format(content,score,time))
            # return "The returned result is {}, and the score is {}, the inference time is {}s".format(content,score,time)
        self.garbage(recognition, score)


if __name__ =='__main__':
    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用
    ui = Ui_MainWindow()                    #实例化Ui_MainWindow
    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    # app.exec_()
    sys.exit(app.exec_())

