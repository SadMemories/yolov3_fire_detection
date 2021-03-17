import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from models import Darknet
from utils.utils import *
from utils.datasets import *


def prep_image(img, input_dim):
    """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
    '''
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    # print(img_.shape)
    '''
    orig_im = img
    img = letterbox(img, new_shape=input_dim)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).cuda().float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orig_im


def Process(output, ori_img, img):

    for i, det in enumerate(output):  # detections for image i
        # gn = torch.tensor(ori_img.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ori_img.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % ('fire', conf)
                    plot_one_box(xyxy, ori_img, label=label, color=[122, 98, 43])


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1042, 921)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 显示摄像头画面
        self.cam_frame = QtWidgets.QFrame(self.centralwidget)
        self.cam_frame.setGeometry(QtCore.QRect(10, 110, 521, 571))
        self.cam_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cam_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cam_frame.setObjectName("cam_frame")

        self.label_img_show = QtWidgets.QLabel(self.cam_frame)
        self.label_img_show.setGeometry(QtCore.QRect(10, 10, 501, 551))
        self.label_img_show.setObjectName("label_img_show")
        # self.label_img_show.setStyleSheet(("border:2px solid red"))

        # 显示检测画面
        self.detect_frame = QtWidgets.QFrame(self.centralwidget)
        self.detect_frame.setGeometry(QtCore.QRect(540, 110, 491, 571))
        self.detect_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detect_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_frame.setObjectName("detect_frame")
        self.label_detect_show = QtWidgets.QLabel(self.detect_frame)
        self.label_detect_show.setGeometry(QtCore.QRect(10, 10, 481, 551))
        self.label_detect_show.setObjectName("label_detect_show")
        # self.label_detect_show.setStyleSheet(("border:2px solid green"))
        # 按钮框架
        self.btn_frame = QtWidgets.QFrame(self.centralwidget)
        self.btn_frame.setGeometry(QtCore.QRect(10, 20, 1021, 80))
        self.btn_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.btn_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.btn_frame.setObjectName("frame_3")

        # 按钮水平布局
        self.widget = QtWidgets.QWidget(self.btn_frame)
        self.widget.setGeometry(QtCore.QRect(20, 10, 501, 60))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 20, 20, 20)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # 打开摄像头
        self.btn_opencam = QtWidgets.QPushButton(self.widget)
        self.btn_opencam.setObjectName("btn_opencam")
        self.horizontalLayout.addWidget(self.btn_opencam)
        # 加载模型文件
        self.btn_model_add_file = QtWidgets.QPushButton(self.widget)
        self.btn_model_add_file.setObjectName("btn_model_add_file")
        self.horizontalLayout.addWidget(self.btn_model_add_file)
        # 加载cfg文件
        self.btn_cfg_add_file = QtWidgets.QPushButton(self.widget)
        self.btn_cfg_add_file.setObjectName("btn_cfg_add_file")
        self.horizontalLayout.addWidget(self.btn_cfg_add_file)
        # 开始检测
        self.btn_detect = QtWidgets.QPushButton(self.widget)
        self.btn_detect.setObjectName("btn_detect")
        self.horizontalLayout.addWidget(self.btn_detect)
        # 退出
        self.btn_exit = QtWidgets.QPushButton(self.widget)
        self.btn_exit.setObjectName("btn_exit")
        self.horizontalLayout.addWidget(self.btn_exit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1042, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # 这里将按钮和定义的动作相连，通过click信号连接openfile槽？
        # 加载模型文件
        self.btn_model_add_file.clicked.connect(self.open_model)
        # 加载cfg文件
        self.btn_cfg_add_file.clicked.connect(self.open_cfg)
        # 打开摄像头
        self.btn_opencam.clicked.connect(self.opencam)
        # 开始识别
        self.btn_detect.clicked.connect(self.detect)
        # 这里是将btn_exit按钮和Form窗口相连，点击按钮发送关闭窗口命令
        self.btn_exit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测"))

        self.label_img_show.setText(_translate("MainWindow", "摄像头原始画面"))
        self.label_detect_show.setText(_translate("MainWindow", "实时检测效果"))
        self.btn_opencam.setText(_translate("MainWindow", "打开摄像头"))
        self.btn_model_add_file.setText(_translate("MainWindow", "加载模型文件"))
        self.btn_cfg_add_file.setText(_translate("MainWindow", "加载cfg文件"))
        self.btn_detect.setText(_translate("MainWindow", "开始检测"))
        self.btn_exit.setText(_translate("MainWindow", "退出"))

    def open_model(self):
        global openfile_name_mdoel
        openfile_name_mdoel, _ = QFileDialog.getOpenFileName(self.btn_model_add_file, '选择模型文件',
                                                             '/home/wwj/FireDetection/yolov3-archive-v2/Tmp_File/tmp_weight')
        print('加载模型文件地址为：' + str(openfile_name_mdoel))

    def open_cfg(self):
        global openfile_name_cfg
        openfile_name_cfg, _ = QFileDialog.getOpenFileName(self.btn_cfg_add_file, '选择cfg文件',
                                                           '/home/wwj/FireDetection/yolov3-archive-v2/Tmp_File/tmp_cfg')
        print('加载cfg文件地址为：' + str(openfile_name_cfg))

    def opencam(self):
        self.camcapture = cv2.VideoCapture(0)  # 参数0表示表示打开笔记本内置摄像头 参数是视频文件路径则打开视频文件
        self.timer = QtCore.QTimer()
        self.timer.start()
        self.timer.setInterval(3)  # 0.1s刷新一次
        self.timer.timeout.connect(self.camshow)

    def camshow(self):
        # global self.camimg
        ret, self.camimg = self.camcapture.read()  #.read()表示按帧读取视频 self.caming是每一帧图像 BGR
        if ret:
            camimg = cv2.cvtColor(self.camimg, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
            self.label_img_show.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            sys.exit(app.exec_())

    def detect(self):
        self.frames = 0
        self.start = time.time()
        cfgfile = openfile_name_cfg
        weightsfile = openfile_name_mdoel
        # self.num_classes = 1
        # args = arg_parse()
        self.confidence = 0.2
        self.nms_thesh = 0.5
        self.CUDA = torch.cuda.is_available()
        self.inp_dim = 512  # inp_dim表示图片的高度
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        self.model = Darknet(cfgfile, self.inp_dim)
        # attempt_download(weights)
        if weightsfile.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weightsfile, map_location=self.device)['model'])
        # self.model.load_weights(weightsfile)
        # self.model.net_info["height"] = 512  # self.model.net_info["height"]表示图像的分辨率
        self.timerdec = QtCore.QTimer()
        self.timerdec.start()
        self.timerdec.setInterval(3)  # 0.1s刷新一次
        self.timerdec.timeout.connect(self.object_detection)

    def object_detection(self):
        img, orig_im = prep_image(self.camimg,
                                  self.inp_dim)  # 返回的img是调整为inp_dim之后的图片 orig_im表示原始的图片 dim表示原始图片的height和width
        if self.CUDA:
            self.model.cuda()
            img = img.cuda()
        self.model.eval()
        output = self.model(img)[0]  # 传入模型
        # print(output.shape)
        output = non_max_suppression(output, self.confidence, self.nms_thesh,
                                   multi_label=False, classes=None, agnostic=False)
        # print(output)
        Process(output, orig_im, img)
        '''
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
        output[:, [1, 3]] *= self.camimg.shape[1]
        output[:, [2, 4]] *= self.camimg.shape[0]
        '''
        # list(map(lambda x: write(x, orig_im), output))
        self.frames += 1
        print("FPS of the video is {:5.2f}".format(self.frames / (time.time() - self.start)))
        camimg = cv2.cvtColor(self.camimg, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
        self.label_detect_show.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())