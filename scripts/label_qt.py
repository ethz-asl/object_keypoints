#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
from turtle import right
import random
from skvideo import io as video_io
import h5py
import numpy as np
from perception.utils import camera_utils
from perception.constants import *

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout


# TODO change with flags
PATH = "/media/giuseppe/My Passport/2021-12-10-11-35-45_valve_perception"
CALIBRATION = "/media/giuseppe/My Passport/calibration/intrinsics.yaml"

class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabelLeft = QLabel()
        self.imageLabelLeft.setBackgroundRole(QPalette.Base)
        self.imageLabelLeft.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelLeft.setScaledContents(True)
        self.imageLabelLeft.mousePressEvent = self.getPosLeft

        self.imageLeftNextButton = QPushButton('Next Left', self)
        self.imageLeftNextButton.setToolTip('Move to the next frame on the left')
        self.imageLeftNextButton.resize(400, 50)
        self.imageLeftNextButton.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLeftNextButton.clicked.connect(self._onClickLeft)

        self.imageLabelRight = QLabel()
        self.imageLabelRight.setBackgroundRole(QPalette.Base)
        self.imageLabelRight.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelRight.setScaledContents(True)
        self.imageLabelRight.mousePressEvent = self.getPosRight

        self.imageRightNextButton = QPushButton('Next Right', self)
        self.imageRightNextButton.setToolTip('Move to the next frame on the right')
        self.imageRightNextButton.resize(400, 50)
        self.imageRightNextButton.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageRightNextButton.clicked.connect(self._onClickRight)
        
        self.hlayout = QHBoxLayout()
        self.hlayout.addWidget(self.imageLabelLeft)
        self.hlayout.addWidget(self.imageLabelRight)

        self.hblayout = QHBoxLayout()
        self.hblayout.addWidget(self.imageLeftNextButton)
        self.hblayout.addWidget(self.imageRightNextButton)

        self.vlayout = QVBoxLayout()
        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addLayout(self.hblayout)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setLayout(self.vlayout)
        #self.scrollArea.setWidget(self.imageLabel1)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()
        
        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

        self.flags = None
        self.current_dir = None
        self.video = None
        self.commands = []
        self.left_keypoints = []
        self.right_keypoints = []
        self.world_points = []
        self.alpha = 1.0
        self.hdf = None
        self.done = False
        self.camera = None

    def _load_camera_params(self):
        self.camera = camera_utils.from_calibration(CALIBRATION) #self.flags.calibration TODO(giuseppe) restore from flag

    def _onClickLeft(self):
        self.left_frame_index = random.randint(0, self.hdf['camera_transform'].shape[0]-1)
        left_frame = self.video[self.left_frame_index]
        self.imageLabelLeft.setPixmap(QPixmap.fromImage(self._np2qt_image(left_frame)))

    def _onClickRight(self):
        self.right_frame_index = random.randint(0, self.hdf['camera_transform'].shape[0]-1)
        right_frame = self.video[self.right_frame_index]
        self.rightLabelLeft.setPixmap(QPixmap.fromImage(self._np2qt_image(right_frame)))

    def _find_furthest(self):
        video_length = self.hdf['camera_transform'].shape[0]
        smallest_index = (None, None)
        value = 1.0
        stride = video_length // 30
        for i in range(0, video_length, stride):
            for j in range(i, video_length, stride):
                T_WL = self.hdf['camera_transform'][i]
                T_WR = self.hdf['camera_transform'][j]
                if np.linalg.norm(T_WL[:3, 3] - T_WR[:3, 3]) < 0.1:
                    # Skip if the viewpoints are too close to each other.
                    continue
                # Points are 1 meter along the z-axis from the camera position.
                z_L = T_WL[2, :3]
                z_R = T_WR[2, :3]

                dot = np.abs(z_L.dot(z_R))
                if dot < value:
                    value = dot
                    smallest_index = (i, j)
        print("Furthest frames: ", *smallest_index)
        return smallest_index

    def getPosRight(self , event):
        # TODO resize point according to the current resizing of the image
        # TODO save point for triangulation
        x = event.pos().x()
        y = event.pos().y()

        curr_w = self.imageLabelLeft.size().width()
        curr_h = self.imageLabelLeft.size().height()
        
        right_frame = self.video[self.left_frame_index]
        scale_x = left_frame.shape[1] / curr_w
        scale_y = left_frame.shape[0] / curr_h
        
        x = x * scale_x
        y = y * scale_y
        right_frame = cv2.circle(right_frame, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=2)
        
        self.imageLabelRight.setPixmap(QPixmap.fromImage(self._np2qt_image(right_frame)))
        
        print(f"[image 1] clicked at ({x}, {y})") 
    
    def getPosLeft(self , event):
        # TOD0 resize point according to the current resizing of the image
        # TODO save point for triangulation
        x = event.pos().x()
        y = event.pos().y()

        curr_w = self.imageLabelLeft.size().width()
        curr_h = self.imageLabelLeft.size().height()
        
        left_frame = self.video[self.left_frame_index]
        scale_x = left_frame.shape[1] / curr_w
        scale_y = left_frame.shape[0] / curr_h
        
        x = x * scale_x
        y = y * scale_y
        left_frame = cv2.circle(left_frame, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=2)
        self.imageLabelLeft.setPixmap(QPixmap.fromImage(self._np2qt_image(left_frame)))

        print(f"[image 2] clicked at ({x}, {y})") 
    
    @staticmethod
    def _np2qt_image(img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    
    def set_current(self, path):
        self.done = False
        self.left_keypoints = []
        self.right_keypoints = []
        self.world_points = []
        
        if self.hdf is not None:
            self.hdf.close()
        self.hdf = h5py.File(os.path.join(path, 'data.hdf5'), 'r')
        self._load_camera_params()

        self.current_dir = path
        self.left_frame_index, self.right_frame_index = self._find_furthest()

        self.video = video_io.vread(os.path.join(path, 'frames_preview.mp4'))
        print(self.hdf['camera_transform'].shape[0], "poses")
        left_frame = self.video[self.left_frame_index]
        right_frame = self.video[self.right_frame_index]

        self.imageLabelLeft.setPixmap(QPixmap.fromImage(self._np2qt_image(left_frame)))
        self.imageLabelRight.setPixmap(QPixmap.fromImage(self._np2qt_image(right_frame)))
        self.scaleFactor = 1.0

        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        # TODO Need to add this back
        # keypoint_path = os.path.join(path, KEYPOINT_FILENAME)
        # if os.path.exists(keypoint_path):
        #     self._load_points(keypoint_path)
    
    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.imageLabelLeft.setPixmap(QPixmap.fromImage(image))
            self.imageLabelRight.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabelLeft.adjustSize()
                self.imageLabelRight.adjustSize()
                

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel1.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel1.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel1.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabelLeft.adjustSize()
        self.imageLabelRight.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel1.resize(self.scaleFactor * self.imageLabel1.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.set_current(PATH)
    imageViewer.show()
    sys.exit(app.exec_())
    # TODO QScrollArea support mouse
    # base on https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
    #
    # if you need Two Image Synchronous Scrolling in the window by PyQt5 and Python 3
    # please visit https://gist.github.com/acbetter/e7d0c600fdc0865f4b0ee05a17b858f2