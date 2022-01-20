#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Created by Giuseppe Rizzi
# (c) Julian Keller, Giuseppe Rizzi, 2022
# Parts on the code rely on the StereoLabeler code by Kenneth Blomqvist

import os
import argparse
import json
import cv2
import random
from skvideo import io as video_io
import h5py
from dataclasses import dataclass
import numpy as np
from perception.utils import camera_utils, linalg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QAbstractScrollArea


@dataclass
class Keypoint:
    x: float
    y: float
    frameId: int
    worldPoint: any


class QCustomImage(QLabel):
    def __init__(self, scrollArea, button):
        QLabel.__init__(self)
        self.setBackgroundRole(QPalette.Base)
        self.setScaledContents(True)
        self.scrollArea = scrollArea
        self.button = button
        self._frameId = None
        self._video = None

    def setVideo(self, video):
        self._video = video

    def setFrameId(self, id):
        assert self._video is not None
        self._frameId = id
        print(f'Frame id {id}')
        self.updateImage()

    def getFrameId(self):
        return self._frameId

    def getFrame(self):
        return self._video[self._frameId]

    def updateImage(self):
        self.setPixmap(QPixmap.fromImage(self._np2qt_image(self._video[self._frameId])))

    @staticmethod
    def _np2qt_image(img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

class QImageViewer(QMainWindow):
    def __init__(self, flags):
        super().__init__()

        self._keypoints: Keypoint = []

        self.printer = QPrinter()

        self.hlayout = QHBoxLayout()
        self.hblayout = QHBoxLayout()

        def createSingleView(id):
            class QCustomScrollArea(QScrollArea):
                def wheelEvent(self2, event):
                    if event.modifiers() == Qt.ControlModifier:
                        event.accept()
                        if event.angleDelta().y() > 0:
                            self.zoomIn()
                        else:
                            self.zoomOut()
                        return
                    QScrollArea.wheelEvent(self2, event)

            scrollArea = QCustomScrollArea()
            scrollArea.horizontalScrollBar().valueChanged.connect(lambda value: self._syncScrollbars(scrollArea.horizontalScrollBar(), False, value))
            scrollArea.verticalScrollBar().valueChanged.connect(lambda value: self._syncScrollbars(scrollArea.verticalScrollBar(), True, value))
            scrollArea.setBackgroundRole(QPalette.Dark)

            button = QPushButton('Random Image', self)
            button.setToolTip('Move to a random frame')
            button.clicked.connect(lambda: self.nextImage(image))
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            image = QCustomImage(scrollArea=scrollArea, button=button)
            image.mousePressEvent = lambda event: self._onImageClick(image, id, event)

            scrollArea.setWidget(image)
            self.hlayout.addWidget(scrollArea)
            self.hblayout.addWidget(button)

            return image

        self.imageLeft = createSingleView(0)
        self.imageRight = createSingleView(1)
        self.images = [self.imageLeft, self.imageRight]

        self.mainWidget = QWidget()
        self.vlayout = QVBoxLayout(self.mainWidget)
        self.vlayout.addLayout(self.hlayout, 4)
        self.vlayout.addWidget(QLabel('Use CTRL+Scroll to zoom, Scroll to go vertical and ALT+Scroll to go horizontal.\n'
                                      'Alternatively use +, - and the arrow keys on your keyboard.\n'
                                      'Click on a keypoint on the left, then on the corresponding one on the right.\n'
                                      'If a keypoint from one image is not available in the other, just use another image.\n'
                                      'You can freely switch to other images while assigning keypoints.\n'
                                      'In the end, you can check the red backprojections by browsing through the images.'))
        self.vlayout.addLayout(self.hblayout, 1)

        self.setCentralWidget(self.mainWidget)

        self._createActions()
        self._createMenus()
        
        self.setWindowTitle("Image Labeler Plus")
        self.resize(800, 600)

        self.commands = []
        self.worldPoints = []
        self.alpha = 1.0
        self.hdf = h5py.File(os.path.join(flags.base_dir, 'data.hdf5'), 'r')
        self.out_file = os.path.join(flags.base_dir, 'keypoints.json')
        self.done = False
        self.camera = camera_utils.from_calibration(flags.calibration)
        print(self.hdf['camera_transform'].shape[0], "poses")

        self._video = video_io.vread(os.path.join(flags.base_dir, 'frames_preview.mp4'))
        for image in self.images:
            image.setVideo(self._video)

        for image, frameId in zip(self.images, self._find_furthest()):
            image.setFrameId(frameId)

        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self._updateActions()

        self._load()

    def showEvent(self, event):
        self.zoomImages()
        QMainWindow.showEvent(self, event)

    def resizeEvent(self, event):
        if self.fitToWindowAct.isChecked():
            self.zoomImages()
        QMainWindow.resizeEvent(self, event)

    def nextImage(self, image, rnd=False):
        frameId = random.randint(0, self.hdf['camera_transform'].shape[0]-1) if rnd else \
            (image.getFrameId() + 1) % self.hdf['camera_transform'].shape[0]
        image.setFrameId(frameId)

    def updateImages(self):
        for image in self.images:
            image.updateImage()

    def _onImageClick(self, image, id, event):
        if len(self._keypoints) % len(self.images) != id:
            QMessageBox(QMessageBox.Information, 'Point order', 'Please click on the other image first').exec_()
            return

        x = event.pos().x()
        y = event.pos().y()

        curr_w = image.size().width()
        curr_h = image.size().height()

        frame = image.getFrame()
        scale_x = frame.shape[1] / curr_w
        scale_y = frame.shape[0] / curr_h

        x = x * scale_x
        y = y * scale_y

        self._keypoints.append(Keypoint(x, y, image.getFrameId(), None))

        cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=2)

        self.compute_all()

        self.updateImages()

        print(f"[image] clicked at ({x}, {y})")

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

    def _backproject(self, worldPoint, index):
        for frameId, frame in enumerate(self._video):
            T_WL = self.hdf['camera_transform'][frameId]
            T_LW = linalg.inv_transform(T_WL)
            worldPoint = (worldPoint / worldPoint[3])
            projected_x = self.camera.project(worldPoint[:3].T, T_LW)
            projected_x = projected_x.ravel()

            cv2.circle(frame, (int(projected_x[0]), int(projected_x[1])), radius=2, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, str(index), (int(projected_x[0]), int(projected_x[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0))

    def compute_all(self):
        if len(self._keypoints) % len(self.images) != 0:
            return

        print('Triangulating')
        for pointLeft, pointRight in zip(self._keypoints[::len(self.images)],
                                         self._keypoints[1::len(self.images)]):
            # Only compute points which have not been computed yet
            # Otherwise, points might get added twice to worldPoints
            # Or, loaded points from the file would be overwritten
            if pointLeft.worldPoint is not None:
                continue
            worldPoint = self._triangulate(pointLeft, pointRight)
            pointLeft.worldPoint = worldPoint
            pointRight.worldPoint = worldPoint
            self._backproject(worldPoint, len(self.worldPoints))
            self.worldPoints.append(worldPoint)

        self._save()

    def _load(self):
        if not os.path.exists(self.out_file):
            return

        with open(self.out_file, 'rt') as f:
            points = json.loads(f.read())
            for x in points['3d_points']:
                worldPoint = np.array(x).reshape(4, 1)
                self._backproject(worldPoint, len(self.worldPoints))
                self.worldPoints.append(worldPoint)

        self.updateImages()

    def _save(self):
        """Writes keypoints to file as json. """
        contents = {
            '3d_points': [x.ravel().tolist() for x in self.worldPoints] # Triangulated 3D points in world frame.
        } # Points are ordered and correspond to each other.
        with open(self.out_file, 'w') as f:
            f.write(json.dumps(contents))

    def _triangulate(self, left_point: Keypoint, right_point: Keypoint):
        T_WL = self.hdf['camera_transform'][left_point.frameId]
        T_WR = self.hdf['camera_transform'][right_point.frameId]
        T_RW = linalg.inv_transform(T_WR)
        T_RL = T_RW @ T_WL

        x = np.array([left_point.x, left_point.y])[:, None]
        xp = np.array([right_point.x, right_point.y])[:, None]

        P1 = camera_utils.projection_matrix(self.camera.K, np.eye(4))
        P2 = self.camera.K @ np.eye(3, 4) @ T_RL

        x = self.camera.undistort(x.T).T
        xp = self.camera.undistort(xp.T).T

        p_LK = cv2.triangulatePoints(P1, P2, x, xp)
        p_LK = p_LK / p_LK[3]
        p_WK = T_WL @ p_LK
        return p_WK

    def print(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel1.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel1.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel1.pixmap())

    def _syncScrollbars(self, sender, vertical, value):
        for image in self.images:
            if vertical:
                bar = image.scrollArea.verticalScrollBar()
            else:
                bar = image.scrollArea.horizontalScrollBar()
            if bar != sender:
                bar.setValue(value)

    def zoomImages(self, factor=0):
        if factor == 0 or self.fitToWindowAct.isChecked():
            for image in self.images:
                w = image.scrollArea.viewport().size().width()
                h = image.getFrame().shape[0] / image.getFrame().shape[1] * w
                image.resize(w, h)
            return

        newSize = factor * self.images[0].size()
        for image in self.images:
            image.resize(newSize)

    def zoomIn(self):
        self.zoomImages(1.25)

    def zoomOut(self):
        self.zoomImages(0.8)

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        if fitToWindow:
            self.zoomImages()
        self._updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Labeler Plus",
                          "<p>The <b>Image Labeler Plus</b> can be used to label keypoints "
                          "on a set of images. The corresponding 2D points are triangulated "
                          "and the resulting 3D points are then stored to a file.")

    def _createActions(self):
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print)
        self.exitAct = QAction("E&xit", self, shortcut="esc", triggered=self.close)
        self.nextLeft = QAction("Next &Left", self, shortcut="q", triggered=lambda: self.nextImage(self.imageLeft, False))
        self.nextRight = QAction("Next &Right", self, shortcut="w", triggered=lambda: self.nextImage(self.imageRight, False))
        self.randomLeft = QAction("Random &Left", self, shortcut="a", triggered=lambda: self.nextImage(self.imageLeft, True))
        self.randomRight = QAction("Random &Right", self, shortcut="s", triggered=lambda: self.nextImage(self.imageRight, True))
        self.zoomInAct = QAction("Zoom &In", self, shortcut="+", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out", self, shortcut="-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="0", enabled=False, triggered=self.zoomImages)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="f",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def _createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.navigationMenu = QMenu("&Navigation", self)
        self.navigationMenu.addAction(self.nextLeft)
        self.navigationMenu.addAction(self.nextRight)
        self.navigationMenu.addSeparator()
        self.navigationMenu.addAction(self.randomLeft)
        self.navigationMenu.addAction(self.randomRight)

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
        self.menuBar().addMenu(self.navigationMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def _updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())


if __name__ == '__main__':
    import sys
    import signal
    from PyQt5.QtWidgets import QApplication

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    parser.add_argument('--calibration', type=str, default='config/calibration.yaml',
                        help="Path to calibration file.")
    flags = parser.parse_args()

    app = QApplication(sys.argv)
    imageViewer = QImageViewer(flags)
    imageViewer.show()
    sys.exit(app.exec_())
