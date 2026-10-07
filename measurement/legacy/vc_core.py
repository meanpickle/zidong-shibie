# -*- coding: utf-8 -*-

from ui_2023110801 import *

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtBluetooth import *
from PyQt5.QtNetwork import *

import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile

import zipfile
 
from collections import defaultdict
from io import StringIO
from PIL import Image
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#################################################
import argparse
import time
from pathlib import Path

import cv2
from numpy import random
from scipy.linalg import svd
import math
from datetime import datetime

import pyrealsense2 as rs
import math
from vc_realsense_state import RealsenseState
from vc_fun import check_line


#################################################


class VC_Core(QtWidgets.QMainWindow):
	def __init__(self, name="VC"):
		# 构造函数
		super().__init__()
		self.initUI(name)

		
		
		#init QTimer
		self.timer_camera = QtCore.QTimer()
		self.timer_run_car = QtCore.QTimer()
		
		
		#
		self.frame_width = 1280
		self.frame_height = 720
		#
		self.rect_alpha = 0.80  # 透明度，取值范围为0-1
		###
		self.vtx_list=[]
		############
		
		#####
		
		
		#
		self.initConnect()
		#初始化Realsense
		self.initRealsense()
		#

		self.timer_run_car.start(10)
		self.now_run_handle_time = datetime.now()
		self.last_run_handle_time = datetime.now()
		self.index_next_time = 6000
		#
		self.click_count=0
		self.click_points=[]
		self.grid_col=10
		self.grid_row=10
		self.grid_data=[]
		#
		self.vc_ui.lineEdit_face_smooth_cols.setText(str(self.grid_col))
		self.vc_ui.lineEdit_face_smooth_grid_rows.setText(str(self.grid_row))
		#
		self.check_count=0

		
		
		pass
	#
	def initUI(self, name):
		# 初始化函数
		self.vc_ui = Ui_MainWindow()
		self.vc_ui.setupUi(self)
		self.setWindowTitle(name)

		#self.vc_ui.horizontalSlider.setValue(25)
		#self.vc_ui.horizontalSlider.setMaximum(100)
		
	#初始化Realsense
	def initRealsense(self):
		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.point_cloud = rs.pointcloud()
		
		self.align_to = rs.stream.color  #与color流对齐
		self.align = rs.align(self.align_to)
		
		#self.config.enable_stream(rs.stream.depth,  self.frame_width, self.frame_height, rs.format.z16, 30)
		self.config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
		self.config.enable_stream(rs.stream.color,  self.frame_width, self.frame_height, rs.format.bgr8, 30)

		self.state = RealsenseState()
		pass
	#
	#
	#绑定槽函数
	def initConnect(self):
		
		#连接Realsense相机
		self.vc_ui.actionOpenRealsense.triggered.connect(self.slot_connect_realsense_action)
		
		
		#
		self.timer_camera.timeout.connect(self.show_camera_action)
		#
		# 连接自定义信号到槽函数
		self.vc_ui.label_img.double_click_signal.connect(self.on_double_clicked_action)
		#
		self.vc_ui.pushButton_draw_line.clicked.connect(self.slot_set_line_action)
		#计算长度
		self.vc_ui.pushButton_calculate_length.clicked.connect(self.slot_calculate_line_length_action)
		#清空线
		self.vc_ui.pushButton_clear_line.clicked.connect(self.slot_clear_line_action)
		#划分网格
		self.vc_ui.pushButton_split_grid.clicked.connect(self.slot_split_grid_action)
		#计算平整度
		self.vc_ui.pushButton_calculate_smooth.clicked.connect(self.slot_calculate_flatness_action)
		#清除网格
		self.vc_ui.pushButton_clear_grid.clicked.connect(self.slot_clear_grid_action)
		pass
	#

	def mouseDoubleClickEvent(self, event):
		#
		pass
	#

	#
	def wait_connect_action(self):
		c, addr = self.server_socket.accept()
		self.send_data_to_webots = True
		pass
	#
	def send_data_to_webots_action(self):
		if self.send_data_to_webots:
			print("send data to webots")

		pass

	#
	def on_combobox_changed(self):
		self.car_direct = int(self.vc_ui.comboBox_car_direct.currentText())
	#

	#
	def handleTcpSocketConnected(self):
		print("Connected! Ready to chat! :)")
		self.vc_ui.textEdit_run_data.append("car connected...")
		pass
	#

	#打开Realsense
	def slot_connect_realsense_action(self):
		if self.timer_camera.isActive() == False:
			#flag = self.camera.open(self.CAM_ID)
			self.pipeline.start(self.config)
			self.out_pointcloud_img = np.empty((self.frame_height, self.frame_width, 3), dtype=np.uint8)
			#刷新频率
			self.timer_camera.start(30)
			
		pass
	#

	#
	def on_double_clicked_action(self,point_param):
		index_point_x = point_param.x()
		index_point_y = point_param.y()
		if self.click_count==0:
			self.vc_ui.lineEdit_startPoint_x.setText(str(index_point_x))
			self.vc_ui.lineEdit_startPoint_y.setText(str(index_point_y))
			self.click_count=1
		elif self.click_count==1:
			self.vc_ui.lineEdit_endPoint_x.setText(str(index_point_x))
			self.vc_ui.lineEdit_endPoint_y.setText(str(index_point_y))
			self.click_count=0
		#

		if len(self.click_points)>=4:
			self.click_points.pop(0)
		self.click_points.append(point_param)
		
		pass
	#

	#
	def  slot_set_line_action(self):
		start_point_x = int(self.vc_ui.lineEdit_startPoint_x.text())
		start_point_y = int(self.vc_ui.lineEdit_startPoint_y.text())
		end_point_x = int(self.vc_ui.lineEdit_endPoint_x.text())
		end_point_y = int(self.vc_ui.lineEdit_endPoint_y.text())

		start_point=QPoint(start_point_x,start_point_y)
		end_point=QPoint(end_point_x,end_point_y)
		self.vc_ui.label_img.addLine(start_point,end_point)
		pass
	#

	#
	def slot_split_grid_action(self):
		is_validate=False
		if len(self.click_points)==4:
			if self.click_points[0].x()<self.click_points[2].x() and self.click_points[1].x()>self.click_points[3].x():
				is_validate=True
			else:
				self.showMessage("请绘制顺时针轮廓")
				self.grid_data=[]
				self.click_points=[]
				return
		else:
			self.showMessage("轮廓只能有四个顶点")
			self.grid_data=[]
			self.click_points=[]
			return
		self.grid_data=[]
		self.grid_col = int(self.vc_ui.lineEdit_face_smooth_cols.text())
		self.grid_row = int(self.vc_ui.lineEdit_face_smooth_grid_rows.text())
		#print(self.click_points)
		if is_validate:
			top_line_points=self.divide_segment((self.click_points[0].x(),self.click_points[0].y()),(self.click_points[1].x(),self.click_points[1].y()),self.grid_col)
			bottom_line_points=self.divide_segment((self.click_points[3].x(),self.click_points[3].y()),(self.click_points[2].x(),self.click_points[2].y()),self.grid_col)
			print(len(top_line_points))
			print(len(bottom_line_points))
			
			for k in range(len(top_line_points)):
				#print(k)
				top_point=top_line_points[k]
				bottom_point=bottom_line_points[k]
				col_line_points=self.divide_segment(top_point,bottom_point,self.grid_row)
				print(len(col_line_points))
				for index_point1 in col_line_points:
					grid_point_id = index_point1[1]*self.frame_width+index_point1[0]
					grid_3dpoint_x,grid_3dpoint_y,grid_3dpoint_z = self.vtx_list[grid_point_id][0],self.vtx_list[grid_point_id][1],self.vtx_list[grid_point_id][2]
					if abs(grid_3dpoint_x-0.0)>=0 and abs(grid_3dpoint_y-0.0)>=0 and abs(grid_3dpoint_z-0.0)>=0:
						index_grid_point_data=[index_point1[0],index_point1[1],grid_3dpoint_x,grid_3dpoint_y,grid_3dpoint_z]
						#
						self.grid_data.append(index_grid_point_data)
					#
				#
			#
		#
		self.vc_ui.label_img.setGridPoint(self.grid_data,self.click_points)
		
		pass
	#
	#
	def showMessage(self,text1):
		#创建一个QMessageBox对象
		msg = QMessageBox()
		# 设置消息框的标题
		msg.setWindowTitle("消息框")
		# 设置消息框的文本内容
		msg.setText(text1)
		# 设置信息图标（可以是信息、警告、错误、询问等）
		msg.setIcon(QMessageBox.Information)
		# 显示消息框，并等待用户响应（这里使用exec_方法，它在Qt5中是exec的别名）
		msg.exec_()
		pass
	
	#计算平整度
	def slot_calculate_flatness_action(self):
		if len(self.grid_data)==0:
			self.showMessage("请先划分网格")
			return
		array1 = np.array(self.grid_data)
		# 选择后三列
		selected_columns = array1[:, 2:]

		x = selected_columns[:, 0]
		y = selected_columns[:, 1]
		z = selected_columns[:, 2]

		X = np.column_stack((x, y))
		y_target = z

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		y_scaled = (y_target - np.mean(y_target)) / np.std(y_target)

		param_grid = {'C': [0.1, 1, 10, 100],'epsilon': [0.01, 0.1, 0.5]}
		svr = SVR(kernel='linear')
		grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
		grid_search.fit(X_scaled, y_scaled)
		best_svr = grid_search.best_estimator_
		coef_scaled = best_svr.coef_[0]
		intercept_scaled = best_svr.intercept_[0]

		scale_x, scale_y = scaler.scale_
		mean_x, mean_y = scaler.mean_
		mean_z = np.mean(y_target)
		std_z = np.std(y_target)

		A = coef_scaled[0] * std_z / scale_x
		B = coef_scaled[1] * std_z / scale_y
		C = (intercept_scaled * std_z) + mean_z - A * mean_x - B * mean_y

		distances = np.abs(A * x + B * y - z + C) / np.sqrt(A**2 + B**2 + 1)

		# 计算平整度（距离的标准差）
		flatness1 = np.std(distances, ddof=1)
		flatness2 = np.max(distances)
		
		print("平整度（距离的标准差）:", flatness1)
		print("平整度（离平面最大距离）:", flatness2)
		
		flatness_str="平整度:"+str(flatness1)+"\n"
		self.vc_ui.textEdit_realsense_data_msg.append(flatness_str)
		pass
	#

	#
	def slot_clear_grid_action(self):
		self.grid_data=[]
		self.click_points=[]
		self.vc_ui.label_img.clearGridPoint()
		pass
	#
	#
	def divide_segment(self,p1, p2, n=6):
		x1, y1 = p1
		x2, y2 = p2
		
		# 计算线段的增量
		dx = x2 - x1
		dy = y2 - y1
		
		# 计算每个等分点的增量
		step_dx = dx / n
		step_dy = dy / n
		
		# 存储等分点的坐标
		points = []
		
		# 计算并存储每个等分点的坐标
		for i in range(1, n):
			x = x1 + i * step_dx
			y = y1 + i * step_dy
			points.append((int(x), int(y)))
		
		# 也包括起点和终点（如果需要）
		points.insert(0, p1)
		points.append(p2)
		
		return points
	#
	#
	def slot_calculate_line_length_action(self):
		start_point_x = int(self.vc_ui.lineEdit_startPoint_x.text())
		start_point_y = int(self.vc_ui.lineEdit_startPoint_y.text())
		end_point_x = int(self.vc_ui.lineEdit_endPoint_x.text())
		end_point_y = int(self.vc_ui.lineEdit_endPoint_y.text())
		
		start_point_id = start_point_y*self.frame_width+start_point_x
		start_3dpoint_x,start_3dpoint_y,start_3dpoint_z = self.vtx_list[start_point_id][0],self.vtx_list[start_point_id][1],self.vtx_list[start_point_id][2]
		#
		end_point_id = end_point_y*self.frame_width+end_point_x
		end_3dpoint_x,end_3dpoint_y,end_3dpoint_z = self.vtx_list[end_point_id][0],self.vtx_list[end_point_id][1],self.vtx_list[end_point_id][2]

		#
		line_length=math.sqrt((start_3dpoint_x-end_3dpoint_x)*(start_3dpoint_x-end_3dpoint_x)+(start_3dpoint_y-end_3dpoint_y)*(start_3dpoint_y-end_3dpoint_y)+(start_3dpoint_z-end_3dpoint_z)*(start_3dpoint_z-end_3dpoint_z))
		#
		line_data_str="line-data:p1("+str(start_point_x)+" "+str(start_point_y)+" "+str(start_3dpoint_x)+"m "+str(start_3dpoint_y)+"m "+str(start_3dpoint_z)+"m)   ,p2("+str(end_point_x)+" "+str(end_point_y)+" "+str(end_3dpoint_x)+"m "+str(end_3dpoint_y)+"m "+str(end_3dpoint_z)+"m) 长度："+str(line_length)+"m\n"
		#
		self.vc_ui.textEdit_realsense_data_msg.append(line_data_str)

		pass
	#
	#
	def slot_clear_line_action(self):
		self.vc_ui.textEdit_realsense_data_msg.clear()
		self.vc_ui.label_img.clearLine()
		pass
	#
	
	#刷新显示
	def show_camera_action(self):
		self.state.reset()
		
		color_intrin, depth_intrin, color_image, depth_image, aligned_depth_frame,color_frame = self.get_aligned_images()

		###############
		self.w, self.h = depth_intrin.width, depth_intrin.height
		points = self.point_cloud.calculate(aligned_depth_frame)
		self.point_cloud.map_to(color_frame)

		# Pointcloud data to arrays
		v, t = points.get_vertices(), points.get_texture_coordinates()
		verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
		texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
		#xyz_map = depth_image.get_xyz_map()
		self.vtx_list = verts


		
		####################

		
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		
		images = np.hstack((color_image, depth_colormap))
		
		self.img_depth = cv2.resize(depth_colormap, (320, 240))
		self.img_test2 = cv2.resize(depth_colormap, (320, 240))
		#self.out_pointcloud_img = cv2.resize(self.out_pointcloud_img, (320, 240))
		
		self.image = color_image.copy()
		
		self.cameraImg =self.image
		#self.cameraImg = cv2.resize(self.image, (320, 240))
		
		self.im0 = self.cameraImg.copy()
		self.cameraImg = cv2.cvtColor(self.cameraImg,cv2.COLOR_BGR2RGB)
		self.img_depth = cv2.cvtColor(self.img_depth,cv2.COLOR_BGR2RGB) 
		

		
		#
		
		#建图
		#self.cameraImg = self.create_map(self.cameraImg)
		#
		
		#self.cameraImg = self.draw_car_and_worker_route(self.cameraImg)
		
		if self.check_count>20:
			self.check_count=0
			#
			length_threshold = 100  # 控制显示的最小线段长度（单位：像素）
			img_line=check_line(self.cameraImg,length_threshold,self.vtx_list,self.frame_width)

			self.showImage = QtGui.QImage(img_line.data, self.cameraImg.shape[1], self.cameraImg.shape[0], QtGui.QImage.Format_RGB888)
			self.vc_ui.label_img.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
		#

		#self.showImage = QtGui.QImage(self.cameraImg.data, self.cameraImg.shape[1], self.cameraImg.shape[0], QtGui.QImage.Format_RGB888)
		self.showImage2 = QtGui.QImage(self.img_depth.data, self.img_depth.shape[1], self.img_depth.shape[0], QtGui.QImage.Format_RGB888)
		
		#self.vc_ui.label_img.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
		self.vc_ui.label_realsense_depth.setPixmap(QtGui.QPixmap.fromImage(self.showImage2))
		
		'''
		#self.vc_ui.label_5.setPixmap(QtGui.QPixmap.fromImage(self.showImage3))
		self.vc_ui.label_3.setPixmap(QtGui.QPixmap.fromImage(self.showImage2))
		self.vc_ui.label_4.setPixmap(QtGui.QPixmap.fromImage(self.showImage))

		if self.isCheck==True:
			self.checkObj1()
		else:
			self.showImageCheck = QtGui.QImage(color_image.data, color_image.shape[1], color_image.shape[0], QtGui.QImage.Format_RGB888)
			self.vc_ui.label_2.setPixmap(QtGui.QPixmap.fromImage(self.showImageCheck))
		'''

		self.check_count=self.check_count+1
		pass
	#


		

	#
	def draw_obj_rect(self,img_param):
		empty_img = np.zeros((self.frame_height,self.frame_width,3),dtype=np.uint8)

		car_3d_x,car_3d_y,car_3d_z=0.0,0.0,0.0
		do_3d_x,do_3d_y,do_3d_z=0.0,0.0,0.0

		car_2d_x,car_2d_y=0,0
		do_2d_x,do_2d_y=0,0

		#
		self.goal_3d_positions=[]
		
		column_id= 1
		for index_rect_id in range(len(self.list_box_rect)):
			index_rect = self.list_box_rect[index_rect_id]
			index_class = self.list_class[index_rect_id]

			index_point = self.list_bottom_center_points[index_rect_id]
			
			
			box_start_x = int(index_rect[0])
			box_start_y = int(index_rect[1])
			box_end_x = int(index_rect[2])
			box_end_y = int(index_rect[3])

			center_point_x = index_point[0]
			center_point_y = index_point[1]
			
			
			
			#print(index_class)
			#if box_end_y<740 and abs(box_end_y-box_start_y)<180:
			if 1==2:
				point_id2 = (box_end_y-10-1)*self.frame_width+center_point_x
				point_id2_3d_x,point_id2_3d_y,point_id2_3d_z = self.vtx_list[point_id2][0],self.vtx_list[point_id2][1],self.vtx_list[point_id2][2]
			
			#
			#	1
			#2		4
			#	3
			#
			#if self.car_direct==1:
			
			
			#		
		#

		

		img_param = cv2.addWeighted(img_param, self.rect_alpha, empty_img, 1 - self.rect_alpha, 0)

		#
		
		
		return img_param
	#

	


	def get_goal_positions(self):
		return self.goal_3d_positions
	


	

	#
	#获取相机数据
	def get_aligned_images(self):
		
		frames = self.pipeline.wait_for_frames()  #等待获取图像帧
		aligned_frames = self.align.process(frames)  #获取对齐帧
		aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
		color_frame = aligned_frames.get_color_frame()   #获取对齐帧中的color帧
		#depth_sensor = self.pipeline.get_depth_sensor()
		
		############### 相机参数的获取 #######################
		intr = color_frame.profile.as_video_stream_profile().intrinsics   #获取相机内参
		depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  #获取深度参数（像素坐标系转相机坐标系会用到）
		
		
		depth_image = np.asanyarray(aligned_depth_frame.get_data())  #深度图（默认16位）
		depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  #深度图（8位）
		depth_image_3d = np.dstack((depth_image_8bit,depth_image_8bit,depth_image_8bit))  #3通道深度图
		color_image = np.asanyarray(color_frame.get_data())  # RGB图
		
		#返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
		return intr, depth_intrin, color_image, depth_image, aligned_depth_frame,color_frame

	#


	#
	#
	def realsense_tool(self,v):
		"""project 3d vector array to 2d"""
		self.h, self.w = self.out_pointcloud_img.shape[:2]
		view_aspect = float(self.h)/self.w

		# ignore divide by zero for invalid depth
		with np.errstate(divide='ignore', invalid='ignore'):
			proj = v[:, :-1] / v[:, -1, np.newaxis] * (self.w*view_aspect, self.h) + (self.w/2.0, self.h/2.0)

		# near clipping
		znear = 0.03
		proj[v[:, 2] < znear] = np.nan
		return proj


	#


	



#
