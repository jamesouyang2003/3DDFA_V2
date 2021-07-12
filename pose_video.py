# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
import csv

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import viz_pose
from utils.functions import get_suffix


def main(args):
	cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

	# Init FaceBoxes and TDDFA, recommend using onnx flag
	if args.onnx:
		import os
		os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
		os.environ['OMP_NUM_THREADS'] = '4'

		from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
		from TDDFA_ONNX import TDDFA_ONNX

		face_boxes = FaceBoxes_ONNX()
		tddfa = TDDFA_ONNX(**cfg)
	else:
		gpu_mode = args.mode == 'gpu'
		tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
		face_boxes = FaceBoxes()

	# Given a video path
	fn = args.video_fp.split('/')[-1]
	reader = imageio.get_reader(args.video_fp)

	fps = reader.get_meta_data()['fps']

	suffix = get_suffix(args.video_fp)
	video_wfp = f'test-videos/{fn.replace(suffix, "")}_pose.mp4'
	writer = imageio.get_writer(video_wfp, fps=fps)

	# Get sensor data
	sensor_data = []
	if args.sensor_data != "":
		with open(args.sensor_data, 'r') as sensor_file:
			sensor_reader = csv.reader(sensor_file)
			for row in sensor_reader:
				# sensor_data.append(np.array(row)[0:9].reshape(3, 3).astype(float)) # for rotation matrix instead of 3 axes
				sensor_data.append(np.array(row).astype(float))

	# run
	pre_ver = None
	for i, frame in tqdm(enumerate(reader)):
		frame_bgr = frame[..., ::-1]  # RGB->BGR

		if i == 0:
			# the first frame, detect face, here we only use the first face, you can change depending on your need
			boxes = face_boxes(frame_bgr)
			boxes = [boxes[0]]
			param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
			ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

			# refine
			param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
			ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
		else:
			param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

			roi_box = roi_box_lst[0]
			# todo: add confidence threshold to judge the tracking is failed
			if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
				boxes = face_boxes(frame_bgr)
				boxes = [boxes[0]]
				param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

			ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

		pre_ver = ver  # for tracking

		if args.sensor_data == "":
			res = viz_pose(frame_bgr.copy(), param_lst, [ver], print_angle=False)
		elif i < len(sensor_data):
			res = viz_pose(frame_bgr.copy(), param_lst, [ver], print_angle=False, rotation=sensor_data[i])
		else:
			res = frame_bgr.copy()

		writer.append_data(res[..., ::-1])  # BGR->RGB

	writer.close()
	print(f'Dump to {video_wfp}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
	parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
	parser.add_argument('-f', '--video_fp', type=str)
	parser.add_argument('-d', '--sensor_data', type=str, default="")
	parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
	parser.add_argument('--onnx', action='store_true', default=False)

	args = parser.parse_args()
	main(args)
