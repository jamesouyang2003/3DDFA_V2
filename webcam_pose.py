# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import viz_pose


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

	# Given a camera
	# before run this line, make sure you have installed `imageio-ffmpeg`
	reader = imageio.get_reader("<video0>")

	# the simple implementation of average smoothing by looking ahead by n_next frames
	# assert the frames of the video >= n
	n_pre, n_next = args.n_pre, args.n_next
	n = n_pre + n_next + 1
	queue_ver = deque()
	queue_frame = deque()

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

			# padding queue
			for _ in range(n_pre):
				queue_ver.append(ver.copy())
			queue_ver.append(ver.copy())

			for _ in range(n_pre):
				queue_frame.append(frame_bgr.copy())
			queue_frame.append(frame_bgr.copy())
		else:
			param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

			roi_box = roi_box_lst[0]
			# todo: add confidence threshold to judge the tracking is failed
			if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
				boxes = face_boxes(frame_bgr)
				boxes = [boxes[0]]
				param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

			ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

			queue_ver.append(ver.copy())
			queue_frame.append(frame_bgr.copy())

		pre_ver = ver  # for tracking

		# smoothing: enqueue and dequeue ops
		if len(queue_ver) >= n:
			ver_ave = np.mean(queue_ver, axis=0)

			# get pose image
			img_draw = viz_pose(queue_frame[n_pre], param_lst, [ver_ave], print_angle=False)

			cv2.imshow('image', img_draw)
			k = cv2.waitKey(20)
			if (k & 0xff == ord('q')):
				break

			queue_ver.popleft()
			queue_frame.popleft()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
	parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
	parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
	parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
	parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
	parser.add_argument('--onnx', action='store_true', default=False)

	args = parser.parse_args()
	main(args)
