import cv2
import argparse
import os


def load_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str)
	parser.add_argument('--frame_path', type=str)

	args = parser.parse_args()
	
	return args


def video2frame(video_path, frame_path):
	path_lst = video_path.split(" ")
	count = 0

	for path in path_lst:
		vidcap = cv2.VideoCapture(path)
		success, frame = vidcap.read()
		if not success:
			print("Video path non valid, check again: ", path)
			return
		vidcap.release()

	if not os.path.exists(frame_path):
		print("Frame path non valid, check again: ", frame_path)	
		return 

	for path in path_lst:
		print "Loading from ", path
		vidcap = cv2.VideoCapture(path)
		success, frame = vidcap.read()
		while success:
			if count % 2 == 0:
				if count % 30 == 0:
					idx = int(count/30)
					cv2.imwrite(frame_path + "frame%06d.jpg" % idx, frame)
				else:
					idx_frame = int(count/30)
					idx_add = int((count % 30) / 2)
					cv2.imwrite(frame_path + "frame%06d_%02d.jpg" % (idx_frame, idx_add), frame)

			success, frame = vidcap.read()
			count += 1
			if count % 2000 == 0:
				print count
	
		vidcap.release()
	
	print("Frames saved at {}".format(frame_path))
	

def main():
	args = load_args()
	print(args)
	video2frame(args.video_path, args.frame_path)


if __name__ == "__main__":
	main()	

