import os
import argparse
from tqdm import tqdm

def load_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--nb', type=int)

	args = parser.parse_args()
	
	return args


def select_frame(nb):
	direc = "Frames/Debate{}/".format(nb)
	new_direct = "Frames_fps1/Debate{}/".format(nb)
	num = len(os.listdir(direc))
	print("Total frames: ", num)

	for i in tqdm(range(num)):
		path = direc + "frame%06d.jpg"%i		
		if os.path.exists(path):
			if i % 30 == 0:
				idx = int(i / 30)
				new_path = new_direct + "frame%06d.jpg"%idx 
				os.rename(path, new_path)
			else:
				os.remove(path)
	print("Finished!")
	

def main():
	args = load_args()
	print(args)
	select_frame(args.nb)


if __name__ == "__main__":
	main()	

