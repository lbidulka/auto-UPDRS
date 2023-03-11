import cv2
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="input video dataset path", type=str)
    parser.add_argument("--outframes_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="output frame data path", type=str)
    return parser.parse_args()

'''
Converts videos to a sequence of png frames
'''
def main():
    input_args = get_args()
    
    subj = "9769"
    task = "free_form_oval_" #"tug_stand_walk_sit_"
    ch = "CH3"

    in_file = input_args.videos_path + subj + '/vids/' + task + ch + '.mp4'
    out_path = input_args.outframes_path + subj + '/' + task + ch + '/frames/'

    # Make sure its all good to go
    if not os.path.exists(in_file):
        print("ERR Input file not found: ", in_file)
        return
    if not os.path.exists(out_path):
        print("Creating output directory: ", out_path)
        os.makedirs(out_path)
    capture = cv2.VideoCapture(in_file)
    if not capture.isOpened():
        print("ERR Cannot open input file")
        return

    # Process frames
    print("Processing: ", in_file, "...")
    frame_nr = 0
    while(True):
        ret, frame = capture.read()
        if ret: 
            cv2.imwrite(out_path + '/' + str(frame_nr) + '.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            break
        frame_nr += 1
    capture.release()
    print("Done.")

if __name__ == '__main__':
    main()
