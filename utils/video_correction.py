import torch
import cv2

from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def video2video(input_video_path, output_video_path):
    # Open the input video
    vidcap = cv2.VideoCapture(input_video_path)

    # Get the video's width, height, frames per second (fps), and total frame count
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter for the output video
    vidwriter = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Create a tqdm progress bar
    pbar = tqdm(total=total_frames)

    success, frame = vidcap.read()
    while success:
        # Enhance the frame
        # enhanced_frame = process(frame, waternet)

        # Write the enhanced frame to the output video
        vidwriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Read the next frame from the input video
        success, frame = vidcap.read()

        # Update the progress bar
        pbar.update(1)

    # Close the VideoWriter and VideoCapture objects
    # vidwriter.release()
    vidcap.release()

    # Close the progress bar
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='Video Enhancement')
    parser.add_argument('--target', type=str, default='trim2', help='Target video name without extension')
    args = parser.parse_args()

    video2video(f"./results/{args.target}.mp4", f"./results/corrected_{args.target}.mp4")

if __name__ == "__main__":
    main()