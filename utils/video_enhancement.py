import torch
import cv2

from tqdm import tqdm
import argparse

from waternet.inference_utils import process
from waternet.net import WaterNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def video2video(input_video_path, output_video_path, waternet):
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
        enhanced_frame = process(frame, waternet)

        # Write the enhanced frame to the output video
        vidwriter.write(cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR))

        # Read the next frame from the input video
        success, frame = vidcap.read()

        # Update the progress bar
        pbar.update(1)

    # Close the VideoWriter and VideoCapture objects
    vidwriter.release()
    vidcap.release()

    # Close the progress bar
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='Video Enhancement')
    parser.add_argument('--target', type=str, default='trim2', help='Target video name without extension')
    parser.add_argument('--network', type=str, default='base', help='Network name: base, vivid_mid, color_cast, exposure, all0_last')
    args = parser.parse_args()

    kinds = {
        "base": "weights/pretrained/waternet.pt",
        "vivid_mid": "weights/color-enhanced.pt",
        "color_cast": "weights/wb-enhanced.pt",
        "exposure": "weights/expo-enhanced.pt",
        "all0_last": "weights/all-enhanced.pt",
    }

    waternet = WaterNet()
    check_point = torch.load(f'./{kinds[args.network]}', map_location=device)
    waternet.load_state_dict(check_point)
    waternet.eval()
    waternet = waternet.to(device)

    video2video(f"./results/{args.target}.mp4", f"./results/{args.target}_{args.network}.mp4", waternet)

if __name__ == "__main__":
    main()