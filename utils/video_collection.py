from moviepy.editor import VideoFileClip, TextClip, clips_array, CompositeVideoClip


def concatenate_videos(video_paths, labels, output_path):
    # Load the videos
    clips = [VideoFileClip(path) for path in video_paths]

    # Create text clips for the labels
    text_clips = [TextClip(txt, fontsize=24, color='white').set_position(("left", "top")).set_duration(clip.duration)
                  for txt, clip in zip(labels, clips)]

    # Overlay the text clips on the videos
    clips = [CompositeVideoClip([clip, text]) for clip, text in zip(clips, text_clips)]

    # Arrange the videos into a 3x2 grid
    final_clip = clips_array([[clips[0], clips[1]], [clips[2], clips[3]], [clips[4], clips[5]]])

    # Write the final video to a file
    final_clip.write_videofile(output_path, codec='libx264')


# Usage:
video_paths = ["./results/trim2.mp4",
               "./results/corrected_output_base.mp4",
               "./results/corrected_output_vivid.mp4",
               "./results/corrected_output_color.mp4",
               "./results/corrected_output_expo.mp4",
               "./results/corrected_output_all.mp4",
               ]

labels = ["Raw", "Base", "Color Enhanced", "White Balance Enhanced", "Exposure Enhanced", "All Enhanced"]
output_path = "./results/final.mp4"
concatenate_videos(video_paths, labels, output_path)

if __name__ == "__main__":
    pass