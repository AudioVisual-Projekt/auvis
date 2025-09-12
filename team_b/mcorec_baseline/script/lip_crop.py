import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retinaface.detector import LandmarksDetector
from src.retinaface.video_process import VideoProcess
from src.retinaface.utils import save_vid_aud_txt
from tqdm import tqdm
import traceback
import math

import torch
import torchaudio
import torchvision
import json
import ffmpeg
import glob

# ==================== LOAD MODEL ====================

landmarks_detector = LandmarksDetector(device="cuda:0")
video_process = VideoProcess(convert_gray=False)

def process_video(video_path, output_dir=None):
    try:
        # Load and process audio and video
        audio, sample_rate = torchaudio.load(video_path, normalize=True)

        video = torchvision.io.read_video(video_path)[0].numpy()
        landmarks = landmarks_detector(video)
        video = video_process(video, landmarks)
        video = torch.tensor(video)   
        
        segment_name = video_path.split("/")[-1].replace(".mp4", "_lip")
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        os.makedirs(output_dir, exist_ok=True)
        
        dst_vid_filename = os.path.join(output_dir, f"{segment_name}.mp4")
        dst_aud_filename = os.path.join(output_dir, f"{segment_name}.wav")
        text_filename = os.path.join(output_dir, f"{segment_name}.json")
        save_vid_aud_txt(
            dst_vid_filename,
            dst_aud_filename,
            text_filename,
            video,
            audio,
            json.dumps({
                "path": video_path
            }, indent=4),
            video_fps=25,
            audio_sample_rate=16000,
        )

        # Combine audio and video
        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            dst_vid_filename[:-4] + ".av.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run(overwrite_output=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {video_path} segment {segment_frame[0]}-{segment_frame[-1]}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Active Speaker Detection")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    opt = parser.parse_args()
        
    process_video(opt.video)


if __name__ == "__main__":
    main()