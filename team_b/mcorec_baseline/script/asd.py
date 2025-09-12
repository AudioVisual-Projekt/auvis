import os, cv2, math, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torchaudio
import python_speech_features
import json
from src.talking_detector.ASD import ASD

# ==================== LOAD MODEL ====================
ASD_MODEL = ASD()
ASD_MODEL.loadParameters("model-bin/finetuning_TalkSet.model")
ASD_MODEL = ASD_MODEL.cuda().eval()
print("Model loaded successfully.")

def process_video(video_path, output_dir=None, frame_offset=0):
    """
    Process a single video file to detect active speakers and output ASD results.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str, optional): Directory to save the output JSON. If None, saves in same directory as video.
    
    Returns:
        str: Path to the output JSON file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Create output directory if specified
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Load audio directly using torchaudio
    audio, sample_rate = torchaudio.load(video_path, normalize=False)
    assert sample_rate == 16000
    
    # Convert to numpy for MFCC computation
    audio_np = audio[0].numpy()
    
    # Compute MFCC features
    audioFeature = python_speech_features.mfcc(audio_np, 16000, numcep=13, winlen=0.025, winstep=0.010)
    
    # Load video frames
    video = cv2.VideoCapture(video_path)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
        else:
            break
    video.release()
    
    videoFeature = np.array(videoFeature)
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    
    # Evaluate using model
    durationSet = {1,1,1,2,2,2,3,3,4,5,6}
    allScore = []
    
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                embedA = ASD_MODEL.model.forward_audio_frontend(inputA)
                embedV = ASD_MODEL.model.forward_visual_frontend(inputV)
                out = ASD_MODEL.model.forward_audio_visual_backend(embedA, embedV)
                score = ASD_MODEL.lossAV.forward(out, labels=None)
                scores.extend(score)
        allScore.append(scores)
    
    # Calculate final scores
    final_scores = np.round((np.mean(np.array(allScore), axis=0)), 1).astype(float)
    
    # Create frame-wise scores dictionary
    frame_scores = {frame_idx + frame_offset: round(float(score), 2) for frame_idx, score in enumerate(final_scores)}
    
    # Save results
    output_json = os.path.join(output_dir, f"{video_name}_asd.json")
    with open(output_json, 'w') as f:
        json.dump(frame_scores, f, indent=4)
    
    return output_json

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Active Speaker Detection")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output JSON (optional)')
    opt = parser.parse_args()
    
    video_info = opt.video.replace(".mp4", ".json")
    frame_offset = 0
    if os.path.exists(video_info):
        with open(video_info, 'r') as f:
            video_data = json.load(f)
        frame_offset = video_data.get("frame_start", 0)
    output_path = process_video(opt.video, opt.output_dir, frame_offset)
    print(f"ASD results saved to: {output_path}")

if __name__ == "__main__":

    main()