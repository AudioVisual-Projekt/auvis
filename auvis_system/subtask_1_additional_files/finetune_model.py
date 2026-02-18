import time
import re
import pandas as pd

import os, cv2, math, sys
import numpy as np
import torch
import torchaudio
import python_speech_features
import json

from src.talking_detector.ASD import ASD


def train_network(loader, epoch):

    durationSet = {1,1,1,2,2,2,3,3,4,5,6}
    allScore = []

    print(f"Anzahl zu verarbeitender Tracks: {len(loader)}")

    for j in range(len(loader)):

        print(j)

        audioFeature = loader[j][0]
        videoFeature = loader[j][1]
        label = loader[j][2]
    
        print(audioFeature.shape)
        print(videoFeature.shape)
        print(label.shape)
        
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            print(batchSize)
            ASD_MODEL.scheduler.step(epoch - 1)  # StepLR
            index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
            lr = ASD_MODEL.optim.param_groups[0]['lr']
            scores = []
            for i in range(batchSize):

                ASD_MODEL.zero_grad()
        
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()  # torch.Size([1, 256, 13])
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()  # torch.Size([1, 64, 112, 112])
                labelASD = label[i * duration * 25: (i+1) * duration * 25].cuda()
                # print(i, labelASD.shape)
                
                embedA = ASD_MODEL.model.forward_audio_frontend(inputA)  # torch.Size([1, 64, 128])
                embedV = ASD_MODEL.model.forward_visual_frontend(inputV)  # torch.Size([1, 64, 128])
        
                outsAV= ASD_MODEL.model.forward_audio_visual_backend(embedA, embedV)
                outsV = ASD_MODEL.model.forward_visual_backend(embedV)

                if not outsAV.size()[0] == labelASD.size()[0]:
                    print(f"Size not matching: {outsAV.size()}, {labelASD.size()}")
                    labelASD = (label[i * duration * 25: (i+1) * duration * 25] + [0]).cuda()
                
                score = ASD_MODEL.lossAV.forward(outsAV, labels=None)
                scores.extend(score)

                nlossAV, _, _, prec = ASD_MODEL.lossAV.forward(outsAV, labels=labelASD)
                nlossV = ASD_MODEL.lossV.forward(outsV, labels=labelASD)
                nloss = nlossAV + 0.5 * nlossV

                lossV += nlossV.detach().cpu().numpy()
                lossAV += nlossAV.detach().cpu().numpy()
                loss += nloss.detach().cpu().numpy()
                top1 += prec
                nloss.backward()
                ASD_MODEL.optim.step()
                index += len(labelASD)
    
                num = i+1
                
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
                " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
                sys.stderr.flush()   
    
        sys.stdout.write("\n")      
    
    return lossAV/num, lr


def get_frame_offset(video_path):
    video_info = video_path.replace(".mp4", ".json")
    frame_offset = 0
    if os.path.exists(video_info):
        with open(video_info, 'r') as f:
            video_data = json.load(f)
        print(video_data)
        frame_offset = video_data.get("frame_start", 0)
    return frame_offset


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

    print(audioFeature.shape)
    print(videoFeature.shape)
    
    # Evaluate using model
    durationSet = {1,1,1,2,2,2,3,3,4,5,6}
    allScore = []
    
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()  # torch.Size([1, 256, 13])
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()  # torch.Size([1, 64, 112, 112])
                
                embedA = ASD_MODEL.model.forward_audio_frontend(inputA)  # torch.Size([1, 64, 128])
                embedV = ASD_MODEL.model.forward_visual_frontend(inputV)  # torch.Size([1, 64, 128])

                outsAV= ASD_MODEL.model.forward_audio_visual_backend(embedA, embedV)  
                outsV = ASD_MODEL.model.forward_visual_backend(embedV)

                score = ASD_MODEL.lossAV.forward(outsAV, labels=None)
                scores.extend(score)

    return length, audioFeature, videoFeature, outsAV, outsV, scores


############################# MAIN Function ##########################

# LOAD MODEL
ASD_MODEL = ASD()
ASD_MODEL.loadParameters("model-bin/finetuning_TalkSet.model")
ASD_MODEL = ASD_MODEL.cuda().train()

# LOAD DATA
train_path = "/home/maegg004/Projektgruppe/mcorec/dataset/train"
output_dir = "/home/maegg004/Projektgruppe/mcorec/dataset/output"

loader = list()  # [(audioFeature, videoFeature, label)]

print(os.listdir(train_path))

for session in os.listdir(train_path):

    print(session)

    for speaker in os.listdir(f"{train_path}/{session}/speakers"):

        print(speaker)

        files = os.listdir(f"{train_path}/{session}/speakers/{speaker}/central_crops")
        
        pattern = re.compile(r'^(track_\d{2})\.mp4$')
        
        tracks = [m.group(1) for f in files if (m := pattern.match(f))]

        for track in tracks:

            print(track)
        
            video_path = f"{train_path}/{session}/speakers/{speaker}/central_crops/{track}.mp4"
            print(video_path)

            # process video
            frame_offset = get_frame_offset(video_path)
            length, audioFeature, videoFeature, outsAV, outsV, scores = process_video(video_path, output_dir, frame_offset)

            # load label from csv
            df = pd.read_csv(f"{train_path}/{session}/labels/{speaker}_{track}.csv", index_col="frame")
            labels_np = df.loc[:, "label"].to_numpy(dtype="float32")  # label as numpy array
            label = torch.from_numpy(labels_np)  # label as torch tensor

            loader.append((audioFeature, videoFeature, label))

# TRAINING
train_network(loader, epoch=20)
torch.save(ASD_MODEL.state_dict(), "/home/maegg004/Projektgruppe/mcorec/task_1_active_speaker_detection/models/finetuning_MCoRec_20_epoch.model")

