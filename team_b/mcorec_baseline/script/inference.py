import os
import sys
import argparse
import json
import math
import glob
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import torchvision
import torchaudio

# Add src to path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.tokenizer.spm_tokenizer import TextTransform
from src.talking_detector.segmentation import segment_by_asd
from src.cluster.conv_spks import (
    get_speaker_activity_segments, 
    calculate_conversation_scores, 
    cluster_speakers, 
    get_clustering_f1_score
)


class BaseInferenceModel(ABC):
    """Abstract base class for all inference models"""
    
    def __init__(self, checkpoint_path=None, cache_dir=None, beam_size=3):
        self.model = None
        self.text_transform = None
        self.av_data_collator = None
        self.beam_search = None
        self.tokenizer = None
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir or "./model-bin"
        self.beam_size = beam_size
        
    @abstractmethod
    def load_model(self):
        """Load the specific model architecture"""
        pass
    
    @abstractmethod
    def inference(self, videos, audios, **kwargs):
        """Perform inference on audio-visual data"""
        pass
    
    def get_tokenizer_paths(self):
        """Get paths for tokenizer files"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sp_model_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000.model")
        dict_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000_units.txt")
        return sp_model_path, dict_path


class AVSRCocktailModel(BaseInferenceModel):
    """AVSR Cocktail model implementation"""
    
    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
        from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model
        model_path = self.checkpoint_path or "./model-bin/avsr_cocktail"
        print(f"Loading model from {model_path}")
        avsr_model = AVHubertAVSR.from_pretrained(model_path)
        avsr_model.eval().cuda()
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)
    
    def inference(self, videos, audios, **kwargs):
        avhubert_features = self.model.encoder(
            input_features=audios, 
            video=videos,
        )
        audiovisual_feat = avhubert_features.last_hidden_state
        audiovisual_feat = audiovisual_feat.squeeze(0)
        
        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class AutoAVSRModel(BaseInferenceModel):
    """Auto AVSR model implementation"""
    
    def load_model(self):
        from src.dataset.av_dataset import AudioTransform, VideoTransform, DataCollator
        from src.auto_avsr.configuration_avsr import AutoAVSRConfig
        from src.auto_avsr.avsr_model import AutoAVSR, get_beam_search_decoder
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model    
        avsr_config = AutoAVSRConfig()
        avsr_model = AutoAVSR(avsr_config)    
        ckpt_path = self.checkpoint_path or "./model-bin/auto_avsr/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
        print(f"Loading model from {ckpt_path}")
        pretrained_weights = torch.load(ckpt_path, weights_only=True)
        avsr_model.avsr.load_state_dict(pretrained_weights)
        avsr_model.eval().cuda()
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)
    
    def inference(self, videos, audios, **kwargs):
        video_feat, _ = self.model.encoder(videos, None)
        audio_feat, _ = self.model.aux_encoder(audios, None)
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class MuAViCModel(BaseInferenceModel):
    """MuAViC model implementation"""
    
    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_muavic.avhubert2text import AV2TextForConditionalGeneration
        from transformers import Speech2TextTokenizer
        
        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")
        
        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )
        
        # Load model
        model_name = self.checkpoint_path or 'nguyenvulebinh/AV-HuBERT-MuAViC-en'
        print(f"Loading model from {model_name}")
        self.model = AV2TextForConditionalGeneration.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir
        )
        self.tokenizer = Speech2TextTokenizer.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir
        )
        self.model = self.model.cuda().eval()
    
    def inference(self, videos, audios, **kwargs):
        attention_mask = torch.BoolTensor(audios.size(0), audios.size(-1)).fill_(False).cuda()
        output = self.model.generate(
            audios,
            attention_mask=attention_mask,
            video=videos,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].upper()
        return output


class InferenceEngine:
    """Main inference engine that handles model selection and processing"""
    
    def __init__(self, model_type: str, checkpoint_path=None, cache_dir=None, beam_size=3, max_length=15):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.beam_size = beam_size
        self.max_length = max_length
        self.model_impl = self._get_model_implementation()
        
    def _get_model_implementation(self) -> BaseInferenceModel:
        """Factory method to get the appropriate model implementation"""
        if self.model_type == "avsr_cocktail":
            return AVSRCocktailModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        elif self.model_type == "auto_avsr":
            return AutoAVSRModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        elif self.model_type == "muavic_en":
            return MuAViCModel(self.checkpoint_path, self.cache_dir, self.beam_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_model(self):
        """Load the selected model"""
        print(f"Loading {self.model_type} model...")
        self.model_impl.load_model()
        print(f"{self.model_type} model loaded successfully!")
    
    def chunk_video(self, video_path, asd_path=None, max_length=15):
        """Split video into chunks for inference"""
        if asd_path is not None:
            with open(asd_path, "r") as f:
                asd = json.load(f)

            # Convert frame numbers to integers and sort them
            frames = sorted([int(f) for f in asd.keys()])
            # Find the minimum frame number to normalize frame indices
            min_frame = min(frames)

            segments_by_frames = segment_by_asd(asd, {
                "max_chunk_size": max_length,  # in seconds
            })
            # Normalize frame indices, for inference, don't care about the actual frame indices
            segments = [((seg[0] - min_frame) / 25, (seg[-1] - min_frame) / 25) for seg in segments_by_frames]

        else:
            # Get video duration
            audio, rate = torchaudio.load(video_path)
            video_duration = audio.shape[1] / rate
            # num chunks
            num_chunks = math.ceil(video_duration / max_length)
            chunk_size = math.ceil(video_duration / num_chunks)
            segments = []
            # Convert to integer steps for range
            steps = int(video_duration * 100)  # Convert to centiseconds for precision
            step_size = int(chunk_size * 100)
            for i in range(0, steps, step_size):
                start_time = i / 100
                end_time = min((i + step_size) / 100, video_duration)
                segments.append((start_time, end_time))
            
        return segments
    
    def format_vtt_timestamp(self, timestamp):
        """Format timestamp for VTT output"""
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp - int(timestamp)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def infer_video(self, video_path, asd_path=None, offset=0., desc=None):
        """Perform inference on a video file"""
        segments = self.chunk_video(video_path, asd_path, max_length=self.max_length)
        segment_output = []
        
        for seg in tqdm(segments, desc="Processing segments" if desc is None else desc, total=len(segments)):
            # Prepare sample
            sample = {
                "video": video_path,
                "start_time": seg[0],
                "end_time": seg[1],
            }
            sample_features = self.model_impl.av_data_collator([sample])
            audios = sample_features["audios"].cuda()
            videos = sample_features["videos"].cuda()
            audio_lengths = sample_features["audio_lengths"].cuda()
            video_lengths = sample_features["video_lengths"].cuda()
            
            try:
                output = self.model_impl.inference(videos, audios)
            except Exception as e:
                print(f"Error during inference for segment {sample}")
                raise e
            
            segment_output.append(output)

            # GPU Memory Cleanup
            del audios, videos, audio_lengths, video_lengths, sample_features
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return [
            {
                "start_time": seg[0] + offset,
                "end_time": seg[1] + offset,
                "text": output
            } for seg, output in zip(segments, segment_output)
        ]
    
    def mcorec_session_infer(self, session_dir, output_dir):
        """Process a complete MCoReC session"""
        # Load session metadata
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        # Process speaker clustering
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = []
            for track in speaker_data['central']['crops']:
                list_tracks_asd.append(os.path.join(session_dir, track['asd']))
            uem_start = speaker_data['central']['uem']['start']
            uem_end = speaker_data['central']['uem']['end']
            speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)
            speaker_segments[speaker_name] = speaker_activity_segments
        
        scores = calculate_conversation_scores(speaker_segments)
        clusters = cluster_speakers(scores, list(speaker_segments.keys()))   
        output_clusters_file = os.path.join(output_dir, "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)    
        
        # Process speaker transcripts
        for speaker_name, speaker_data in tqdm(metadata.items(), desc="Processing speakers", total=len(metadata)):
            print()
            speaker_track_hypotheses = []
            for idx, track in enumerate(speaker_data['central']['crops']):
                video_path = os.path.join(session_dir, track['lip'])
                asd_path = os.path.join(session_dir, track['asd']) if 'asd' in track else None
                with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
                    crop_metadata = json.load(f)
                track_start_time = crop_metadata['start_time']
                hypotheses = self.infer_video(video_path, asd_path, offset=track_start_time, desc=f"Processing speaker {speaker_name} track {idx+1} of {len(speaker_data['central']['crops'])}")
                speaker_track_hypotheses.extend(hypotheses)

                # GPU Memory Cleanup after each track
                torch.cuda.empty_cache()

            output_file = os.path.join(output_dir, f"{speaker_name}.vtt")
            with open(output_file, "w") as f:
                f.write("WEBVTT\n\n")
                for hyp in speaker_track_hypotheses:
                    text = hyp["text"].strip().replace("<unk>", "").strip()
                    start_time = self.format_vtt_timestamp(hyp["start_time"])
                    end_time = self.format_vtt_timestamp(hyp["end_time"])
                    if len(text) == 0:
                        continue
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Unified inference script for multiple AVSR models")
    
    # Model selection argument
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
        choices=['avsr_cocktail', 'auto_avsr', 'muavic_en'],
        help='Type of model to use for inference'
    )
    
    # Input/output arguments
    parser.add_argument(
        '--session_dir', 
        type=str, 
        required=True, 
        help='Path to folder containing session data (supports glob patterns with *)'
    )
    
    # Model checkpoint arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to model checkpoint or pretrained model name'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./model-bin',
        help='Directory to cache downloaded models (default: ./model-bin)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--max_length',
        type=int,
        default=15,
        help='Maximum length of video segments in seconds (default: 15)'
    )
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='Beam size for beam search decoding (default: 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='output',
        help='Name of the output directory within each session (default: output)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(args.model_type, args.checkpoint_path, args.cache_dir, args.beam_size, args.max_length)
    engine.load_model()
    
    # Process session directories
    if args.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(args.session_dir)
    else:
        all_session_dirs = [args.session_dir]
    
    print(f"Inferring {len(all_session_dirs)} sessions using {args.model_type} model")
    
    for session_dir in all_session_dirs:
        output_dir = os.path.join(session_dir, args.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        session_name = session_dir.split('/')[-1]
        print(f"Processing session {session_name}")
        
        if args.verbose:
            print(f"  Model: {args.model_type}")
            print(f"  Input: {session_dir}")
            print(f"  Output: {output_dir}")
        
        engine.mcorec_session_infer(session_dir, output_dir)
        
        if args.verbose:
            print(f"  Completed session {session_name}")


if __name__ == "__main__":
    main()
