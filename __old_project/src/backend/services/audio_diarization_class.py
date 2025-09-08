
import os
import pandas as pd
import numpy as np
import whisper
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


class Audio_Diarization(object):
    
    WHISPER_MODELS = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'tiny', 'base', 'small', 'medium', 'large', 'turbo']
    ## models tiny, base und small are not recommended here because of low transcription performance
    ## they are still listed for test purposes
    
    def __init__(self, wav_fname, model_type='turbo'):
        self.wav_fname = wav_fname  # audio Filename
        if model_type in self.WHISPER_MODELS:
            self.model_type = model_type
        else:
            raise ValueError(f"Ungültiger Wert für das Whisper Modell: {model_type}. Erlaubt sind {self.WHISPER_MODELS}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model(self.model_type).to(self.device)
        self.dotenv_path =  './.env'    # Contains Huggingface Token
        self.pyannote_token = self.get_pyannote_access_token()
        
        self.whisper_inference_result = None  # Ergebnis-Cache für whisper transcript
        self.diarization = None # Ergebnis-Cache für pyannote diarization in RTTM Format
        self.text_df = None  
        self.speaker_df = None
        self.text_speaker_df_raw = None
        self.text_speaker_df = None
        self.output_str = None
        self.out_fname = self.wav_fname[:-4] + '_transcription.txt'
        
 
    def wav_to_transcript(self):
        
        self.text_df = self.whisper_inference_with_segments_df() 
        self.text_df = self.text_df.reset_index() 
        self.speaker_df = self.pyannote_inference_df()
        self.text_speaker_df_raw = self.combine_whisper_and_pyannote()
        self.text_speaker_df = self.combine_consecutive_speakers()
        self.output_str = self.text_speaker_df_to_text()
        self.save_to_file(self.output_str)
        
        return self.output_str
        
    
    def save_to_file(self, text):
        
        with open(self.out_fname, 'w', encoding='utf-8') as f:
            f.write(text)


    def whisper_inference(self):
        
        if self.whisper_inference_result is None:
            self.whisper_inference_result = self.model.transcribe(self.wav_fname, verbose=False)
        return self.whisper_inference_result
    

    def whisper_inference_with_segments_df(self):
        
        result = self.whisper_inference()
        all_seg_df_list = []
        
        for this_seg in result['segments']:
            this_seg.pop('tokens', None)
            this_df = pd.DataFrame.from_dict({0:this_seg}, orient='index')
            all_seg_df_list.append(this_df)
        
        all_seg_df = pd.concat(all_seg_df_list, axis=0)
        all_seg_df.set_index('id')
        
        return all_seg_df
   
    
    def get_pyannote_access_token(self):
        
        load_dotenv(self.dotenv_path)
        pyannote_token = os.environ.get('PYANNOTE_ACCESS_TOKEN')
        
        if pyannote_token is None:
            raise ValueError("PYANNOTE_ACCESS_TOKEN konnte nicht geladen werden. Bitte .env prüfen.")

        return pyannote_token

    
    def pyannote_inference_df(self):
        
        # apply_pyannote_model
        pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=self.pyannote_token)

        # run the pipeline on an audio file
        self.diarization = pipeline(self.wav_fname)
        
        # write pyannote Output in DataFrame
        seg_info_list = []
        for speech_turn, track, speaker in self.diarization.itertracks(yield_label=True):
            
            this_seg_info = {'start': np.round(speech_turn.start,2),
                            'end': np.round(speech_turn.end,2),
                            'speaker': speaker}
            this_df = pd.DataFrame.from_dict({track: this_seg_info}, orient='index')
            
            seg_info_list.append(this_df)
            
        all_seg_infos_df = pd.concat(seg_info_list, axis=0)
        all_seg_infos_df = all_seg_infos_df.reset_index()
    
        return all_seg_infos_df


    def combine_whisper_and_pyannote(self):
            
        # find overlapping speakers for each text segment
        overlap_list = []

        for idx, this_row in self.speaker_df.iterrows():
            
            this_start = this_row['start']
            this_end = this_row['end']
            this_speaker = this_row['speaker']
            
            # wenn text seg endet, bevor speaker starts oder 
            # wenn text seg startet, nachdem speaker endet
            # dann KEIN Overlap, ~ ist ein bitweises NOT (NOT False = True)
            # xx_inds ist True für einen Overlap
            xx_inds = ~((self.text_df['end'] < this_start) | (self.text_df['start'] > this_end))
            
            this_overlap_texts = self.text_df.loc[xx_inds,:].copy()
            this_overlap_texts['speaker_start'] = this_start
            this_overlap_texts['speaker_end'] = this_end
            this_overlap_texts['speaker'] = this_speaker
            overlap_list.append(this_overlap_texts)
            
        all_overlaps = pd.concat(overlap_list)
        all_overlaps = all_overlaps.reset_index(drop=True)
            
        #  compute overlap durations
        all_overlaps['max_start'] = np.maximum(all_overlaps['start'],
                                            all_overlaps['speaker_start'])
        all_overlaps['min_end'] = np.minimum(all_overlaps['end'],
                                            all_overlaps['speaker_end'])
        all_overlaps['overlap_duration'] = all_overlaps['min_end'] - all_overlaps['max_start']
        
        # pick only one text/speaker combination for each text

        max_overlap_indices = all_overlaps.groupby('id')['overlap_duration'].idxmax()
        text_speaker_df = all_overlaps.loc[max_overlap_indices, :]
        
        return text_speaker_df

    def combine_consecutive_speakers(self):

        text_speaker_df = self.text_speaker_df_raw.copy()

        n_iter = text_speaker_df.shape[0]

        for counter in range(1, n_iter ): 
            is_same_speaker = (text_speaker_df['speaker'].iloc[counter] == text_speaker_df['speaker'].iloc[counter-1])
            
            if is_same_speaker:
                new_start = text_speaker_df['start'].iloc[counter-1]
                previous_text = text_speaker_df['text'].iloc[counter-1]
                new_text = previous_text + ' ' + text_speaker_df['text'].iloc[counter]
                
                text_speaker_df.iloc[counter, text_speaker_df.columns.get_loc('start')] = new_start
                text_speaker_df.iloc[counter, text_speaker_df.columns.get_loc('text')] = new_text
                text_speaker_df.iloc[counter-1, text_speaker_df.columns.get_loc('start')] = np.nan
                text_speaker_df.iloc[counter-1, text_speaker_df.columns.get_loc('end')] = np.nan
             
        text_speaker_df = text_speaker_df.dropna(subset=['start', 'end', 'text', 'speaker'])
        text_speaker_df = text_speaker_df.sort_values('start')    
        text_speaker_df = text_speaker_df.reset_index(drop=True)
        
        return text_speaker_df

    def text_speaker_df_to_text(self):

        output_str = ''
        
        for idx, this_row in self.text_speaker_df.iterrows():
            
            this_start = np.round(this_row['start'], 2)
            this_end = np.round(this_row['end'], 2)
            this_text = this_row['text']
            this_speaker = this_row['speaker']
            
            output_str += f"{this_start} - {this_end}: {this_speaker}\n"
            output_str += f"{this_text}\n\n"

        return output_str



if __name__ == '__main__':
    my_audio2 = Audio_Diarization('freakonomics_short.wav')
    transcript2 = my_audio2.wav_to_transcript()
    print(transcript2) 
