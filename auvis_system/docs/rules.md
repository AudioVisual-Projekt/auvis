---
layout: page
title: Rules
parent: CHiME-9 Task 1 - MCoRec
nav_order: 5
---

## Summary of the rules for systems participating in the challenge:


- The participants can use the development set to evaluate model performance during system development. It can be used to select the best model checkpoint, tune hyperparameters, and compare different system configurations. However, the dev set must not be used to train the model or update its internal parameters in any way.

- For system development, participants are permitted to use the MCoRec training subset, as well as the external data and pre-trained models listed in the [Data and Pre-trained Models](#external-data-and-pre-trained-models) subsection. If you believe a public dataset or model is missing from this list, you may propose its addition until **September 30, 2025**.

- For evaluation, each recording must be considered separately. The system should not be in any way fine-tuned on the entire evaluation set (e.g. by computing global statistics, gathering speaker information across multiple recordings).

- Participants must submit a system description with details about their submitted system. Publishing code is not required but encouraged.

Systems that do not comply with these rules (e.g., by using a private dataset) may still be submitted but will be excluded from the final rankings.

## Evaluation

### Overview

The system is evaluated on **three main metrics**:

1. **Individual Speaker's WER**
2. **Conversation Clustering Performance (Pairwise F1 Score)**
3. **Joint ASR-Clustering Error Rate** - *Primary Evaluation Metric*

---

### 1. Individual Speaker's WER

- **Output Required:**  
  For each speaker, the system must produce a [`.vtt`](./data.md#detailed-desciption-of-data-structure-and-formats) file containing their speech transcript, time-aligned to the video.

- **Reference:**  
  Ground-truth `.vtt` files are provided for each speaker.

- **Evaluation Steps:**
    - For each speaker:
        - Extract the reference and hypothesis transcripts from their respective `.vtt` files.
        - Restrict evaluation to the time intervals specified by the speaker's UEM (Un-partitioned Evaluation Map) in the session metadata.
        - Normalize the text (including removal of disfluencies and standard text normalization).
    - Compute the WER for each speaker:
        - WER = (Substitutions + Deletions + Insertions) / Number of words in reference
    - Average WER is calculated across all speakers across all sessions.

- **Implementation:** [script/evaluate.evaluate_speaker_transcripts](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/script/evaluate.py#L56)
---

### 2. Conversation Clustering Performance (Pairwise F1 Score)

- **Output Required:**  
  The system must output a mapping ([`speaker_to_cluster.json`](./data.md#detailed-desciption-of-data-structure-and-formats)) assigning each speaker to a conversation cluster (cluster ID).

- **Reference:**  
  Ground-truth cluster assignments are provided for each speaker.

- **Evaluation Steps:**
    - For all unordered pairs of speakers in a session:
        - Determine if the pair is in the same cluster in both the system output and the ground truth.
    - Compute the following:
        - **True Positives (TP):** Pairs correctly predicted to be in the same cluster.
        - **False Positives (FP):** Pairs predicted to be in the same cluster but are not in the ground truth.
        - **False Negatives (FN):** Pairs in the same cluster in the ground truth but not predicted as such.
    - Calculate:
        - **Precision:** TP / (TP + FP)
        - **Recall:** TP / (TP + FN)
        - **Pairwise F1 Score:** 2 * (Precision * Recall) / (Precision + Recall)
    - **Average F1 Score** is reported across all sessions.

- **Implementation:** [script/evaluate.evaluate_conversation_clustering](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/script/evaluate.py#L15)
---

### 3. Joint ASR-Clustering Error Rate - *Primary Metric*

This is the **main evaluation metric** that combines both transcription and clustering performance at the speaker level.

#### 3.1 Per-Speaker Clustering F1 Score

For each speaker, a clustering F1 score is computed using a one-vs-rest approach:

- **For speaker i:**
    - Consider all other speakers j in the same session
    - For each pair (i, j):
        - Check if they are in the same cluster in the ground truth: `true_same`
        - Check if they are in the same cluster in the prediction: `pred_same`
    - Count:
        - **TP:** Cases where `pred_same = True` and `true_same = True`
        - **FP:** Cases where `pred_same = True` and `true_same = False`
        - **FN:** Cases where `pred_same = False` and `true_same = True`
    - Compute speaker i's clustering F1:
        - **Precision:** TP / (TP + FP)
        - **Recall:** TP / (TP + FN)
        - **F1:** 2 * (Precision * Recall) / (Precision + Recall)

- **Implementation:** [script/evaluate.evaluate_speaker_clustering](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/script/evaluate.py#L22)

#### 3.2 Combined Metric Calculation

For each speaker:

```math
\text{Joint ASR-Clustering Error Rate} = 0.5 \times \text{Speaker\_WER} + 0.5 \times (1 - \text{Per\_Speaker\_Clustering\_F1})
```

This metric:
- Ranges from 0 (perfect) to 1 (worst possible)
- Equally weights transcription accuracy and clustering accuracy
- **Lower values are better**

The **final primary metric** is the average Joint ASR-Clustering Error Rate across all speakers in all sessions. 

---


## External data and pre-trained models

Besides the MCoRec dataset published with this challenge, the participants are allowed to use public datasets and pre-trained models listed below. In case you want to propose additional dataset or pre-trained model to be added to these lists, do so by contacting us at [Slack](https://join.slack.com/t/chimechallenge/shared_invite/zt-37h0cfpeb-qg5jwCgqRWCKc_3mLWVsYA) or email to mcorecchallenge@gmail.com until **September 30, 2025**. If you want to use a private dataset or model, you may still submit your system to the challenge, but we will not include it in the final rankings.

Participants may use these publicly available datasets for building the systems:


- [AVA](https://research.google.com/ava/download.html#ava_active_speaker_download)
- [Lip Reading Sentences 2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
- [AMI](https://groups.inf.ed.ac.uk/ami/corpus/)
- [LibriSpeech](https://www.openslr.org/12/)
- [TEDLIUM](https://www.openslr.org/51/)
- [MUSAN](https://www.openslr.org/17/)
- [RWCP Sound Scene Database](https://www.openslr.org/13/)
- [REVERB Challenge RIRs.](http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz)
- [Aachen AIR dataset.](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
- [BUT Reverb database.](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database)
- [SLR28 RIR and Noise Database (contains Aachen AIR, MUSAN noise, RWCP sound scene database and REVERB challenge RIRs, plus simulated ones).](https://www.openslr.org/28/)
- [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [FSD50k](https://zenodo.org/record/4060432)
- WSJ0-2mix, WHAM, [WHAMR](http://wham.whisper.ai/), WSJ
- [SINS](https://github.com/fgnt/sins)
- [LibriCSS acoustic transfer functions (ATF)](https://github.com/jsalt2020-asrdiar/jsalt2020_simulate/tree/master)
- [NOTSOFAR1 simulated CSS dataset](https://github.com/microsoft/NOTSOFAR1-Challenge#2-training-on-the-full-simulated-training-dataset)
- [Ego4D](https://ego4d-data.org/)
- [Project Aria Datasets](https://www.projectaria.com/datasets/)
- [DNS challenge noises](https://github.com/microsoft/DNS-Challenge)

In addition, following pre-trained models may be used:
- Audio-Visual Speech Recognition:
  - [AV-HuBERT](https://github.com/facebookresearch/av_hubert)
  - [MuAViC](https://github.com/facebookresearch/muavic)
    - [MuAViC Huggingface](https://huggingface.co/nguyenvulebinh/AV-HuBERT)
  - [Auto-AVSR](https://github.com/mpc001/auto_avsr)
  - [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo)
  - [(B)RAVEn](https://github.com/ahaliassos/raven)
- Wav2vec:
  - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wav2vec-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec-large)
- Wav2vec 2.0:
  - [Fairseq:](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20)
    - All models including [Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH)](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv_ftsb300_updated.pt) and the multi-lingual [XLSR-53 56k](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)
  - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html)
    - [WAV2VEC2_BASE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-base)
    - [WAV2VEC2_LARGE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large)
    - [WAV2VEC2_LARGE_LV60K](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large-lv60k)
    - [WAV2VEC2_XLSR53](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-xlsr53)
    - [WAV2VEC2_ASR_LARGE_LV60K_960H](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-large-lv60k-960h)
    - [WAV2VEC2_ASR_BASE_960](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-base-960h)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
    - [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
    - [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)
    - [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
    - [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
    - [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60)
    - [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
    - [wav2vec2-large lv60 + speaker verification](https://huggingface.co/anton-l/wav2vec2-base-superb-sv)
    - Other models on Huggingface using the same weights as the [Fairseq ones](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20).
  - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2)
    - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_base_960)
    - [wav2vec2_large_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_960)
    - [wav2vec2_large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_ll60k)
    - [wav2vec2_large_lv60_cv_swbd_fsh](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_lv60_cv_swbd_fsh)
    - [wav2vec2_conformer_relpos](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_relpos)
    - [wav2vec2_conformer_rope](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_rope)
    - [Xlsr_53](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#xlsr_53)
- [HuBERT](https://arxiv.org/abs/2106.07447)
  - [Torchaudio](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec-2-0-hubert-representation-learning)
    - [HuBERT base](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-base)
    - [HuBERT large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-large)
    - [HuBERT xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-xlarge)
    - [HuBERT ASR large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-large)
    - [HuBERT ASR xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-xlarge)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/hubert)
    - [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
    - [hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
    - [hubert-xlarge-ll60k](https://huggingface.co/facebook/hubert-xlarge-ll60k)
    - [hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
    - [hubert-xlarge-ls960-ft](https://huggingface.co/facebook/hubert-xlarge-ls960-ft)
  - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [hubert-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-base)
    - [hubert-large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-large_ll60k)
- [WavLM](https://arxiv.org/abs/2110.13900)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wavlm)
    - [wavlm-base](https://www.chimechallenge.org/current/task1/rules#:~:text=Huggingface%3A-,wavlm%2Dbase,-wavlm%2Dbase%2Dsv)
    - [wavlm-base-sv](https://huggingface.co/microsoft/wavlm-base-sv)
    - [wavlm-base-sd](https://huggingface.co/microsoft/wavlm-base-sd)
    - [wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
    - [wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
    - [wavlm-base-plus-sd](https://huggingface.co/microsoft/wavlm-base-plus-sd)
    - [wavlm-large](https://huggingface.co/microsoft/wavlm-large)
  - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wavlm-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base)
    - [wavlm-base-plus](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base-plus)
    - [wavlm-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-large)
- [Tacotron2](https://github.com/NVIDIA/tacotron2)
  - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-text-to-speech)
    - [tacotron2-wavernn-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-phone-ljspeech)
    - [tacotron2-griffinlim-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-phone-ljspeech)
    - [tacotron2-wavernn-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-char-ljspeech)
    - [tacotron2-griffinlim-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-char-ljspeech)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
  - [Speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
  - [NeMo toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [X-vector extractor](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
  - [VBx repo ResNet101 16kHz](https://github.com/BUTSpeechFIT/VBx/tree/master/VBx/models/ResNet101_16kHz)
  - [Kaldi](https://kaldi-asr.org/models/m7)
  - [Pyannote Audio](https://huggingface.co/pyannote/embedding)
- [Pyannote Segmentation](https://huggingface.co/pyannote/segmentation)
- [Pyannote Diarization (Pyannote Segmentation+ECAPA-TDNN from SpeechBrain)](https://huggingface.co/pyannote/speaker-diarization)
- [NeMo toolkit ASR pre-trained models:](https://github.com/NVIDIA/NeMo)
  - [Citrinet](https://huggingface.co/nvidia/stt_en_citrinet_1024_gamma_0_25)
  - [Conformer-CTC](https://huggingface.co/nvidia/stt_en_conformer_ctc_large)
  - [Conformer-Transducer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-transducer)
  - [FastConformer](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [NeMo toolkit speaker ID embeddings models:](https://github.com/NVIDIA/NeMo)
  - [TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#titanet)
  - [SpeakerNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet)
  - [ECAPA-TDNN](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [NeMo toolkit VAD models:](https://github.com/NVIDIA/NeMo)
  - [MarbleNet VAD](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad)
- [NeMo toolkit diarization models:](https://github.com/NVIDIA/NeMo)
  - [Multi-Scale Diarization Decoder](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/models.html#multi-scale-diarization-decoder)
- [Whisper](https://arxiv.org/abs/2212.04356)
  - [Whisper official repo (all versions: small, medium, large v1/v2/v3)](https://github.com/openai/whisper)
- [OWSM: Open Whisper-style Speech Model](https://www.wavlab.org/activities/2024/owsm/)
  - [OWSM v3 E-Branchformer](https://huggingface.co/espnet/owsm_v3.1_ebf)
  - [OWSM v3 with E-Branchformer (base, smaller version)](https://huggingface.co/espnet/owsm_v3.1_ebf_base)
  - [OWSM v3](https://huggingface.co/espnet/owsm_v3)
  - [OWSM v2](https://huggingface.co/espnet/owsm_v2)
  - [OWSM v2 E-Branchformer (note: this is probably the best OWSM for English ASR)](https://huggingface.co/espnet/owsm_v2_ebranchformer)
  - [OWSM v1](https://huggingface.co/espnet/owsm_v1)
- [Icefall Zipformer](https://huggingface.co/yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17)
- [RWKV Transducer](https://www.modelscope.cn/models/iic/speech_rwkv_transducer_asr-en-16k-gigaspeech-vocab5001-pytorch-online/summary)