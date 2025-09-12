---
layout: page
title: Submission
parent: CHiME-9 Task 1 - MCoRec
nav_order: 6
---

The submission of the systems is open until **TBU** and should be done through [Google Form - TBU](#). We allow each team to submit up to **three systems** for the challenge. For the submission, make sure to have the following ready:

- Technical description paper
- System outputs for development and evaluation subset

## Technical description paper

For the technical description, follow the instructions for CHiME-9 challenge papers at the [workshop page](TBU). The papers should be 4 pages long with one additional page for references. Please describe all of your submitted systems and the results on development subset. Please submit your abstract before your results and note the CMT paper ID assigned to your abstract as you will need to include it in your Google form submission.

## System outputs

Participants should submit a zip file containing the output files for each submitted system. The zip file should contain the following directory structure:

    ├── name_of_system_1
    │   ├── dev
    │   │   ├── session_id_1
    │   │   │   ├── speaker_to_cluster.json
    │   │   │   ├── spk_0.vtt
    │   │   │   ├── spk_1.vtt
    │   │   │   └── ...
    │   │   ├── session_id_2
    │   │   │   ├── speaker_to_cluster.json
    │   │   │   ├── spk_0.vtt
    │   │   │   ├── spk_1.vtt
    │   │   │   └── ...
    │   │   └── ...
    │   └── eval
    │       ├── session_id_1
    │       │   ├── speaker_to_cluster.json
    │       │   ├── spk_0.vtt
    │       │   ├── spk_1.vtt
    │       │   └── ...
    │       ├── session_id_2
    │       └── ...
    ...
    └── name_of_system_N
        ├── dev
        └── eval

- Feel free to choose any naming of the systems, but please make sure that they are consistent between all submitted archives.

Each session directory contains:

- `speaker_to_cluster.json`: Contains the conversation clustering assignments for all speakers in that session, following the same format as in the dataset labels
- `spk_0.vtt`, `spk_1.vtt`, etc.: WebVTT files containing time-aligned transcriptions for each target speaker, following the same format as the dataset labels

The file formats for system outputs should follow the same structure and format as described in the [Detailed description of data structure and formats](./data.md#detailed-desciption-of-data-structure-and-formats) section of the data documentation.

## Important Notes

- **Evaluation Metrics**: The primary ranking metric is the **Joint ASR-Clustering Error Rate**, which equally weights transcription accuracy (WER) and clustering accuracy (per-speaker clustering F1)
- **Clustering Requirements**: Each speaker must be assigned to exactly one conversation cluster per session. Cluster IDs can be any integer values but must be consistent within each session
- **Text Normalization**: The evaluation script will automatically normalize text and remove disfluencies before computing WER
- **Data Usage Compliance**: Systems must comply with the [challenge rules](./rules.md). Only approved external datasets and pre-trained models may be used
- **Processing Independence**: Each evaluation recording must be processed independently. The development set cannot be used for training or parameter updates

If you are unsure about how to provide any of the above files, please contact us at the [Slack](https://join.slack.com/t/chimechallenge/shared_invite/zt-37h0cfpeb-qg5jwCgqRWCKc_3mLWVsYA) channel or at [mcorecchallenge@gmail.com](mailto:mcorecchallenge@gmail.com).
