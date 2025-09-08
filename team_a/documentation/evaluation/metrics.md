# Evaluationsmetriken für Audio Visual Speaker Recognition (AVSR)

## 1. Individual Speaker's Word Error Rate (WER)

Grundlage für die Berechnung sind die `.vtt`-Dateien, die zu je einem Sprecher die Zeitspannen mit den jeweils gesprochenen Wörtern enthalten. Diese sollen als Output generiert werden (Hypothesis) und liegen als Ground-Truth (Reference) vor.

**Beispiel:**

```
WBVTT

00:00:00.000 --> 00:00:03.000
Hallo und willkommen zu unserem Video.

00:00:03.500 --> 00:00:06.000
In diesem Abschnitt lernen wir die Grundlagen.

00:00:06.500 --> 00:00:09.000
Viel Spaß beim Zuschauen!

```

Zur Berechnung werden folgende Schritte vorgenommen: 

### 1.1. Transkriptionen aus der Hypothesis und der Reference extrahieren

### 1.2. Zeitbereiche für die Auswertung festlegen

Die Auswertung soll nur auf den in den Session-Metadaten durch die Speaker's UEM ((Un-partitioned Evaluation Map) festgelegten Zeitbereichen stattfinden.

### 1.3. Textnormalisierung vornehmen

Für die Normalisierung sollen sowohl Sprachstörungen (Disfluencies) entfernt, als eine Standard-Textnormalisierung durchgeführt werden. 

Sprachstörungen werden folgendermaßen eliminiert (vgl. [src/tokenizer/norm_text.remove_disfluencies](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/src/tokenizer/norm_text.py#L4)): 

```python
def remove_disfluencies(text):
    list_disfluencies = set(
        ["oh","ha","um","uh","ah","hmm","haahaa","mmm","ohhh","ohh","ahh","hahaha","ohhhh","haaa","hmmm","haa","ahhh","umm","haha","mmmm","ummm","hah","hhh","ahw","hm","haahaahaa","hahahaha","hmmmm","hmmmmmmmm","aah","haaaa","uhh","hahah","hai","uhhh","ohw","ahhhh","haahaaa","hahahah","hhhh","hahahahaha","mmmmm","ummmm","aaaa","ohhhhh","sss","uuu","000","aaah","hhhhh","hmmmmm","hmmmmmmm","www","aaahh","haaaaaa","huu","ohhhhhhh","ohhhhhhhh","ohhhhhhhhhhhhhh","aaa","aahw","eee","hahahha","hh","hmmmmmm","hoo","ooo","uhhhh","uhhhhh","aaaaa","aahhh","haaaaa","haah","hahahahahahaha","hahhaha","ohhhhhh","rrr","ummmmm","uuuu","wwww","aahm","ahhhhhhhhh","er","haaaaaaa","haaaaaaaa","hahaa","hahaaaa","hahahaa","hahahahahaha","hahahahha","hahahuh","hahhah","hahhhh","hahuhu","hooo","mmmmmmm","oooo","ssssss","ummmmmmm","yah","yyyyyyyyyyyy","999","aaaahhm","aaahhh","aaahhhmmm","aahh","aahmm","ahhhhh","ahhhhhhhhhh","ahhhhhhhhhhh","eeee","ffff","haaaaaaaaa","haaaaaaaaaa","haaaaaaaaaaaaaaaaaaa","haahaha","haahahaha","haahuuuuu","hahaaa","hahaaaaa","hahaaha","hahahaaah","hahahahaahahha","hahahahah","hahahahahah","hahahahahahahaha","hahahahahha","hahahahu","hahahahuh","hahahahuhu","hahahhaa","hahahoho","hahahu","hahahuha","hahha","hahhaaha","hahhh","hahu","hahuh","hahuhahuh","hahuhuhu","haisho","hap","haummm","hhhhhh","hhhhhhh","huhahihi","huhuhuha","huuu","huuuu","huuuuu","lll","mchhh","mmmmmm","nnn","nnnnn","nnnnnn","ohahahahhu","ohhhhhhhhh","ohhhhhhhhhhh","ohhhhhhhhhhhh","ohhhhhhhhhhhhhhhhh","ohhn","ohhp","ohooo","ooooo","oooooo","ooooooooo","oooooooooooooooooooooooooo","ppppppp","ssss","sssss","uhhhhhhh","uhhhhhhhhhhhh","ummmmmmmm","ummmmmmmmm","yyy","yyyyyyy"]
    )
    refined_text = []
    for word in text.split():
        if word.lower() in list_disfluencies:
            continue
        refined_text.append(word)
    return " ".join(refined_text)
```

Für die Texnormalisierung wird der `EnglishTextNormalizer` aus der Transformer-Bibliothek verwendet (vgl. [script/evaluate.text_normalizer](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/script/evaluate.py#L13)): 

```python
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

text_normalizer = EnglishTextNormalizer({})
```

### 1.4. WER-Berechnung

Der WER ist folgendermaßen definiert (vgl. [Wikipedia - Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate)):

WER = (Substitutions + Deletions + Insertions) / Number of Words in Reference

Die Intuition hinter "Deletion" und "Insertion" ist, wie man von der Reference zur Hypothesis kommt. Wenn die Reference "This is AVSR" und die Hypothesis "This _ AVSR" ist, dann handelt es sich um eine Deletion.

Der WER berechnet sich folgendermaßen (vgl. [script/evaluate.evaluate_speaker_transcripts](https://github.com/MCoRec/mcorec_baseline/blob/0b9b1197adf182109771d623c7a537d326efff21/script/evaluate.py#L56)):

```python
def evaluate_speaker_transcripts(label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end):
    speaker_to_wer = {}
    for speaker, uem_start, uem_end in zip(speaker_list, speaker_uem_start, speaker_uem_end):
        ref_vtt = os.path.join(label_path, f"{speaker}.vtt")
        hypo_vtt = os.path.join(output_path, f"{speaker}.vtt")
        wer_score = benchmark_vtt_wer(ref_vtt, hypo_vtt, uem_start, uem_end, uem_start, uem_end)
        # print(f"WER for {speaker}: {wer_score}")
        speaker_to_wer[speaker] = round(wer_score, 4)
    return speaker_to_wer
```

In dieser Funktion wird der WER für jeden Sprecher einzeln berechnet, da ein `.vtt`-File immer nur die Sprache eines einzelnen Sprechers repräsentiert.

Ein Avarage-WER kann ermittelt werden, indem alle WER-Werte über alle Sprecher und Sessions gemittelt werden. 

## 2. Conversation Clustering Performance (Pairwise F1 Score) 

## 3. Joint ASR-Clustering Error Rate - Primary Metric