# CHiME-MCoRec Subtask 4 – WER-Optimierung

Dieses Repository enthält alle Experiment-Notebooks der Projektgruppe für **CHiME-MCoRec Subtask 4** (Speaker-attributed ASR). Ziel war es, die Word Error Rate (WER) gegenüber der Challenge-Baseline zu minimieren.

---

## Ergebnis auf einen Blick

| | Speaker WER |
|---|---|
| BL4-Baseline (beam=3, len=15) | 0.4987 |
| **Finale Konfiguration (25 Dev-Sessions)** | **0.4943** |
| Verbesserung | −0.0044 (−0.88%) |

Finale Konfiguration: `beam_size=12`, `max_length=20s`, `min_duration_on=1.0s`, `min_duration_off=1.2s`, Modell BL4.

---

## Übersicht der Notebooks

### Phase 1 – Baseline

**`01_avsr_baseline_eval`**
Evaluation der vier Challenge-Baselines (BL1–BL4). BL4 (AV-HuBERT Cocktail + MCoRec-Finetuning) ist mit WER 0.4987 das beste Modell und Ausgangspunkt aller weiteren Experimente.

**`01a_bugfix_avsr_baseline`**
Dokumentation des Segmentierungs-Bugfixes in `segmentation.py`: `min_duration_off` las fälschlicherweise den Fallback-Wert von `min_duration_on`.

---

### Phase 2 – Experimente auf 5-Session-Dev-Subset

#### Decoding-Hyperparameter

**`02_experiments_beam_size_max_length`**
Grid-Search über `beam_size ∈ {4, 8, 12}` × `max_length ∈ {10, 15, 18, 20}s` auf BL4.
→ Beste Kombination: **beam=12, len=20 → WER 0.4954** (−0.0033 vs. BL4-Default)

#### Fine-Tuning

**`02a_experiements_finetuning`**
Stage-2-Fine-Tuning von BL4 (30k Schritte, LR 5·10⁻⁵).
→ WER +0.036 vs. BL4 – Überanpassung, schlechter als Ausgangspunkt

**`02b_experiements_finetuning2`**
Stage-2-Light-Fine-Tuning (5k Schritte, LR 1·10⁻⁵, MCoRec-Fokus 70%).
→ WER +0.032 vs. BL4 – minimale Verbesserung zu 02a, BL4 weiterhin besser

#### Modell-Vergleiche

**`02c_experiements_model`**
Whisper Large-v3 (Audio-only) vs. BL4.
→ WER ~1.03 – ca. 108% schlechter als BL4 (kein visueller Input, kein MCoRec-FT)

**`02d_experiements_model_whisper_flamingo`**
Whisper-Flamingo (audiovisuell, kein MCoRec-FT).
→ WER ~0.92 – ca. 86% schlechter als BL4

**`02e_experiements_model_whisper_flamingo_finetune`**
Whisper-Flamingo mit MCoRec-Fine-Tuning (2k Schritte, nur Decoder).
→ WER ~7–9 – drastische Verschlechterung durch Catastrophic Forgetting

#### LLM-Postprocessing

**`02f_experiements_llm`**
SEP-Block-Prompting mit Token- und WER-Guard. Modelle: Qwen3-8B, Qwen2.5-7B, Qwen2.5-Coder, DeepSeek-R1.
→ Beste Variante E38 (Qwen3-8B): WER 0.4948 (−0.0006) – statistisch nicht signifikant

**`02g_experiements_llm_qwen_v2`**
Verfeinerte Pipeline für Qwen3-8B, kein WER-Guard.
→ WER 0.5235 (+0.028) – ohne Guard deutliche Verschlechterung

**`02h_experiements_llm_coder`**
Regelbasierter Ansatz mit Qwen2.5-Coder, Zeile-für-Zeile, Sampling.
→ WER 0.5467 (+0.051) – schlechtestes LLM-Experiment

**`02i_experiments_llm_qwen_v3`**
VTT-nativer Ansatz: Qwen3-8B erhält die gesamte VTT-Datei + Chain-of-Thought.
→ Keine WER-Verbesserung auf 5 Sessions; Volltest in `03a2_`

#### Post-Bugfix-Experimente

**`02j_experiments_bugfix_beam_size_max_length`**
Decoding-Grid (beam × max_length) nach Bugfix. Der Bugfix allein verschlechtert die WER.
→ beam=12, len=20 → WER 0.5245 (+0.029 vs. Pre-Bugfix)

**`02k_experiments_bugfix_minduration`**
Grid-Search über `min_duration_on ∈ {0.4, 0.6, 0.8, 1.0}s` × `min_duration_off ∈ {0.5, 0.8, 1.0, 1.2}s` (16 Kombinationen).
→ **E72 (min_on=1.0, min_off=1.2): WER 0.4902** – übertrifft Pre-Bugfix-Ergebnis

---

### Phase 3 – Vollläufe auf 25 Dev-Sessions

**`03_results`**
beam=12, len=20 auf allen 25 Sessions (Pre-Bugfix).
→ WER 0.4980

**`03a_results_postprocessing_llm`**
E38-LLM-Konfiguration (Qwen3-8B, SEP-Block) auf allen 25 Sessions.
→ WER 0.4987 (+0.0007) – keine Verbesserung

**`03a2_results_postprocessing_llm_v3`**
VTT-nativer LLM-Ansatz (`02i_`) auf allen 25 Sessions.
→ WER 0.4996 (+0.0016) – leichte Verschlechterung, LLM-Postprocessing endgültig verworfen

**`03b_results_bugfix`**
beam=12, len=20 nach Bugfix auf allen 25 Sessions. Sanity-Check: Bugfix allein reicht nicht.
→ WER verschlechtert sich; motiviert Grid-Search in `02k_`

---

### Phase 4 – Abschlussläufe

**`04_dev_final_results`**
Finale Konfiguration (E72: min_on=1.0, min_off=1.2, beam=12, len=20, BL4) auf allen 25 Dev-Sessions.
→ **WER 0.4943** (−0.0044 vs. BL4-Baseline)

**`04_eval_final_results`**
Gleiche Konfiguration auf dem ungesehenen Eval-Set. OOM-Handling über 4 Resume-Durchläufe mit `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
→ VTT-Dateien eingereicht; WER-Evaluation durch Organisatoren ausstehend

---

## Experimenteller Ablauf

```
01_    Baseline-Evaluation (BL4 als Referenz)
 │
02_    Hyperparameter-Suche → beam=12, len=20 (beste Decoding-Konfig)
02a/b  Fine-Tuning → kein Nutzen
02c–e  Modell-Vergleiche → BL4 klar überlegen
02f–i  LLM-Postprocessing → kein robuster Nutzen
 │
01a_   Bugfix entdeckt
02j_   Post-Bugfix-Baseline → Bugfix allein verschlechtert WER
02k_   min_duration-Grid-Search → E72 als optimale Konfiguration
 │
03_    Validierung beam=12/len=20 auf 25 Sessions (Pre-Bugfix)
03b_   Sanity-Check: Bugfix-Effekt auf 25 Sessions
 │
04_dev_   Finaler Dev-Lauf → WER 0.4943
04_eval_  Einreichungs-Lauf auf Eval-Set
```

---

## Baseline-Modelle

| Modell | WER |
|--------|-----|
| BL1 – AV-HuBERT Cocktail | 0.5538 |
| BL2 – MuAViC-EN | 0.7178 |
| BL3 – Auto-AVSR | 0.8316 |
| **BL4 – AV-HuBERT Cocktail + MCoRec-FT** | **0.4987** |

---

## Hinweis zum Segmentierungs-Bugfix

In `segmentation.py` las `min_duration_off_frames` fälschlicherweise den Fallback-Wert von `min_duration_on` statt `min_duration_off`. Der Bugfix allein verschlechterte die WER zunächst (+0.029), da die Segmentierung auf den fehlerhaften Wert kalibriert war. Erst durch den Grid-Search in `02k_` konnte das Optimum (min_on=1.0s, min_off=1.2s) gefunden werden, das die Baseline schließlich übertrifft.
