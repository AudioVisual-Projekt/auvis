# Ergebnisse (results/)

Dieser Ordner enthält alle CSV-Dateien die im Rahmen der Subtask-4-Experimente erzeugt wurden.
Die Werte beziehen sich auf den Dev-Datensatz (25 Sessions, sofern nicht anders angegeben).

---

## Baseline

**`results_baseline_dev_without_central_videos.csv`**
Evaluationsergebnisse der vier Challenge-Baselines (BL1–BL4) mit Standardparametern.
→ Referenz für alle Experimente. BL4 (`avsr_cocktail_finetuned`): WER 0.4987

---

## Experimente auf dem 5-Session-Subset

**`results_dev_subset_by_session.csv`**
Alle 55 Experimente E01–E55, ausgeführt auf einem Subset von 5 Dev-Sessions.
Enthält: Decoding-Grid (E01–E10), Fine-Tuning (E11–E15), Modell-Vergleiche Whisper/Flamingo (E16–E37), LLM-Postprocessing (E38–E43, E55), Post-Bugfix-Decoding-Grid (E44–E54).
→ Basis für Notebooks `02_` bis `02i_` sowie `02j_`

**`results_grid_segmentation_by_session.csv`**
Grid-Search über `min_duration_on` × `min_duration_off` nach Bugfix, 17 Konfigurationen E56–E72 auf 5 Sessions.
→ E72 (min_on=1.0s, min_off=1.2s): WER 0.4902 – beste Konfiguration
→ Basis für Notebook `02k_`

**`results_grid_segmentation_summary.csv`**
Zusammenfassung des Grid-Search (ein Wert pro Konfiguration, gemittelt über 5 Sessions).

---

## Durchläufe auf allen 25 Dev-Sessions

**`final_results_by_session.csv`**
LLM-Postprocessing E38 (Qwen3-8B, SEP-Block) auf allen 25 Sessions.
→ WER 0.4987 (+0.0007 vs. Baseline) – keine Verbesserung
→ Basis für Notebook `03a_`

**`final_results_llm_qwen3_v3_by_session.csv`**
LLM-Postprocessing v3 (VTT-nativ, Qwen3-8B) auf allen 25 Sessions.
→ WER 0.4996 (+0.0016 vs. Baseline) – leichte Verschlechterung
→ Basis für Notebook `03a2_`

**`results_DEV_final_bugfix_mdOn1p0_mdOff1p2_bs12_len20_by_session.csv`**
Finale Konfiguration (E72: min_on=1.0s, min_off=1.2s, beam=12, len=20) auf allen 25 Sessions, session-basiert.
→ WER 0.4943 (−0.88% vs. BL4-Baseline)
→ Basis für Notebook `04_dev_`

**`results_DEV_final_bugfix_mdOn1p0_mdOff1p2_bs12_len20_summary.csv`**
Zusammenfassung der finalen Konfiguration (ein Wert, gemittelt über alle 25 Sessions).
