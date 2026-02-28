- der Großteil des Codes in ASD_Clustering wurde im stetigen Austausch mit Googles Gemini erzeugt

- pipeline_start.ipynb dient als Start um einen oder mehrere Durchläufe des ASD_Clusterings durchzuführen
- alternativ kann auch direkt die unter /script befindliche main.py ausgeführt werden
- es sind überall die Segmentierungs- und Clustering-Parameter hinterlegt, die zum höchsten "per speaker F1-Score" geführt haben

- die zu Grunde liegenden Datensätze sind in /data-bin zu finden
- dabei liegt jeweils ein Trainings- und ein Dev-Datensatz vor
- eine Ausnahme bildet der Evaluation-Datensatz, den wir kurz vor Abgabe erhalten haben (ohne entsprechende Lösung)

- im Unterordner /notebooks sind weitere Notebooks zu finden, die zur Analyse herangezogen werden können
- die besten "per speaker F1-Scores" und die dazugehörigen Parameter sind in /notebooks/data zu finden
- ebenso die Parametersensitivitätsanalyse
