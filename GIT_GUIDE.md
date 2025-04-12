# Git-Test & Kurzanleitung für das AUVIS-Team

**Ziel:** Diese Datei zeigt, wie ihr Änderungen im Repository vornehmen, committen und pushen könnt – Schritt für Schritt. Da ich Erol Celik ein Noob bin und sicher häufiger in diese Datei schauen muss :D :D :D 

Stand: Erste Version erstellt von Erol und somit auch erster commit/push Test. 

# Git-Anleitung für das AUVIS-Team

## Teil 1: Git auf dem eigenen Rechner einrichten

### Voraussetzung: Visual Studio Code (VSCode) ist bereits installiert

### Schritt 1: Git installieren
1. Rufe die Website https://git-scm.com/downloads auf.
2. Klicke auf "Download for Windows" (bzw. dein Betriebssystem).
3. Führe die heruntergeladene Datei aus.
4. Klicke dich durch den Installer – **du kannst alle vorgeschlagenen Einstellungen einfach übernehmen**.
5. Nach der Installation: 
   - Öffne Visual Studio Code.
   - Öffne das Terminal.
   - Gib folgenden Befehl ein, um zu testen, ob Git richtig installiert wurde:
     ```
     git --version
     ```
     Wenn eine Versionsnummer erscheint (z. B. `git version 2.43.0`), ist alles korrekt installiert.

### Schritt 2: Git-Konfiguration durchführen
Diese Daten werden bei deinen Änderungen im Projekt als "Autor" angezeigt.

1. Gib im Terminal ein:
   ```
   git config --global user.name "Dein Vorname Nachname"
   ```
2. Dann:
   ```
   git config --global user.email "deine.mail@beispiel.de"
   ```
3. Zum Prüfen:
   ```
   git config --list
   ```
   Du solltest nun deinen Namen und deine E-Mail-Adresse sehen.

## Teil 2: Arbeiten mit dem Repository

### Schritt 1: Repository vom Server auf den eigenen Rechner holen (Klonen)

1. Gehe auf die GitHub-Seite des Projekts.
2. Klicke auf den grünen Button **„Code“** und kopiere den **HTTPS-Link**.
3. Öffne das Terminal in VSCode.
4. Wechsle in den Ordner, in dem du das Projekt speichern möchtest, z. B.:
   ```
   cd C:/Users/DeinName/Dokumente/Uni
   ```
5. Gib folgenden Befehl ein:
   ```
   git clone https://github.com/euer-team/auvis.git
   ```
6. Wechsle in den neuen Projektordner:
   ```
   cd auvis
   ```
7. Öffne das Projekt in VSCode:
   ```
   code .
   ```

### Schritt 2: Virtuelle Umgebung einrichten (venv)

1. Gib im Terminal ein:
   ```
   python -m venv venv
   ```
   → Dadurch wird ein Ordner namens `venv` erstellt.

2. Aktiviere die virtuelle Umgebung:

   **Windows:**
   ```
   .\venv\Scripts\activate
   ```
   **macOS/Linux:**
   ```
   source venv/bin/activate
   ```

3. Wenn es funktioniert, steht im Terminal vorne `(venv)`.

### Schritt 3: Projektdateien vorbereiten (nur wenn vorhanden)
Falls es eine Datei `requirements.txt` gibt, installiere die Abhängigkeiten mit:
``` 
pip install -r requirements.txt
```

### Schritt 4: Änderungen machen, speichern und hochladen

#### 1. Eine Datei ändern oder neu erstellen
- Beispiel: Lege im Projekt eine Datei `git_test.txt` an oder öffne eine bestehende Datei z.B. diese Anleitung.
- Schreibe eine kurze Änderung hinein (z. B. deinen Namen oder einen Kommentar).
- Speichern mit `Strg + S`.

#### 2. Änderungen zum Speichern vormerken (stagen)
Im Terminal eingeben:
```
git add DATEINAME
```
Beispiel:
```
git add git_test.txt
```

#### 3. Änderung als neuen Stand festhalten (committen)
Gib im Terminal ein:
```
git commit -m "Kurze Beschreibung der Änderung"
```
Beispiel:
```
git commit -m "Testdatei hinzugefügt"
```

#### 4. Änderungen ins zentrale Repository hochladen (pushen)
Im Terminal:
```
git push origin main
```
(Wenn ihr einen anderen Branch nutzt, z. B. `develop`, dann entsprechend `git push origin develop` schreiben.)

### Schritt 5: Änderungen anderer Teammitglieder holen (pullen)
Bevor du mit der Arbeit beginnst, solltest du die aktuellste Version des Projekts holen:
```
git pull origin main
```
Auch hier ggf. den Branchnamen anpassen.


