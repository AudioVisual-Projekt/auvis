import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QLabel
)
from PySide6.QtCore import Qt, QUrl, QTime, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class VideoPlayer(QWidget):

    def __init__(self):
        super().__init__()


        self.setWindowTitle("Videoplayer mit Transkription")
        self.setGeometry(200, 200, 800, 600)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        self.open_button = QPushButton("Videodatei öffnen")
        self.transcribe_button = QPushButton("Transkribieren")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")

        self.time_label = QLabel("00:00")
        self.duration_label = QLabel("00:00")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.transcribe_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.time_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.duration_label)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addLayout(slider_layout)
        layout.addLayout(control_layout)
        self.setLayout(layout)

        self.open_button.clicked.connect(self.open_file)
        self.transcribe_button.clicked.connect(self.transcribe_video)
        self.play_button.clicked.connect(self.media_player.play)
        self.pause_button.clicked.connect(self.media_player.pause)
        
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.slider.sliderMoved.connect(self.set_position)

    def open_file(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Videodatei öffnen", "", "Video Files (*.mp4 *.avi *.mkv, *.mov)")
        if file_path:
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.play()

    def update_position(self, position):
        self.slider.setValue(position)
        self.time_label.setText(self.format_time(position))

    def update_duration(self, duration):
        self.slider.setRange(0, duration)
        self.duration_label.setText(self.format_time(duration))

    def set_position(self, position):
        self.media_player.setPosition(position)

    def format_time(self, ms):
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    def transcribe_video(self):
        print("Transkription gestartet (Platzhalter)")
        #TODO:

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())