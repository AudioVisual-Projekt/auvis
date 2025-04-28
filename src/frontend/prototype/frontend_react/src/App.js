// src/App.js
import React from 'react';
import styled from 'styled-components';
import VideoScreenshots from './components/VideoScreenshots';
import AudioWaveform from './components/AudioWaveform';
import SegmentationBar from './components/SegmentationBar';
import './App.css'; // Behalten Sie Ihre bestehende CSS-Datei

const AppContainer = styled.div`
  padding: 20px;
  font-family: Arial, sans-serif;
`;

const Title = styled.h1`
  text-align: center;
`;

const Section = styled.div`
  margin-bottom: 20px;
`;

function App() {
  const screenshots = [
    'assets/screenshot1.jpg',
    'assets/screenshot2.jpg',
    'assets/screenshot3.jpg',
    'assets/screenshot4.jpg',
  ];

  const audioUrl = 'assets/audio.mp3';

  // Beispiel-Daten für die Segmentierung (Zeitabschnitte in Sekunden)
  const sdResult = [
    { start: 0, end: 5, speaker: 1, color: 'red' },
    { start: 5, end: 10, speaker: 2, color: 'blue' },
    { start: 10, end: 15, speaker: 3, color: 'green' },
    { start: 15, end: 20, speaker: 4, color: 'yellow' },
  ];

  const groundTruth = [
    { start: 0, end: 6, speaker: 1, color: 'red' },
    { start: 6, end: 9, speaker: 2, color: 'blue' },
    { start: 9, end: 16, speaker: 3, color: 'green' },
    { start: 16, end: 20, speaker: 4, color: 'yellow' },
  ];

  return (
    <AppContainer>
      <Title>Audiovisuelle Gesprächsanalyse</Title>

      <Section>
        <VideoScreenshots screenshots={screenshots} />
      </Section>

      <Section>
        <AudioWaveform audioUrl={audioUrl} />
      </Section>

      <Section>
        <h3>SD Result</h3>
        <SegmentationBar segments={sdResult} totalDuration={20} />
      </Section>

      <Section>
        <h3>Ground Truth</h3>
        <SegmentationBar segments={groundTruth} totalDuration={20} />
      </Section>
    </AppContainer>
  );
}

export default App;