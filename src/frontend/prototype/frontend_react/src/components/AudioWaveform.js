// src/components/AudioWaveform.js
import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';

function AudioWaveform({ audioUrl }) {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);

  useEffect(() => {
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'blue',
      progressColor: 'red',
      height: 100,
    });

    wavesurfer.current.load(audioUrl);

    return () => wavesurfer.current.destroy();
  }, [audioUrl]);

  return <div ref={waveformRef} />;
}

export default AudioWaveform;