// src/components/VideoScreenshots.js
import React from 'react';
import styled from 'styled-components';

const ScreenshotContainer = styled.div`
  display: flex;
  gap: 10px;
  justify-content: center;
`;

const Screenshot = styled.img`
  width: 150px;
  height: 100px;
  object-fit: cover;
  border: 1px solid #ccc;
`;

function VideoScreenshots({ screenshots }) {
  return (
    <ScreenshotContainer>
      {screenshots.map((src, index) => (
        <Screenshot key={index} src={require(`../assets/screenshot${index+1}.jpg`)} alt={`Screenshot ${index + 1}`} />
      ))}
    </ScreenshotContainer>
    //<img src={ require('../assets/screenshot1.jpg') } />
  );
}

export default VideoScreenshots;