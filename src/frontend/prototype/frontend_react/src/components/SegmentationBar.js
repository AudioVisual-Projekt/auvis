// src/components/SegmentationBar.js
import React from 'react';
import styled from 'styled-components';

const BarContainer = styled.div`
  position: relative;
  width: 100%;
  height: 20px;
  background-color: #f0f0f0;
`;

const Segment = styled.div`
  position: absolute;
  height: 100%;
  background-color: ${(props) => props.color};
`;

function SegmentationBar({ segments, totalDuration }) {
  return (
    <BarContainer>
      {segments.map((segment, index) => {
        const left = (segment.start / totalDuration) * 100;
        const width = ((segment.end - segment.start) / totalDuration) * 100;

        return (
          <Segment
            key={index}
            style={{
              left: `${left}%`,
              width: `${width}%`,
            }}
            color={segment.color}
          />
        );
      })}
    </BarContainer>
  );
}

export default SegmentationBar;