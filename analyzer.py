import argparse
import sys
import librosa
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

def extract_and_analyze(video_path):
    # 1. Extract audio to a temporary buffer
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    
    # 2. Load and process (using the previous logic)
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    shot_times = librosa.frames_to_time(peaks, sr=sr)
    
    # 3. Build the Split Table
    data = []
    for i in range(len(shot_times)):
        shot_num = i + 1
        timestamp = shot_times[i]
        split = shot_times[i] - shot_times[i-1] if i > 0 else 0.0
        # Calculate instantaneous RPM for that specific gap
        inst_rpm = 60 / split if split > 0 else 0
        
        data.append({
            "Shot #": shot_num,
            "Timestamp (s)": round(timestamp, 3),
            "Split (s)": round(split, 3),
            "Inst. RPM": int(inst_rpm)
        })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="ShotClock AI: Command Line Shot Analyzer")
    parser.add_argument("input", help="Path to the video file")
    parser.add_argument("-s", "--sensitivity", type=float, default=0.5, help="Detection sensitivity (0.1 to 1.0)")
    parser.add_argument("-o", "--output", help="Path to save CSV results")

    args = parser.parse_args()

    print(f"--- Analyzing {args.input} ---")
    try:
        # We reuse the logic from our existing function
        df = extract_and_analyze(args.input) # You'd pass sensitivity here if updated
        print(df.to_string(index=False))
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Usage
# df = extract_and_analyze("my_range_video.mp4")
# print(df.to_string(index=False))
