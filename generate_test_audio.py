import numpy as np
from scipy.io import wavfile
from pathlib import Path

def generate_shot_impulse(duration_s=0.1, sr=48000):
    """Creates a synthetic shot impulse: noise with fast attack and exponential decay."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Fast attack (2ms), exponential decay
    envelope = np.where(t < 0.002, t / 0.002, np.exp(-(t - 0.002) / 0.015))
    noise = np.random.normal(0, 0.5, len(t))
    return (noise * envelope).astype(np.float32)

def create_test_wav(filename, shot_times, amplitude=0.8, sr=48000, noise_floor=0.01, echoes=None, interference=None):
    """
    Generates a WAV file with specific shot timings.
    echoes: list of (delay_ms, attenuation_factor)
    interference: list of (time, amplitude) for background shots
    """
    total_duration = max(shot_times) + 0.5 if shot_times else 1.0
    audio = np.random.normal(0, noise_floor, int(sr * total_duration)).astype(np.float32)
    shot_template = generate_shot_impulse(sr=sr)
    
    def add_impulse(t_sec, amp):
        start_idx = int(t_sec * sr)
        end_idx = min(start_idx + len(shot_template), len(audio))
        audio[start_idx:end_idx] += shot_template[:end_idx-start_idx] * amp

    # Add primary shots
    for t in shot_times:
        add_impulse(t, amplitude)
        # Add simulated echoes to test echo_window_ms
        if echoes:
            for delay_ms, factor in echoes:
                add_impulse(t + (delay_ms / 1000.0), amplitude * factor)

    # Add background shooter interference
    if interference:
        for t_int, amp_int in interference:
            add_impulse(t_int, amp_int)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio /= max_val
        
    wavfile.write(filename, sr, audio)
    print(f"Generated: {filename}")

# Scenarios
Path("test_audio").mkdir(exist_ok=True)

# 1. Baseline: Steady 600 RPM (0.1s splits)
create_test_wav("test_audio/01_steady_600.wav", [0.5 + i*0.1 for i in range(10)])

# 2. High Speed: 1000 RPM (0.06s splits) to test min_separation_ms
create_test_wav("test_audio/02_fast_1000.wav", [0.5 + i*0.06 for i in range(15)])

# 3. Two Bursts: Test burst_gap_ms (default 250ms)
# 5 shots, 500ms gap, 5 shots
create_test_wav("test_audio/03_two_bursts.wav", 
                [0.5 + i*0.1 for i in range(5)] + [1.5 + i*0.1 for i in range(5)])

# 4. Inconsistent Cadence: Test Cadence CV metrics
create_test_wav("test_audio/04_jittery_cadence.wav", [0.5, 0.62, 0.7, 0.85, 0.92, 1.05])

# 5. Background Interference: Primary shots vs faint background shots
bg_shots = [(0.55, 0.2), (0.65, 0.15), (0.75, 0.25)]
create_test_wav("test_audio/05_interference.wav", [0.5 + i*0.1 for i in range(10)], interference=bg_shots)

# 6. Heavy Echoes: Test echo_window_ms (default 45ms)
# Each shot followed by a strong echo at 35ms
create_test_wav("test_audio/06_echo_chamber.wav", [0.5 + i*0.15 for i in range(5)], echoes=[(35, 0.5)])

# 7. Accelerating ROF: 400 RPM to 900 RPM
accel_times = [0.5, 0.65, 0.78, 0.89, 0.98, 1.06, 1.13, 1.19]
create_test_wav("test_audio/07_accelerating.wav", accel_times)

# 8. Single Shots: Long gaps
create_test_wav("test_audio/08_slow_fire.wav", [0.5, 2.0, 3.5, 5.0])

# 9. Double-Taps: Very close pairs (40ms) to test min_separation_ms floor
double_taps = [0.5, 0.54, 1.0, 1.04, 1.5, 1.54]
create_test_wav("test_audio/09_double_taps.wav", double_taps)

# 10. Low Signal-to-Noise: Faint shots in loud background hiss
create_test_wav("test_audio/10_noisy_env.wav", [0.5 + i*0.1 for i in range(10)], amplitude=0.15, noise_floor=0.08)

# 11. Multi-Burst: 3 distinct strings
multi = [0.5 + i*0.1 for i in range(3)] + [1.5 + i*0.1 for i in range(3)] + [2.5 + i*0.1 for i in range(3)]
create_test_wav("test_audio/11_multi_burst.wav", multi)

# 12. Extreme ROF: 1200 RPM (0.05s splits) - edge case for most detectors
create_test_wav("test_audio/12_extreme_1200.wav", [0.5 + i*0.05 for i in range(20)])
