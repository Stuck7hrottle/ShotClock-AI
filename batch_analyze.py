import subprocess
import json
import pandas as pd
from pathlib import Path

def run_batch():
    test_dir = Path("test_audio")
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    summary_data = []
    
    # Get all wav files generated earlier
    wav_files = sorted(list(test_dir.glob("*.wav")))
    
    print(f"🚀 Starting batch analysis of {len(wav_files)} files...\n")
    
    for wav in wav_files:
        output_json = results_dir / f"{wav.stem}_results.json"
        
        # Construct the command. 
        # Note: We omit 'detect' since your help output showed single-command mode.
        cmd = [
            "rof", str(wav),
            "--out", str(output_json),
            "--echo-window-ms", "45"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load the result to build the summary
            with open(output_json, "r") as f:
                data = json.load(f)
                
            rof_stats = data.get("rof", {})
            summary_data.append({
                "Scenario": wav.stem,
                "Shots": data.get("rof", {}).get("n_shots", 0),
                "Avg RPM": round(rof_stats.get("mean_rpm") or 0, 1),
                "Max RPM": round(rof_stats.get("max_rpm") or 0, 1),
                "Bursts": len(data.get("bursts", [])),
                "Status": "✅ Pass" if data.get("events") else "⚠️ No Shots"
            })
            print(f"Finished: {wav.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error analyzing {wav.name}: {e.stderr}")
            summary_data.append({"Scenario": wav.stem, "Status": "FAIL"})

    # Create a nice summary table
    df = pd.DataFrame(summary_data)
    print("\n--- BATCH SUMMARY REPORT ---")
    print(df.to_string(index=False))
    
    # Save report
    df.to_csv("batch_report.csv", index=False)
    print("\nDetailed report saved to batch_report.csv")

if __name__ == "__main__":
    run_batch()
