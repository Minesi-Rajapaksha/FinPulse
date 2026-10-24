import sys
import subprocess
import time

def run_step(step_name, script_name):
    print(f"\n{'='*70}")
    print(f"STEP: {step_name} ({script_name})")
    print(f"{'='*70}")
    start_time = time.time()
    
    cmd = [sys.executable, script_name]
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"[FAILED] Step '{step_name}' failed with exit code {result.returncode} in {elapsed:.2f}s.")
        sys.exit(result.returncode)
    else:
        print(f"[SUCCESS] Step '{step_name}' completed successfully in {elapsed:.2f}s.")

def main():
    total_start = time.time()
    print("======================================================================")
    print("FINPULSE: FULL REPRODUCIBLE END-TO-END MASTER PIPELINE EXECUTION")
    print("======================================================================")
    
    run_step("1. Data Ingestion & Integrity", "combine_data.py")
    run_step("2. Feature Engineering Pipeline", "preprocessing.py")
    run_step("3. Exploratory Data Analysis & Econometric Diagnostics", "eda.py")
    run_step("4. Model Estimation, Tuning & Benchmark Forecasting", "models.py")
    run_step("5. In-Depth Structural Analysis (Ablation, Regimes, Shocks, DM)", "analysis.py")
    run_step("6. Report Tables & LaTeX Compilation", "build_report_tables.py")
    run_step("7. Academic Report Document Generation (MD, DOCX, PDF)", "generate_report.py")
    
    total_elapsed = time.time() - total_start
    print("\n======================================================================")
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("Generated Artifacts:")
    print("  - Figures: figures/ (fig01 through fig18 at 200 dpi)")
    print("  - Models: best_model.pkl, scaler.pkl, feature_cols.pkl, ridge_model.pkl")
    print("  - Data & Forecasts: predictions_test.csv, model_comparison.csv, model_info.json")
    print("  - Analysis CSVs: ablation_results.csv, shock_regime_results.csv, vix_regime_results.csv, top10_errors.csv, horizon_results.csv, dm_tests.csv, var_backtest.csv")
    print("  - Reports: REPORT.md, report.docx, report.pdf, report_tables.md")
    print("======================================================================")

if __name__ == "__main__":
    main()
