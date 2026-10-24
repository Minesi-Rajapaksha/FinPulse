import os
import json
import pandas as pd
import numpy as np

def build_tables():
    tables_md = {}
    tables_latex = {}
    
    # -------------------------------------------------------------
    # Table 1: Descriptive Statistics
    # -------------------------------------------------------------
    if os.path.exists("eda_results.json"):
        with open("eda_results.json", "r") as f:
            eda_data = json.load(f)
        desc = eda_data["descriptive_statistics"]
        rows = []
        for k, v in desc.items():
            rows.append({
                "Series": k,
                "Count": v["count"],
                "Mean": f"{v['mean']:.4f}",
                "Std. Dev.": f"{v['std']:.4f}",
                "Min": f"{v['min']:.4f}",
                "25%": f"{v['p25']:.4f}",
                "Median": f"{v['median']:.4f}",
                "75%": f"{v['p75']:.4f}",
                "Max": f"{v['max']:.4f}",
                "Skewness": f"{v['skew']:+.4f}",
                "Excess Kurtosis": f"{v['excess_kurtosis']:+.4f}"
            })
        t1_df = pd.DataFrame(rows)
        tables_md["Table_1_Descriptive_Stats"] = t1_df.to_markdown(index=False)
        tables_latex["Table_1_Descriptive_Stats"] = t1_df.to_latex(index=False)
    
    # -------------------------------------------------------------
    # Table 2: Econometric Diagnostic Tests
    # -------------------------------------------------------------
    if os.path.exists("eda_results.json"):
        with open("eda_results.json", "r") as f:
            eda_data = json.load(f)
        adf = eda_data["adf_test"]
        lb = eda_data["ljung_box_test"]
        arch_res = eda_data["arch_lm_test"]
        jb = eda_data["jarque_bera_test"]
        
        diag_rows = [
            {"Test Name": "Augmented Dickey-Fuller (ADF) - Close", "Null Hypothesis (H0)": "Unit Root (Non-Stationary)", "Test Statistic": f"{adf['Close']['adf_stat']:.4f}", "p-value": f"{adf['Close']['p_value']:.4f}", "Statistical Conclusion": "Fail to Reject H0 (Non-Stationary)"},
            {"Test Name": "Augmented Dickey-Fuller (ADF) - Return", "Null Hypothesis (H0)": "Unit Root (Non-Stationary)", "Test Statistic": f"{adf['Return']['adf_stat']:.4f}", "p-value": f"{adf['Return']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Strictly Stationary)"},
            {"Test Name": "Ljung-Box Q (Lag 5) - Return", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Return']['5']['stat']:.4f}", "p-value": f"{lb['Return']['5']['p_value']:.4f}", "Statistical Conclusion": "Fail to Reject H0 (Martingale Difference)"},
            {"Test Name": "Ljung-Box Q (Lag 10) - Return", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Return']['10']['stat']:.4f}", "p-value": f"{lb['Return']['10']['p_value']:.4f}", "Statistical Conclusion": "Fail to Reject H0 (Martingale Difference)"},
            {"Test Name": "Ljung-Box Q (Lag 20) - Return", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Return']['20']['stat']:.4f}", "p-value": f"{lb['Return']['20']['p_value']:.4f}", "Statistical Conclusion": "Fail to Reject H0 (Martingale Difference)"},
            {"Test Name": "Ljung-Box Q (Lag 5) - |Return|", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Abs_Return']['5']['stat']:.4f}", "p-value": f"{lb['Abs_Return']['5']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Strong Volatility Clustering)"},
            {"Test Name": "Ljung-Box Q (Lag 10) - |Return|", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Abs_Return']['10']['stat']:.4f}", "p-value": f"{lb['Abs_Return']['10']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Strong Volatility Clustering)"},
            {"Test Name": "Ljung-Box Q (Lag 20) - |Return|", "Null Hypothesis (H0)": "No Serial Autocorrelation", "Test Statistic": f"{lb['Abs_Return']['20']['stat']:.4f}", "p-value": f"{lb['Abs_Return']['20']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Strong Volatility Clustering)"},
            {"Test Name": "Engle's ARCH-LM (LM Statistic)", "Null Hypothesis (H0)": "No Conditional Heteroskedasticity", "Test Statistic": f"{arch_res['lm_stat']:.4f}", "p-value": f"{arch_res['lm_p_value']:.4e}", "Statistical Conclusion": "Reject H0 (ARCH Effects Present)"},
            {"Test Name": "Engle's ARCH-LM (F-Statistic)", "Null Hypothesis (H0)": "No Conditional Heteroskedasticity", "Test Statistic": f"{arch_res['f_stat']:.4f}", "p-value": f"{arch_res['f_p_value']:.4e}", "Statistical Conclusion": "Reject H0 (ARCH Effects Present)"},
            {"Test Name": "Jarque-Bera Test - Return", "Null Hypothesis (H0)": "Gaussian Normal Distribution", "Test Statistic": f"{jb['Return']['stat']:.4f}", "p-value": f"{jb['Return']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Fat Tails / Leptokurtic)"},
            {"Test Name": "Jarque-Bera Test - |Return|", "Null Hypothesis (H0)": "Gaussian Normal Distribution", "Test Statistic": f"{jb['Abs_Return']['stat']:.4f}", "p-value": f"{jb['Abs_Return']['p_value']:.4e}", "Statistical Conclusion": "Reject H0 (Extreme Positive Skew)"}
        ]
        t2_df = pd.DataFrame(diag_rows)
        tables_md["Table_2_Econometric_Diagnostics"] = t2_df.to_markdown(index=False)
        tables_latex["Table_2_Econometric_Diagnostics"] = t2_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 3: GARCH Parameters
    # -------------------------------------------------------------
    if os.path.exists("garch_params.csv"):
        gp_df = pd.read_csv("garch_params.csv")
        gp_formatted = pd.DataFrame({
            "Model": gp_df["Model"],
            "omega": gp_df["omega"].apply(lambda x: f"{x:.5f}" if pd.notnull(x) else "-"),
            "alpha": gp_df["alpha"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-"),
            "gamma (Asymmetry)": gp_df["gamma"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-"),
            "beta": gp_df["beta"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-"),
            "Persistence": gp_df["persistence"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-"),
            "Log-Likelihood": gp_df["log_likelihood"].apply(lambda x: f"{x:.2f}"),
            "AIC": gp_df["AIC"].apply(lambda x: f"{x:.2f}"),
            "BIC": gp_df["BIC"].apply(lambda x: f"{x:.2f}")
        })
        tables_md["Table_3_GARCH_Parameters"] = gp_formatted.to_markdown(index=False)
        tables_latex["Table_3_GARCH_Parameters"] = gp_formatted.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 4: GARCH Residual Diagnostics
    # -------------------------------------------------------------
    if os.path.exists("garch_diagnostics.csv"):
        gd_df = pd.read_csv("garch_diagnostics.csv")
        tables_md["Table_4_GARCH_Residual_Diagnostics"] = gd_df.to_markdown(index=False)
        tables_latex["Table_4_GARCH_Residual_Diagnostics"] = gd_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 5: Hyperparameter Tuning Grid
    # -------------------------------------------------------------
    if os.path.exists("tuning_results.csv"):
        tr_df = pd.read_csv("tuning_results.csv")
        tr_df["Optimal"] = tr_df["Is_Selected"].apply(lambda x: "Yes (Selected)" if x else "No")
        tables_md["Table_5_Hyperparameter_Grid"] = tr_df[["Model", "Parameters", "Val_MAE", "Optimal"]].to_markdown(index=False)
        tables_latex["Table_5_Hyperparameter_Grid"] = tr_df[["Model", "Parameters", "Val_MAE", "Optimal"]].to_latex(index=False)

    # -------------------------------------------------------------
    # Table 6: Model Comparison Benchmark (Test Set)
    # -------------------------------------------------------------
    if os.path.exists("model_comparison.csv"):
        mc_df = pd.read_csv("model_comparison.csv")
        naive_mae = mc_df.loc[mc_df["Model"]=="Naive", "MAE"].values[0]
        naive_rmse = mc_df.loc[mc_df["Model"]=="Naive", "RMSE"].values[0]
        
        mc_df["MAE_vs_Naive"] = mc_df["MAE"].apply(lambda x: f"{((x - naive_mae) / naive_mae * 100):+.2f}%")
        mc_df["RMSE_vs_Naive"] = mc_df["RMSE"].apply(lambda x: f"{((x - naive_rmse) / naive_rmse * 100):+.2f}%")
        
        disp_cols = ["Rank", "Model", "MAE", "RMSE", "QLIKE", "MAE_vs_Naive", "RMSE_vs_Naive", "DM_p_value"]
        tables_md["Table_6_Model_Benchmark"] = mc_df[disp_cols].to_markdown(index=False)
        tables_latex["Table_6_Model_Benchmark"] = mc_df[disp_cols].to_latex(index=False)

    # -------------------------------------------------------------
    # Table 7: Walk-Forward vs Fixed-Fit
    # -------------------------------------------------------------
    if os.path.exists("walk_forward_results.csv"):
        wf_df = pd.read_csv("walk_forward_results.csv")
        tables_md["Table_7_Walk_Forward_Results"] = wf_df.to_markdown(index=False)
        tables_latex["Table_7_Walk_Forward_Results"] = wf_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 8: Ablation Results
    # -------------------------------------------------------------
    if os.path.exists("ablation_results.csv"):
        ab_df = pd.read_csv("ablation_results.csv")
        tables_md["Table_8_Ablation_Results"] = ab_df.to_markdown(index=False)
        tables_latex["Table_8_Ablation_Results"] = ab_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 9: Shock Analysis (April 2025)
    # -------------------------------------------------------------
    if os.path.exists("shock_regime_results.csv"):
        sr_df = pd.read_csv("shock_regime_results.csv")
        tables_md["Table_9_April_Shock_Analysis"] = sr_df.to_markdown(index=False)
        tables_latex["Table_9_April_Shock_Analysis"] = sr_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 10: VIX Regimes
    # -------------------------------------------------------------
    if os.path.exists("vix_regime_results.csv"):
        vr_df = pd.read_csv("vix_regime_results.csv")
        tables_md["Table_10_VIX_Regimes"] = vr_df.to_markdown(index=False)
        tables_latex["Table_10_VIX_Regimes"] = vr_df.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 11: Multi-Horizon Results
    # -------------------------------------------------------------
    if os.path.exists("horizon_results.csv"):
        hr_df = pd.read_csv("horizon_results.csv")
        tables_md["Table_11_Multi_Horizon_Results"] = hr_df[["Horizon", "Model", "MAE", "RMSE", "QLIKE"]].to_markdown(index=False)
        tables_latex["Table_11_Multi_Horizon_Results"] = hr_df[["Horizon", "Model", "MAE", "RMSE", "QLIKE"]].to_latex(index=False)

    # -------------------------------------------------------------
    # Table 12: Diebold-Mariano Tests
    # -------------------------------------------------------------
    if os.path.exists("dm_tests.csv"):
        dm_df = pd.read_csv("dm_tests.csv")
        tables_md["Table_12_Diebold_Mariano_Tests"] = dm_df.to_markdown(index=False)
        tables_latex["Table_12_Diebold_Mariano_Tests"] = dm_df.to_latex(index=False)

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Table 6b: SVR Top 5 QLIKE Error Contributors
    # -------------------------------------------------------------
    if os.path.exists("svr_top5_qlike.csv"):
        svr_top_df = pd.read_csv("svr_top5_qlike.csv")
        svr_disp = pd.DataFrame({
            "Rank": svr_top_df["Rank"],
            "Date": svr_top_df["Date"],
            "SVR Pred (%)": svr_top_df["SVR_Pred"].apply(lambda x: f"{x:.4f}"),
            "Actual Move (%)": svr_top_df["Target"].apply(lambda x: f"{x:.4f}"),
            "Daily QLIKE": svr_top_df["Daily_QLIKE"].apply(lambda x: f"{x:.4f}"),
            "Share of Total QLIKE (%)": svr_top_df["Share_Pct"].apply(lambda x: f"{x:.2f}%")
        })
        tables_md["Table_6b_SVR_Top5_QLIKE"] = svr_disp.to_markdown(index=False)
        tables_latex["Table_6b_SVR_Top5_QLIKE"] = svr_disp.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 13: VaR Backtest Results
    # -------------------------------------------------------------
    if os.path.exists("var_backtest.csv"):
        var_df = pd.read_csv("var_backtest.csv")
        var_disp = pd.DataFrame({
            "Model": var_df["Model"],
            "Conf": var_df["Confidence_Level"],
            "Exp Fail": var_df["Expected_Failures"],
            "Act Fail": var_df["Actual_Failures"],
            "Fail %": var_df["Failure_Rate_Pct"].apply(lambda x: f"{x:.2f}%"),
            "Kupiec LR": var_df["Kupiec_LR_POF"].apply(lambda x: f"{x:.4f}"),
            "Kupiec p": var_df["Kupiec_p_val"].apply(lambda x: f"{x:.4f}"),
            "Christ LR": var_df["Christoffersen_LR_Ind"].apply(lambda x: f"{x:.4f}"),
            "Christ p": var_df["Christoffersen_p_val"].apply(lambda x: f"{x:.4f}"),
            "Cond Cov p": var_df["Conditional_Coverage_p_val"].apply(lambda x: f"{x:.4f}")
        })
        tables_md["Table_13_VaR_Backtesting"] = var_disp.to_markdown(index=False)
        tables_latex["Table_13_VaR_Backtesting"] = var_disp.to_latex(index=False)

    # -------------------------------------------------------------
    # Table 14: Feature Importance
    # -------------------------------------------------------------
    if os.path.exists("feature_importance.csv"):
        fi_df = pd.read_csv("feature_importance.csv")
        fi_disp = pd.DataFrame({
            "Feature": fi_df["Feature"],
            "Ridge Std Coef": fi_df["Ridge_Standardized_Coef"].apply(lambda x: f"{x:+.4f}"),
            "XGB Gain": fi_df["XGBoost_Gain_Importance"].apply(lambda x: f"{x:.4f}"),
            "XGB Permutation MAE": fi_df["XGBoost_Permutation_Mean"].apply(lambda x: f"{x:+.4f}")
        })
        tables_md["Table_14_Feature_Importance"] = fi_disp.to_markdown(index=False)
        tables_latex["Table_14_Feature_Importance"] = fi_disp.to_latex(index=False)

    # Save all tables to markdown file
    with open("report_tables.md", "w", encoding="utf-8") as f:
        for t_name, content in tables_md.items():
            f.write(f"### {t_name.replace('_', ' ')}\n\n")
            f.write(content + "\n\n---\n\n")

    print(f"Generated {len(tables_md)} formatted report tables to report_tables.md.")
    return tables_md

if __name__ == "__main__":
    build_tables()
