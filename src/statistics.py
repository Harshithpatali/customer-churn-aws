from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu

def categorical_tests(df: pd.DataFrame, target="Churn") -> pd.DataFrame:
    rows=[]
    for col in df.select_dtypes(include="object").columns:
        if col in {"customerID",target}: continue
        table=pd.crosstab(df[col],df[target]); chi2,p,_,_=chi2_contingency(table); n=table.to_numpy().sum()
        rows.append({"feature":col,"test":"chi_square","statistic":float(chi2),"p_value":float(p),"effect_size_phi":float(np.sqrt(chi2/n)) if n else 0.0,"significant_0_05":bool(p<0.05)})
    return pd.DataFrame(rows).sort_values("p_value")

def numeric_tests(df: pd.DataFrame, target="Churn") -> pd.DataFrame:
    rows=[]
    for col in ["tenure","MonthlyCharges","TotalCharges"]:
        values=pd.to_numeric(df[col],errors="coerce"); churned=values[df[target]=="Yes"].dropna(); retained=values[df[target]=="No"].dropna()
        statistic,p_value=mannwhitneyu(churned,retained,alternative="two-sided")
        rows.append({"feature":col,"test":"mann_whitney_u","statistic":float(statistic),"p_value":float(p_value),"churn_median":float(churned.median()),"non_churn_median":float(retained.median()),"significant_0_05":bool(p_value<0.05)})
    return pd.DataFrame(rows).sort_values("p_value")

def logistic_odds_ratios(df: pd.DataFrame, output: Path) -> dict:
    work=df.copy(); work["Churn"]=work["Churn"].map({"Yes":1,"No":0}); work["TotalCharges"]=pd.to_numeric(work["TotalCharges"],errors="coerce")
    cols=["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]; x=sm.add_constant(work[cols].fillna(work[cols].median())); model=sm.Logit(work["Churn"],x).fit(disp=False); conf=model.conf_int()
    pd.DataFrame({"feature":model.params.index,"odds_ratio":np.exp(model.params.values),"p_value":model.pvalues.values,"ci_low":np.exp(conf[0].values),"ci_high":np.exp(conf[1].values)}).to_csv(output,index=False)
    return {"pseudo_r2":float(model.prsquared),"aic":float(model.aic),"bic":float(model.bic),"n":int(model.nobs),"odds_ratios_path":str(output)}

def run_statistics(csv_path: Path, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True,exist_ok=True); df=pd.read_csv(csv_path); categorical=categorical_tests(df); numeric=numeric_tests(df)
    categorical.to_csv(report_dir/"categorical_significance.csv",index=False); numeric.to_csv(report_dir/"numeric_significance.csv",index=False)
    logit=logistic_odds_ratios(df,report_dir/"statistical_logit_odds_ratios.csv")
    summary={"categorical_tests":len(categorical),"numeric_tests":len(numeric),"logistic_regression":logit,"multiple_testing_note":"P-values are exploratory; formal inference should pre-specify hypotheses and apply an appropriate multiplicity correction."}
    (report_dir/"statistical_summary.json").write_text(json.dumps(summary,indent=2)); return summary
