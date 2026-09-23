from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import CATEGORICAL_FEATURES, ID_COL, NUMERIC_FEATURES, RANDOM_STATE, TARGET
from .data_quality import save_profile
from services.data_pipeline.pipeline import build_dataset

ENGINEERED_NUMERIC=["service_count","monthly_to_tenure_value","avg_monthly_revenue","contract_risk_flag","fiber_risk_flag","support_gap_flag"]

def clean(df: pd.DataFrame)->pd.DataFrame:
    from services.data_pipeline.pipeline import clean as _clean
    return _clean(df)

def add_features(df: pd.DataFrame)->pd.DataFrame:
    from services.data_pipeline.pipeline import engineer
    return engineer(df)

def build_preprocessor()->ColumnTransformer:
    numeric=NUMERIC_FEATURES+ENGINEERED_NUMERIC
    categorical=CATEGORICAL_FEATURES+["tenure_group"]
    return ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical),
    ])

def make_pipeline(model:Any)->Pipeline:
    return Pipeline([("preprocess",build_preprocessor()),("model",model)])

def metrics(y_true,prob,threshold=0.5):
    pred=(prob>=threshold).astype(int)
    return {"roc_auc":float(roc_auc_score(y_true,prob)),"pr_auc":float(average_precision_score(y_true,prob)),"brier_score":float(brier_score_loss(y_true,prob)),"accuracy":float(accuracy_score(y_true,pred)),"precision":float(precision_score(y_true,pred,zero_division=0)),"recall":float(recall_score(y_true,pred,zero_division=0)),"f1":float(f1_score(y_true,pred,zero_division=0))}

def optimize_threshold(y,prob):
    rows=[]
    for threshold in np.arange(0.15,0.81,0.01):
        m=metrics(y,prob,float(threshold)); m["threshold"]=float(threshold); m["business_score"]=float(0.65*m["recall"]+0.35*m["precision"]); rows.append(m)
    return max(rows,key=lambda r:r["business_score"])

def train(csv_path:Path,output_dir:Path,report_dir:Path|None=None):
    output_dir.mkdir(parents=True,exist_ok=True); report_dir=report_dir or output_dir.parent/"reports"; report_dir.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(csv_path); save_profile(raw,report_dir/"data_quality.json")
    df=build_dataset(csv_path); X=df.drop(columns=[TARGET,ID_COL],errors="ignore"); y=df[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,stratify=y,random_state=RANDOM_STATE)
    candidates={
        "logistic_regression":make_pipeline(LogisticRegression(max_iter=3000,class_weight="balanced",random_state=RANDOM_STATE)),
        "xgboost":make_pipeline(XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.035,subsample=0.85,colsample_bytree=0.85,reg_lambda=2.0,min_child_weight=3,objective="binary:logistic",eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=4))
    }
    comparison={}
    for name,model in candidates.items():
        model.fit(X_train,y_train); comparison[name]=metrics(y_test,model.predict_proba(X_test)[:,1])
    search=RandomizedSearchCV(candidates["xgboost"],{"model__max_depth":[3,4,5,6],"model__learning_rate":[0.02,0.035,0.05,0.08],"model__subsample":[0.75,0.85,1.0],"model__colsample_bytree":[0.75,0.85,1.0],"model__min_child_weight":[1,3,5,8]},n_iter=16,scoring="average_precision",cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE),random_state=RANDOM_STATE,n_jobs=-1,refit=True)
    search.fit(X_train,y_train); best=search.best_estimator_; test_prob=best.predict_proba(X_test)[:,1]; threshold=optimize_threshold(y_test,test_prob); final_metrics=metrics(y_test,test_prob,threshold["threshold"])
    artifact={"pipeline":best,"threshold":threshold["threshold"],"target":TARGET,"version":"2.0.0","feature_columns":list(X.columns),"model_family":"XGBoost","random_state":RANDOM_STATE}
    joblib.dump(artifact,output_dir/"churn_pipeline.joblib")
    transformed=best.named_steps["preprocess"].get_feature_names_out().tolist()
    (output_dir/"feature_contract.json").write_text(json.dumps({"raw_features":list(X.columns),"transformed_features":transformed},indent=2))
    report={"dataset_rows":int(len(df)),"churn_rate":float(y.mean()),"comparison":comparison,"best_params":search.best_params_,"test":final_metrics,"threshold_selection":threshold,"cv":{"folds":5,"scoring":"average_precision"}}
    (output_dir/"metrics.json").write_text(json.dumps(report,indent=2))
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(20).to_csv(report_dir/"hyperparameter_search.csv",index=False)
    pd.DataFrame([{"model":k,**v} for k,v in comparison.items()]+[{"model":"xgboost_tuned",**final_metrics}]).to_csv(report_dir/"model_comparison.csv",index=False)
    test_out=X_test.copy(); test_out[TARGET]=y_test.values; test_out["churn_probability"]=test_prob; test_out["risk_band"]=pd.cut(test_prob,bins=[-0.01,threshold["threshold"],0.70,1.01],labels=["Low","Medium","High"],include_lowest=True); test_out.to_csv(report_dir/"test_predictions.csv",index=False)
    return report
