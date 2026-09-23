"""Create a tiny deterministic smoke-test artifact when the IBM CSV is unavailable.
This is for API/container tests only; it must NOT be reported as the IBM model.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import train

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/raw/bootstrap_telco.csv'
rng=np.random.default_rng(42)
n=500
rows={
'customerID':[f'DEMO-{i:04d}' for i in range(n)],'gender':rng.choice(['Female','Male'],n),'SeniorCitizen':rng.integers(0,2,n),'Partner':rng.choice(['Yes','No'],n),'Dependents':rng.choice(['Yes','No'],n),'tenure':rng.integers(0,73,n),'PhoneService':rng.choice(['Yes','No'],n),'MultipleLines':rng.choice(['Yes','No','No phone service'],n),'InternetService':rng.choice(['DSL','Fiber optic','No'],n),'OnlineSecurity':rng.choice(['Yes','No','No internet service'],n),'OnlineBackup':rng.choice(['Yes','No','No internet service'],n),'DeviceProtection':rng.choice(['Yes','No','No internet service'],n),'TechSupport':rng.choice(['Yes','No','No internet service'],n),'StreamingTV':rng.choice(['Yes','No','No internet service'],n),'StreamingMovies':rng.choice(['Yes','No','No internet service'],n),'Contract':rng.choice(['Month-to-month','One year','Two year'],n),'PaperlessBilling':rng.choice(['Yes','No'],n),'PaymentMethod':rng.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],n),'MonthlyCharges':rng.uniform(18,120,n).round(2),'TotalCharges':rng.uniform(20,8000,n).round(2)
}
df=pd.DataFrame(rows)
logit=-1.2 + 0.02*(df.tenure<12) + 0.9*(df.Contract=='Month-to-month') + 0.4*(df.PaymentMethod=='Electronic check') + 0.012*(df.MonthlyCharges-60)
p_churn=1/(1+np.exp(-logit)); df['Churn']=np.where(rng.random(n)<p_churn,'Yes','No')
p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False)
print(train(p,ROOT/'models'))
