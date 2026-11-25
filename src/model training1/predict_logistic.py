"""Score Caroline-format normalized CSV using saved adapter + logistic pipeline.
This script is not run on the test set during development.
Usage: python3 src/predict_logistic.py --input INPUT.csv --output predictions.csv
"""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from train_logistic import decode_records

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',type=Path,default=root/'models/logistic_regression/logistic_pipeline.joblib')
    a=p.parse_args()
    if a.input.resolve()==a.output.resolve():
        raise ValueError('Output must not overwrite input')
    bundle=joblib.load(a.model)
    rows=decode_records(a.input,bundle['denormalization_params'])
    X=np.array([[r[c] for c in bundle['features']] for r in rows])
    scores=bundle['pipeline'].predict_proba(X)[:,1]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','predicted_score','predicted_cardio']);w.writeheader()
        w.writerows({'id':r['id'],'predicted_score':float(s),'predicted_cardio':int(s>=bundle['threshold'])} for r,s in zip(rows,scores))
    print('Scored',len(rows),'records')

if __name__=='__main__': main()
