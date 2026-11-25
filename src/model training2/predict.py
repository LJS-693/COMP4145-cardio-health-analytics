"""Predict with an explicitly selected model training2 pipeline; no default model."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from data_utils import decode,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--original-units',action='store_true',help='Input contains the seven canonical features; original gender 1/2 and cholesterol/gluc 1/2/3 codes.')
    args=parser.parse_args()
    if args.output.resolve() in [args.input.resolve(),args.model.resolve()]:raise ValueError('Output must not overwrite input/model')
    bundle=joblib.load(args.model)
    rows=decode(args.input,bundle['denormalization_params'],args.original_units)
    X=np.array([[r[c] for c in bundle['features']] for r in rows],dtype=float)
    score=bundle['pipeline'].predict_proba(X)[:,1]
    assert np.isfinite(score).all() and np.all((score>=0)&(score<=1))
    write_csv(args.output,[{'id':r['id'],'predicted_score':float(p),'predicted_cardio':int(p>=bundle['threshold'])}
                          for r,p in zip(rows,score)])
    print(f'Scored {len(rows)} records. Scores estimate the cardio label; they are not validated clinical priorities.')


if __name__=='__main__':main()
