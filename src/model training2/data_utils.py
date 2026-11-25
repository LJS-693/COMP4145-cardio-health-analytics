"""Canonical seven-feature inputs, metrics and deterministic grouped CV."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = ['age_years','gender','bmi','ap_hi','ap_lo','cholesterol','gluc']
NUMERIC = ['age_years','bmi','ap_hi','ap_lo']
CATEGORIES = ['gender','cholesterol','gluc']
METRICS = ['roc_auc','accuracy','precision','recall','f1','specificity','average_precision']
PRIMARY = METRICS[:5]
DATA = Path('data/processed/caroline_cleaned version')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def decode(source,adapter,original_units=False):
    rows=[]
    for line,r in enumerate(read_csv(source),1):
        out={'id':r.get('id',str(line))}
        for col in NUMERIC:
            value=float(r[col])
            if not original_units:
                value=value*(adapter[col]['max']-adapter[col]['min'])+adapter[col]['min']
            out[col]=value
        if original_units:
            for col in CATEGORIES:out[col]=int(r[col])
        else:
            out['gender']=int(r['gender'])+1
            for col in ['cholesterol','gluc']:
                flags=[str(r[f'{col}_{i}']).lower() in ['true','1','1.0'] for i in [1,2,3]]
                assert sum(flags)==1,f'Invalid one-hot {col}'
                out[col]=flags.index(True)+1
        if 'cardio' in r:out['cardio']=int(r['cardio'])
        assert out['gender'] in [1,2] and out['cholesterol'] in [1,2,3] and out['gluc'] in [1,2,3]
        assert all(np.isfinite(out[c]) for c in NUMERIC)
        rows.append(out)
    return rows


def load_development(project,plan):
    source=project/DATA/'cardio_train_80.csv'
    adapter_path=project/DATA/'denormalization_params.json'
    adapter=json.loads(adapter_path.read_text())
    rows=decode(source,adapter)
    raw_path=project/'data/raw/cardio_train.csv'
    raw={r['id']:r for r in read_csv_semicolon(raw_path)}
    fields=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    for row in rows:
        r=raw[row['id']]
        canonical={'age_years':float(r['age'])/365.25,'bmi':float(r['weight'])/(float(r['height'])/100)**2,
                   'ap_hi':float(r['ap_hi']),'ap_lo':float(r['ap_lo'])}
        for col,value in canonical.items():
            assert abs(row[col]-value)<1e-8
            row[col]=value
        for col in [*CATEGORIES,'cardio']:assert row[col]==int(r[col])
        signature=json.dumps([float(r[c]) for c in fields],separators=(',',':'))
        row['profile_group']=hashlib.sha256(signature.encode()).hexdigest()
    X=np.array([[r[c] for c in FEATURES] for r in rows],dtype=float)
    y=np.array([r['cardio'] for r in rows],dtype=int)
    groups=np.array([r['profile_group'] for r in rows])
    assert len(y)==plan['records'] and len({r['id'] for r in rows})==len(y)
    assert sha256(source)==plan['training_sha256'] and np.isfinite(X).all()
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=plan['seed']).split(X,y,groups))
    fold_ids=np.zeros(len(y),dtype=int)
    for k,(train,valid) in enumerate(folds,1):
        assert not set(groups[train])&set(groups[valid]) and not np.any(fold_ids[valid])
        fold_ids[valid]=k
    assert np.all(fold_ids>0)
    legacy=project/'outputs/model training1/predictions/xgboost/development_oof_predictions.csv'
    checked=False
    if legacy.exists():
        lookup={r['id']:int(r['fold']) for r in read_csv(legacy)}
        assert all(lookup[r['id']]==fold_ids[i] for i,r in enumerate(rows))
        checked=True
    provenance={'training_source':str(DATA/'cardio_train_80.csv'),'training_sha256':sha256(source),
        'raw_sha256':sha256(raw_path),'adapter_sha256':sha256(adapter_path),
        'records':len(y),'positive_records':int(y.sum()),'legacy_folds_verified':checked}
    return X,y,rows,adapter,folds,fold_ids,provenance


def read_csv_semicolon(path):
    with path.open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file,delimiter=';'))


def metrics(y,probability):
    probability=np.asarray(probability)
    if probability.ndim==2:probability=probability[:,1]
    pred=(probability>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,probability)),'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)),
        'average_precision':float(average_precision_score(y,probability)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def scorer_value(y,probability,metric):
    probability=np.asarray(probability)
    if probability.ndim==2:probability=probability[:,1]
    if metric=='roc_auc':return float(roc_auc_score(y,probability))
    if metric=='average_precision':return float(average_precision_score(y,probability))
    pred=(probability>=.5).astype(int)
    if metric=='specificity':return float(recall_score(y,pred,pos_label=0,zero_division=0))
    functions={'accuracy':accuracy_score,'precision':precision_score,'recall':recall_score,'f1':f1_score}
    return float(functions[metric](y,pred,**({} if metric=='accuracy' else {'zero_division':0})))
