import csv, json, hashlib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from train_logistic import decode_records, DATA_DIR, FEATURE_SET, SEED

def load_development(project):
    data=project/DATA_DIR/'cardio_train_80.csv'
    params=json.loads((project/DATA_DIR/'denormalization_params.json').read_text())
    features=json.loads((project/'config/feature_sets.json').read_text())[FEATURE_SET]
    records=decode_records(data,params)
    with (project/'data/raw/cardio_train.csv').open(encoding='utf-8-sig',newline='') as f:
        raw={r['id']:r for r in csv.DictReader(f,delimiter=';')}
    profile_fields=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    for r in records:
        original=raw[r['id']]
        expected={c:float(original[c]) for c in ['height','weight','ap_hi','ap_lo']}
        expected['age_years']=float(original['age'])/365.25
        expected['bmi']=float(original['weight'])/(float(original['height'])/100)**2
        for c,v in expected.items(): assert abs(r[c]-v)<1e-8; r[c]=v
        for c in ['gender','cholesterol','gluc','smoke','alco','active','cardio']:
            assert r[c]==int(original[c])
        sig=json.dumps([float(original[c]) for c in profile_fields],separators=(',',':'))
        r['profile_group']=hashlib.sha256(sig.encode()).hexdigest()
    X=np.array([[r[c] for c in features] for r in records],dtype=float)
    y=np.array([r['cardio'] for r in records]); groups=np.array([r['profile_group'] for r in records])
    assert np.isfinite(X).all() and len({r['id'] for r in records})==len(records)
    baseline=json.loads((project/'reports/random_forest/model_metadata.json').read_text())
    sha=hashlib.sha256(data.read_bytes()).hexdigest()
    assert baseline['features']==features and baseline['training_sha256']==sha
    with (project/'outputs/predictions/random_forest/development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        baseline_folds={r['id']:int(r['fold']) for r in csv.DictReader(f)}
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    fold_assign=np.zeros(len(y),int)
    for fold,(tr,va) in enumerate(cv,1):
        assert not set(groups[tr])&set(groups[va])
        assert not np.any(fold_assign[va]); fold_assign[va]=fold
        assert all(baseline_folds[records[i]['id']]==fold for i in va)
    assert np.all(fold_assign>0)
    return X,y,features,records,params,cv,sha,baseline
