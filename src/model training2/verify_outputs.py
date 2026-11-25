"""Read-only reconciliation of saved training2 outputs; no fitting or test scoring."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid
from data_utils import FEATURES,METRICS,read_csv,sha256,load_development,metrics

ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();out=args.output_root or args.project_root
    report=out/'reports/model training2'
    config=out/'config/model training2/training_plan.json'
    plan=json.loads(config.read_text())
    X,y,records,adapter,cv,fold_ids,provenance=load_development(args.project_root,plan)
    rows=read_csv(report/'all_rounds_comparison.csv');assert len(rows)==9
    states={};n=0
    for row in rows:
        stage,name=row['stage'],row['model']
        base=report/stage/name
        meta=json.loads((base/'model_metadata.json').read_text())
        assert meta['features']==FEATURES and meta['encoded_feature_count']==9
        assert meta['training_sha256']==provenance['training_sha256']
        assert meta['training_plan_sha256']==sha256(config)
        assert meta['training_code_sha256']==sha256(Path(__file__).parent/'train_all.py')
        assert meta['data_utils_sha256']==sha256(Path(__file__).parent/'data_utils.py')
        assert meta['test_evaluated'] is False and meta['threshold']==.5
        bundle=joblib.load(out/'models/model training2'/stage/name/'model_pipeline.joblib')
        assert bundle['features']==FEATURES and bundle['parameters']==meta['parameters']
        classifier=bundle['pipeline'].named_steps['classifier']
        assert all(classifier.get_params()[k]==v for k,v in meta['parameters'].items())
        assert bundle['pipeline'].named_steps['preprocess'].transform(X[:3]).shape==(3,9)
        saved=read_csv(out/'outputs/model training2/predictions'/stage/name/'development_oof_predictions.csv')
        assert len(saved)==len(y)
        assert all(r['id']==records[i]['id'] and int(r['fold'])==fold_ids[i] and int(r['actual_cardio'])==y[i] for i,r in enumerate(saved))
        p=np.array([float(r['predicted_score']) for r in saved])
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        assert all(int(r['predicted_cardio'])==int(p[i]>=.5) for i,r in enumerate(saved))
        assert metrics(y,p)==meta['pooled_oof_metrics']
        folds=read_csv(base/'cv_fold_metrics.csv');assert len(folds)==5
        for i,(_,valid) in enumerate(cv):
            score=metrics(y[valid],p[valid])
            for m in METRICS:assert np.isclose(score[m],float(folds[i][m]),rtol=0,atol=1e-12)
        for m in METRICS:
            mean=np.mean([float(f[m]) for f in folds])
            assert np.isclose(mean,meta['cv_fold_means'][m],rtol=0,atol=1e-12)
            assert np.isclose(mean,float(row[m]),rtol=0,atol=1e-12)
        if stage!='baseline':
            previous=states[(meta['previous_stage'],name)]
            assert meta['previous_parameters']==previous['parameters']
            assert all(meta['parameters'][k]==v for k,v in previous['parameters'].items() if k not in plan[stage][name])
            grids=read_csv(base/'gridsearch_results.csv')
            assert len(grids)==len(ParameterGrid(plan[stage][name]))
            chosen=[g for g in grids if g['selected']=='True'];assert len(chosen)==1
            if stage=='round3':
                idx=min(range(len(grids)),key=lambda j:(float(grids[j]['mean_primary_rank']),-float(grids[j]['validation_roc_auc_mean']),int(grids[j]['n_estimators']),float(grids[j]['learning_rate']),j))
                assert grids[idx]['selected']=='True'
            else:assert chosen[0]['validation_roc_auc_mean']==str(max(float(g['validation_roc_auc_mean']) for g in grids))
            n+=len(grids)
        states[(stage,name)]=meta
        print('PASS',stage,name,'parameters, OOF, folds and inheritance')
    assert n==55
    assert len(read_csv(report/'all_grid_candidates.csv'))==55
    assert len(read_csv(report/'selected_model_fold_metrics.csv'))==45
    print('PASS: 9 models, 55 candidates, 45 folds; original data and training1 only read; no fit or test evaluation.')

if __name__=='__main__':main()
