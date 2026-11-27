"""Data, metric and export helpers. All model fitting is visible in the notebooks."""
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    roc_auc_score,average_precision_score,confusion_matrix,make_scorer,roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

FEATURES=['age_years','gender','bmi','ap_hi','ap_lo','cholesterol','gluc']
NUMERIC=['age_years','bmi','ap_hi','ap_lo']
CATEGORIES=['gender','cholesterol','gluc']
METRICS=['roc_auc','accuracy','precision','recall','f1','specificity','average_precision']
SEED=4145
EXPECTED_TRAIN_SHA='18373a55c8e6d3b9a7a50f102a98d72b930d523ccf1810759d7b38a81c392325'
MODEL_NAMES={'logistic_regression':'Logistic Regression','random_forest':'Random Forest','xgboost':'XGBoost'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def find_project(override=None):
    if override is not None:
        root=Path(override).expanduser().resolve()
        if not (root/'data/processed/cardio_train_80.csv').exists():
            raise FileNotFoundError('PROJECT_ROOT must point to the project containing data/.')
        return root
    for root in [Path.cwd(),*Path.cwd().parents]:
        if (root/'data/processed/cardio_train_80.csv').exists():return root
    raise FileNotFoundError('Set PROJECT_ROOT to the folder containing data/ and predictive_analytics/.')

def load_development(root):
    """Only development CSV, normalization adapter and source records; never test CSV."""
    data=Path(root)/'data/processed'
    source=data/'cardio_train_80.csv';adapter_path=data/'denormalization_params.json'
    if sha(source)!=EXPECTED_TRAIN_SHA:raise ValueError('Unexpected development data version; review the data and protocol before changing the expected hash.')
    clean=pd.read_csv(source);adapter=json.loads(adapter_path.read_text())
    raw_path=Path(root)/'data/raw/cardio_train.csv'
    raw=pd.read_csv(raw_path,sep=';').set_index('id').loc[clean['id']].reset_index()
    values={c:clean[c].to_numpy(float)*(adapter[c]['max']-adapter[c]['min'])+adapter[c]['min'] for c in NUMERIC}
    canonical={'age_years':raw['age'].to_numpy(float)/365.25,
               'bmi':raw['weight'].to_numpy(float)/(raw['height'].to_numpy(float)/100)**2,
               'ap_hi':raw['ap_hi'].to_numpy(float),'ap_lo':raw['ap_lo'].to_numpy(float)}
    for c in NUMERIC:
        assert np.allclose(values[c],canonical[c],rtol=0,atol=1e-8)
        values[c]=canonical[c]
    values['gender']=clean['gender'].to_numpy(int)+1
    for c in ['cholesterol','gluc']:
        flags=np.column_stack([clean[f'{c}_{k}'].astype(str).str.lower().isin(['true','1','1.0']).to_numpy() for k in [1,2,3]])
        assert np.all(flags.sum(axis=1)==1)
        values[c]=flags.argmax(axis=1)+1
    for c in CATEGORIES:assert np.array_equal(values[c],raw[c].to_numpy(int))
    X=np.column_stack([values[c] for c in FEATURES]).astype(float)
    y=clean['cardio'].to_numpy(int);ids=clean['id'].to_numpy()
    assert len(y)==54855 and len(np.unique(ids))==len(ids)
    assert np.array_equal(y,raw['cardio'].to_numpy(int)) and np.isfinite(X).all()
    fields=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    groups=np.array([hashlib.sha256(json.dumps(list(row),separators=(',',':')).encode()).hexdigest()
                     for row in raw[fields].to_numpy(float)])
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    fold_ids=np.zeros(len(y),int)
    for k,(train,valid) in enumerate(cv,1):
        assert not set(groups[train])&set(groups[valid])
        assert not np.any(fold_ids[valid]);fold_ids[valid]=k
    assert np.all(fold_ids>0)
    fold_hash=hashlib.sha256(fold_ids.tobytes()).hexdigest()
    fingerprint={'training_sha256':sha(source),'adapter_sha256':sha(adapter_path),'raw_sha256':sha(raw_path),
                 'fold_assignment_sha256':fold_hash,'features':FEATURES,'records':len(y),'seed':SEED,'folds':5}
    return {'X':X,'y':y,'ids':ids,'groups':groups,'cv':cv,'fold_ids':fold_ids,'adapter':adapter,'fingerprint':fingerprint}

def build_pipeline(name,parameters):
    preprocessing=ColumnTransformer([
        ('numeric',StandardScaler() if name=='logistic_regression' else 'passthrough',[FEATURES.index(c) for c in NUMERIC]),
        ('categories',OneHotEncoder(categories=[[1,2],[1,2,3],[1,2,3]],drop='first',sparse_output=False,handle_unknown='error'),
         [FEATURES.index(c) for c in CATEGORIES])])
    constructors={'logistic_regression':LogisticRegression,'random_forest':RandomForestClassifier,'xgboost':xgboost.XGBClassifier}
    return Pipeline([('preprocess',preprocessing),('classifier',constructors[name](**parameters))])

def evaluate(y,score):
    score=np.asarray(score)
    if score.ndim==2:score=score[:,1]
    pred=score>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,score)),'accuracy':float(accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)),
            'average_precision':float(average_precision_score(y,score)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def score_metric(y,score,metric):
    score=np.asarray(score)
    if score.ndim==2:score=score[:,1]
    if metric=='roc_auc':return roc_auc_score(y,score)
    if metric=='average_precision':return average_precision_score(y,score)
    pred=score>=.5
    if metric=='specificity':return recall_score(y,pred,pos_label=0,zero_division=0)
    f={'accuracy':accuracy_score,'precision':precision_score,'recall':recall_score,'f1':f1_score}[metric]
    return f(y,pred,**({} if metric=='accuracy' else {'zero_division':0}))
def scorers():return {m:make_scorer(score_metric,response_method='predict_proba',metric=m) for m in METRICS}
def select_auc(results):return int(np.argmax(results['mean_test_roc_auc']))

def begin_stage(run_dir,stage,ctx):
    run_dir=Path(run_dir);target=run_dir/stage
    if target.exists():raise FileExistsError(f'{target} exists. Use a new RUN_NAME or explicitly move the previous output; historical output is never silently overwritten.')
    run_dir.mkdir(parents=True,exist_ok=True)
    config=run_dir/'data_protocol.json'
    if config.exists():assert json.loads(config.read_text())==ctx['fingerprint'],'Data or CV partition differs from the previous stage.'
    else:
        dump_json(config,ctx['fingerprint'])
        pd.DataFrame({'id':ctx['ids'],'profile_group':ctx['groups'],'fold':ctx['fold_ids']}).to_csv(run_dir/'development_folds.csv',index=False)
    target.mkdir();return target

def collect_cv(estimators,ctx,reference=None,best_index=None):
    X,y=ctx['X'],ctx['y'];oof=np.full(len(y),np.nan);folds=[]
    assert len(estimators)==5
    for fold,(est,(train,valid)) in enumerate(zip(estimators,ctx['cv']),1):
        assert np.array_equal(est.classes_,[0,1])
        train_score=evaluate(y[train],est.predict_proba(X[train])[:,1])
        validation_score=evaluate(y[valid],est.predict_proba(X[valid])[:,1])
        oof[valid]=est.predict_proba(X[valid])[:,1]
        folds.append({'fold':fold,'training_records':len(train),'validation_records':len(valid),
            **{f'train_{m}':train_score[m] for m in METRICS},**validation_score})
        if 'scale' in est.named_steps:raise AssertionError('Scaling must be in the preprocessing pipeline.')
        if reference is not None:
            for m in METRICS:
                assert np.isclose(validation_score[m],reference[f'split{fold-1}_test_{m}'][best_index],atol=1e-10,rtol=0)
    assert np.isfinite(oof).all()
    table=pd.DataFrame(folds)
    summary={m:float(table[m].mean()) for m in METRICS}
    summary.update({'training_accuracy':float(table.train_accuracy.mean()),'training_auc':float(table.train_roc_auc.mean())})
    return {'summary':summary,'fold_metrics':table,'oof':oof,'pooled_oof_metrics':evaluate(y,oof)}

def save_model_result(stage_dir,name,final_model,parameters,evaluation,ctx,search_results=None,carried_from=None):
    target=Path(stage_dir)/name
    if target.exists():raise FileExistsError(str(target))
    assert final_model.named_steps['preprocess'].transform(ctx['X'][:2]).shape[1]==9
    assert all(final_model.named_steps['classifier'].get_params()[k]==v for k,v in parameters.items())
    if name=='logistic_regression':
        assert np.allclose(final_model.named_steps['preprocess'].named_transformers_['numeric'].mean_,ctx['X'][:,[FEATURES.index(c) for c in NUMERIC]].mean(axis=0))
    target.mkdir()
    bundle={'pipeline':final_model,'features':FEATURES,'threshold':.5,'parameters':parameters,
            'denormalization_params':ctx['adapter'],'target':'cardio'}
    joblib.dump(bundle,target/'model_pipeline.joblib',compress=('xz',3))
    loaded=joblib.load(target/'model_pipeline.joblib')
    assert np.allclose(loaded['pipeline'].predict_proba(ctx['X'][:10]),final_model.predict_proba(ctx['X'][:10]))
    evaluation['fold_metrics'].to_csv(target/'cv_fold_metrics.csv',index=False)
    pd.DataFrame({'id':ctx['ids'],'fold':ctx['fold_ids'],'actual_cardio':ctx['y'],'predicted_score':evaluation['oof'],
                  'predicted_cardio':(evaluation['oof']>=.5).astype(int)}).to_csv(target/'development_oof_predictions.csv',index=False)
    pd.DataFrame([{'metric':m,'validation_mean':evaluation['summary'][m],
                   'validation_std_sample':float(evaluation['fold_metrics'][m].std(ddof=1))} for m in METRICS]).to_csv(target/'cv_summary.csv',index=False)
    if search_results is not None:pd.DataFrame(search_results).to_csv(target/'gridsearch_results.csv',index=False)
    metadata={'model':name,'stage':Path(stage_dir).name,'parameters':parameters,'summary':evaluation['summary'],
        'pooled_oof_metrics':evaluation['pooled_oof_metrics'],'data_protocol':ctx['fingerprint'],'threshold':.5,
        'test_evaluated':False,'carried_from':carried_from,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'sklearn':sklearn.__version__,'xgboost':xgboost.__version__}}
    dump_json(target/'metadata.json',metadata)

def finish_stage(stage_dir,ctx,models):
    records=[]
    for name in models:
        meta=json.loads((Path(stage_dir)/name/'metadata.json').read_text())
        records.append({'model':name,'stage':Path(stage_dir).name,'status':'carried forward' if meta['carried_from'] else 'trained',**meta['summary']})
    table=pd.DataFrame(records);table.to_csv(Path(stage_dir)/'model_comparison.csv',index=False)
    dump_json(Path(stage_dir)/'complete.json',{'models':list(models),'data_protocol':ctx['fingerprint']})
    return table

def load_stage(run_dir,stage,ctx):
    folder=Path(run_dir)/stage;marker=folder/'complete.json'
    if not marker.exists():raise FileNotFoundError(f'Run the preceding notebook completely first: {folder}')
    saved=json.loads(marker.read_text());assert saved['data_protocol']==ctx['fingerprint']
    output={}
    for name in saved['models']:
        target=folder/name;meta=json.loads((target/'metadata.json').read_text())
        assert meta['data_protocol']==ctx['fingerprint']
        frame=pd.read_csv(target/'development_oof_predictions.csv',float_precision='round_trip')
        assert np.array_equal(frame.id.to_numpy(),ctx['ids']) and np.array_equal(frame.fold.to_numpy(),ctx['fold_ids'])
        assert np.array_equal(frame.actual_cardio.to_numpy(),ctx['y'])
        bundle=joblib.load(target/'model_pipeline.joblib');assert bundle['parameters']==meta['parameters']
        output[name]={'model':bundle['pipeline'],'parameters':meta['parameters'],'metadata':meta,
            'summary':meta['summary'],'fold_metrics':pd.read_csv(target/'cv_fold_metrics.csv'),
            'oof':frame.predicted_score.to_numpy(),'pooled_oof_metrics':meta['pooled_oof_metrics']}
    return output

def comparison_plots(results,ctx,stage_dir):
    fig,ax=plt.subplots(figsize=(8,6))
    for name,r in results.items():
        fpr,tpr,_=roc_curve(ctx['y'],r['oof'])
        ax.plot(fpr,tpr,label=f'{MODEL_NAMES[name]}: pooled OOF AUC={roc_auc_score(ctx["y"],r["oof"]):.4f}')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='Recall',title='Development OOF ROC comparison');ax.legend(loc='lower right')
    fig.tight_layout();fig.savefig(Path(stage_dir)/'roc_comparison.png',dpi=160);plt.show()
    fig,axes=plt.subplots(1,len(results),figsize=(5*len(results),4.8),squeeze=False)
    for ax,(name,r) in zip(axes[0],results.items()):
        cm=confusion_matrix(ctx['y'],r['oof']>=.5,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        ax.set(title=MODEL_NAMES[name],xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='Actual label')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>.65*cm.max() else '#203040',fontsize=15)
    fig.suptitle('Development OOF confusion matrices / threshold=0.5');fig.tight_layout()
    fig.savefig(Path(stage_dir)/'confusion_matrices.png',dpi=160);plt.show()

def compare_history(run_dir,stages):
    frames=[pd.read_csv(Path(run_dir)/stage/'model_comparison.csv') for stage in stages]
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(Path(run_dir)/'all_completed_stages.csv',index=False)
    return result
