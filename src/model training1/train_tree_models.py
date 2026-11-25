"""Fixed-parameter random forest and XGBoost baselines; same grouped folds as LR.
Run: python3 src/train_tree_models.py
Only the current training split is scored. No tuning, early stopping or final test evaluation.
"""
import argparse, csv, json, hashlib, base64, html, platform
from pathlib import Path
import numpy as np
import joblib, sklearn, xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_logistic import decode_records, metrics, write_csv, DATA_DIR, FEATURE_SET, SEED, THRESHOLD

ROOT=Path(__file__).resolve().parents[1]
MODEL_CONFIGS={
 'random_forest':{'n_estimators':300,'max_depth':None,'min_samples_leaf':1,
                  'max_features':'sqrt','class_weight':None,'random_state':SEED,'n_jobs':4},
 'xgboost':{'n_estimators':300,'max_depth':3,'learning_rate':.05,'subsample':1.,
             'colsample_bytree':1.,'reg_lambda':1.,'objective':'binary:logistic',
             'eval_metric':'logloss','tree_method':'hist','random_state':SEED,'n_jobs':4}
}

def build_pipeline(name,features):
    category=['gender','cholesterol','gluc']
    numeric=['age_years','bmi','ap_hi','ap_lo']
    binary=['smoke','alco','active']
    prep=ColumnTransformer([
      ('numeric','passthrough',[features.index(c) for c in numeric]),
      ('categories',OneHotEncoder(categories=[[1,2],[1,2,3],[1,2,3]],drop='first',
                         handle_unknown='error',sparse_output=False),[features.index(c) for c in category]),
      ('binary','passthrough',[features.index(c) for c in binary])])
    model=(RandomForestClassifier(**MODEL_CONFIGS[name]) if name=='random_forest'
           else xgboost.XGBClassifier(**MODEL_CONFIGS[name]))
    return Pipeline([('preprocess',prep),('classifier',model)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--output-root',type=Path)
    a=p.parse_args(); project=a.project_root; out=a.output_root or project
    source=project/DATA_DIR/'cardio_train_80.csv'
    param_file=project/DATA_DIR/'denormalization_params.json'
    params=json.loads(param_file.read_text())
    features=json.loads((project/'config/feature_sets.json').read_text())[FEATURE_SET]
    assert 'cardio' not in features and 'id' not in features
    records=decode_records(source,params)
    with (project/'data/raw/cardio_train.csv').open(encoding='utf-8-sig',newline='') as f:
        raw={r['id']:r for r in csv.DictReader(f,delimiter=';')}
    profile_fields=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    for r in records:
        original=raw[r['id']]
        expected={c:float(original[c]) for c in ['height','weight','ap_hi','ap_lo']}
        expected['age_years']=float(original['age'])/365.25
        expected['bmi']=float(original['weight'])/(float(original['height'])/100)**2
        for c,v in expected.items():
            assert abs(r[c]-v)<1e-8; r[c]=v
        for c in ['gender','cholesterol','gluc','smoke','alco','active','cardio']:
            assert r[c]==int(original[c])
        sig=json.dumps([float(original[c]) for c in profile_fields],separators=(',',':'))
        r['profile_group']=hashlib.sha256(sig.encode()).hexdigest()
    X=np.array([[r[c] for c in features] for r in records],dtype=float)
    y=np.array([r['cardio'] for r in records],dtype=int)
    groups=np.array([r['profile_group'] for r in records])
    assert np.isfinite(X).all() and len({r['id'] for r in records})==len(records)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    lr_meta=json.loads((project/'reports/logistic_regression/model_metadata.json').read_text())
    assert lr_meta['training_sha256']==sha and lr_meta['features']==features
    with (project/'outputs/predictions/logistic_regression/development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        lr_rows=list(csv.DictReader(f))
    lr_by_id={r['id']:r for r in lr_rows}
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    folds=np.zeros(len(y),dtype=int)
    for fold,(tr,va) in enumerate(cv,1):
        assert not set(groups[tr])&set(groups[va])
        assert not np.any(folds[va]); folds[va]=fold
        assert all(int(lr_by_id[records[i]['id']]['fold'])==fold for i in va)
    assert np.all(folds>0)
    pooled_by_model={'logistic_regression':lr_meta['pooled_oof_metrics']}
    probabilities={'logistic_regression':np.array([float(lr_by_id[r['id']]['predicted_score']) for r in records])}
    metric_keys=['accuracy','precision','recall','specificity','f1','roc_auc','average_precision']
    for name in MODEL_CONFIGS:
        print('START',name,flush=True)
        report=out/'reports'/name; preds=out/'outputs/predictions'/name
        model_dir=out/'models'/name; figs=out/'outputs/figures'/name
        for d in [report,preds,model_dir,figs]: d.mkdir(parents=True,exist_ok=True)
        oof=np.full(len(y),np.nan); fold_metrics=[]
        for fold,(tr,va) in enumerate(cv,1):
            pipeline=build_pipeline(name,features).fit(X[tr],y[tr])
            prob=pipeline.predict_proba(X[va])[:,1]; oof[va]=prob
            m=metrics(y[va],prob)
            train_p=pipeline.predict_proba(X[tr])[:,1]
            train_m=metrics(y[tr],train_p)
            fold_metrics.append({'fold':fold,'training_records':len(tr),'validation_records':len(va),
                'training_accuracy':train_m['accuracy'],'training_auc':train_m['roc_auc'],**m})
            print(name,'fold',fold,'AUC',round(m['roc_auc'],5),'Accuracy',round(m['accuracy'],5),flush=True)
        assert np.isfinite(oof).all()
        pooled=metrics(y,oof); pooled_by_model[name]=pooled; probabilities[name]=oof
        write_csv(report/'cv_fold_metrics.csv',fold_metrics)
        write_csv(report/'cv_summary.csv',[{'metric':key,'pooled_oof':pooled[key],
            'fold_mean':float(np.mean([m[key] for m in fold_metrics])),
            'fold_std_sample':float(np.std([m[key] for m in fold_metrics],ddof=1))} for key in metric_keys])
        output_rows=[{'id':r['id'],'profile_group':r['profile_group'],'fold':int(folds[i]),
             'actual_cardio':int(y[i]),'predicted_score':float(oof[i]),
             'predicted_cardio':int(oof[i]>=THRESHOLD)} for i,r in enumerate(records)]
        write_csv(preds/'development_oof_predictions.csv',output_rows)
        write_csv(report/'oof_confusion_matrix.csv',[
            {'actual_cardio':0,'predicted_0':pooled['tn'],'predicted_1':pooled['fp']},
            {'actual_cardio':1,'predicted_0':pooled['fn'],'predicted_1':pooled['tp']}])
        print('REFIT',name,flush=True)
        final=build_pipeline(name,features).fit(X,y)
        path=model_dir/'model_pipeline.joblib'
        joblib.dump({'pipeline':final,'features':features,'threshold':THRESHOLD,'target':'cardio',
             'feature_set':FEATURE_SET,'denormalization_params':params,
             'input_units':'original units; gender 1/2; cholesterol/gluc 1/2/3'},path,
             compress=('xz',3) if name=='random_forest' else 3)
        loaded=joblib.load(path)
        assert np.allclose(loaded['pipeline'].predict_proba(X[:100]),final.predict_proba(X[:100]))
        if name=='xgboost': final.named_steps['classifier'].save_model(model_dir/'xgboost_booster.json')
        metadata={'model':name,'training_source':str(DATA_DIR/'cardio_train_80.csv'),
            'training_sha256':sha,'records':len(y),'features':features,'parameters':MODEL_CONFIGS[name],
            'threshold':THRESHOLD,'cv':'5-fold StratifiedGroupKFold','random_state':SEED,
            'same_folds_as_logistic':True,'pooled_oof_metrics':pooled,'hyperparameter_tuning':False,
            'test_evaluated':False,'threshold_tuning':False,'early_stopping':False,
            'preprocessing':'Restored original units and verified by raw ID; same fixed one-hot encoding as LR; no scaling.',
            'versions':{'python':platform.python_version(),'numpy':np.__version__,
                         'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
            'mean_training_accuracy':float(np.mean([m['training_accuracy'] for m in fold_metrics]))}
        (report/'model_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
        fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.08,1,.92))
        fpr,tpr,_=roc_curve(y,oof); axes[0].plot(fpr,tpr,label=f'AUC={pooled["roc_auc"]:.4f}',color='#23658A')
        axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].legend(loc='lower right')
        axes[0].set(title=name+' out-of-fold ROC',xlabel='False positive rate',ylabel='Recall')
        cm=np.array([[pooled['tn'],pooled['fp']],[pooled['fn'],pooled['tp']]])
        axes[1].imshow(cm,cmap='Blues'); axes[1].set(xticks=[0,1],yticks=[0,1],
            xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'],title='OOF confusion matrix; threshold=0.5')
        for i in range(2):
            for j in range(2): axes[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()*.65 else '#26313c')
        fig.text(.01,.015,f'Development n={len(y):,} | Same grouped folds as LR | Test not evaluated',fontsize=10)
        fig.savefig(figs/'oof_evaluation.png',dpi=180,bbox_inches='tight'); fig.savefig(figs/'oof_evaluation.svg',bbox_inches='tight'); plt.close(fig)
        text=f'{name} 基线结果\n\n新版开发集{len(y):,}条；与逻辑回归相同10项特征、相同五折资料分组。\n固定初始参数：'+json.dumps(MODEL_CONFIGS[name],ensure_ascii=False)+'\n\n'
        text+='\n'.join(f'{key}: {pooled[key]:.6f}' for key in metric_keys)
        text+=f'\n\nTN={pooled["tn"]}, FP={pooled["fp"]}, FN={pooled["fn"]}, TP={pooled["tp"]}。\n'
        text+=f'各折训练准确率均值：{metadata["mean_training_accuracy"]:.2%}，验证准确率：{pooled["accuracy"]:.2%}。训练验证差距可用于检查过拟合。\n'
        text+='未调参、未调整门槛、未进行提前停止或最终测试评估；这是内部开发结果，不能认定最终胜者。\n模型已在全部开发集重拟合保存；该保存模型的训练分数不是独立评估成绩。\n'
        text+='输入连续字段恢复原单位，树模型不做缩放；类别字段与逻辑回归同样编码。\n新版外部train/test有同健康资料跨集的可能；本次内部交叉验证已强制分组。\n'
        text+='后续应适度调参后再比较；预测评分不是临床诊断或未来发病风险。\nChatGPT协助生成并运行代码，结果由本地计算和核验。\n'
        (report/'report_zh.txt').write_text(text,encoding='utf-8')
        b64=base64.b64encode((figs/'oof_evaluation.png').read_bytes()).decode()
        (report/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+name+'结果</title><style>body{max-width:1150px;margin:40px auto;font-family:-apple-system,Arial,sans-serif;line-height:1.8}img{width:100%}pre{white-space:pre-wrap;font-family:inherit}</style><h1>'+name+' 基线结果</h1><img src="data:image/png;base64,'+b64+'"><pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')
        with (preds/'development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
        assert len(saved)==len(y) and len({r['id'] for r in saved})==len(y)
        assert sum(int(r['actual_cardio'])==int(r['predicted_cardio']) for r in saved)/len(y)==pooled['accuracy']
        assert sum(int(r['actual_cardio'])==1 and int(r['predicted_cardio'])==1 for r in saved)==pooled['tp']
        (report/'validation.txt').write_text('PASS: source hash and feature list match current LR.\nPASS: all validation fold assignments exactly match LR OOF file.\nPASS: no profile group crosses training/validation folds.\nPASS: every record has one held-out prediction; exported counts reconcile.\nPASS: model reload reproduces predictions.\nPASS: test CSV not read; no tuning or early stopping.\n',encoding='utf-8')
        print('DONE',name,json.dumps(pooled),flush=True)
    comparison=out/'reports/model_comparison'; comparison.mkdir(parents=True,exist_ok=True)
    rows=[{'model':name,'records':len(y),'threshold':THRESHOLD,**values} for name,values in pooled_by_model.items()]
    write_csv(comparison/'baseline_comparison.csv',rows)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    for name,prob in probabilities.items():
        fpr,tpr,_=roc_curve(y,prob); ax.plot(fpr,tpr,label=f'{name}: {pooled_by_model[name]["roc_auc"]:.4f}')
    ax.plot([0,1],[0,1],'--',color='gray'); ax.legend(loc='lower right')
    ax.set(title='Baseline comparison: grouped out-of-fold ROC',xlabel='False positive rate',ylabel='Recall')
    fig.text(.01,.015,'Same training records, features and validation folds; no final test evaluation',fontsize=9)
    fig.savefig(comparison/'baseline_roc_comparison.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    table='<table border="1" cellpadding="9"><tr><th>Model</th>'+''.join('<th>'+k+'</th>' for k in metric_keys)+'</tr>'+''.join('<tr><td>'+name+'</td>'+''.join('<td>'+f'{values[k]:.4f}'+'</td>' for k in metric_keys)+'</tr>' for name,values in pooled_by_model.items())+'</table>'
    image=base64.b64encode((comparison/'baseline_roc_comparison.png').read_bytes()).decode()
    (comparison/'baseline_comparison.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三个基线模型比较</title><style>body{font-family:-apple-system,Arial,sans-serif;max-width:1150px;margin:40px auto;line-height:1.8}img{width:100%}table{border-collapse:collapse}</style><h1>三个基线模型比较</h1><p>同一开发数据、10项输入、五折分组；阈值0.5，均未调参，最终测试集未评估。</p>'+table+'<img src="data:image/png;base64,'+image+'"><p>当前比较仅用于初步筛选；固定初始参数与默认参数不是最佳配置，不应据此淘汰某一模型。</p></html>',encoding='utf-8')
    print('ALL BASELINES COMPLETE',flush=True)

if __name__=='__main__': main()
