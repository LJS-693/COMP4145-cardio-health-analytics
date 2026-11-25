"""Full-feature logistic baseline with five-fold grouped out-of-fold evaluation.

Uses Caroline's current training split only. Reverses the previously applied
MinMax mapping, verifies against raw records by ID, and fits scaling per fold.
Run: python3 src/train_logistic.py
"""
import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SET = 'complete_logistic_bmi_only'
SEED = 4145
THRESHOLD = .5
DATA_DIR = Path('data/processed/caroline_cleaned version')


def decode_records(source, params):
    """Adapt current normalized/one-hot CSV to original-unit model inputs."""
    with source.open(encoding='utf-8-sig', newline='') as f:
        encoded = list(csv.DictReader(f))
    decoded = []
    for r in encoded:
        out = {'id': r['id']}
        if 'cardio' in r:
            out['cardio'] = int(r['cardio'])
        for field, bounds in params.items():
            value = float(r[field])
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite value in {field}')
            out[field] = value * (bounds['max']-bounds['min']) + bounds['min']
        for prefix in ['cholesterol', 'gluc']:
            values = [str(r[f'{prefix}_{level}']).lower() in ('true', '1', '1.0') for level in [1,2,3]]
            assert sum(values) == 1, f'Invalid one-hot group: {prefix}'
            out[prefix] = values.index(True)+1
        for field in ['gender', 'smoke', 'alco', 'active']:
            value = int(r[field]); assert value in [0,1]
            out[field] = value + 1 if field == 'gender' else value
        decoded.append(out)
    return decoded


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'specificity':float(tn/(tn+fp)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def build_pipeline(features):
    numeric = ['age_years','bmi','ap_hi','ap_lo']
    nominal = ['gender','cholesterol','gluc']
    binary = ['smoke','alco','active']
    prep = ColumnTransformer([
        ('numeric',StandardScaler(),[features.index(c) for c in numeric]),
        ('categories',OneHotEncoder(categories=[[1,2],[1,2,3],[1,2,3]],
                                   drop='first',handle_unknown='error',sparse_output=False),
         [features.index(c) for c in nominal]),
        ('binary','passthrough',[features.index(c) for c in binary])],remainder='drop')
    return Pipeline([('preprocess',prep),('classifier',LogisticRegression(
        C=1.0,solver='lbfgs',max_iter=1000,random_state=SEED))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    project=args.project_root
    out=args.output_root or project
    report=out/'reports/logistic_regression'
    predictions=out/'outputs/predictions/logistic_regression'
    figures=out/'outputs/figures/logistic_regression'
    models=out/'models/logistic_regression'
    for folder in [report,predictions,figures,models]:
        folder.mkdir(parents=True,exist_ok=True)
    config_path=project/'config/feature_sets.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    features=config[FEATURE_SET]
    assert not set(features)&set(config['exclude_from_X'])
    source=project/DATA_DIR/'cardio_train_80.csv'
    params_path=project/DATA_DIR/'denormalization_params.json'
    params=json.loads(params_path.read_text(encoding='utf-8'))
    records=decode_records(source,params)
    # Check restoration against original observations, but never use held-out
    # records for scaling, grouping decisions, modelling, or evaluation.
    raw_path=project/'data/raw/cardio_train.csv'
    with raw_path.open(encoding='utf-8-sig',newline='') as f:
        raw_by_id={r['id']:r for r in csv.DictReader(f,delimiter=';')}
    profile_cols=['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    max_restore_error=0.
    for r in records:
        raw=raw_by_id[r['id']]
        expected={c:float(raw[c]) for c in ['height','weight','ap_hi','ap_lo']}
        expected['age_years']=float(raw['age'])/365.25
        expected['bmi']=float(raw['weight'])/(float(raw['height'])/100)**2
        expected['pulse_pressure']=float(raw['ap_hi'])-float(raw['ap_lo'])
        expected['map']=float(raw['ap_lo'])+expected['pulse_pressure']/3
        for c,v in expected.items():
            max_restore_error=max(max_restore_error,abs(r[c]-v))
            assert abs(r[c]-v)<1e-8, (r['id'],c)
            r[c]=v  # Canonical original units; no global scaling retained.
        for c in ['gender','cholesterol','gluc','smoke','alco','active','cardio']:
            assert r[c]==int(raw[c])
        signature=json.dumps([float(raw[c]) for c in profile_cols],separators=(',',':'))
        r['profile_group']=hashlib.sha256(signature.encode()).hexdigest()
    X=np.array([[float(r[c]) for c in features] for r in records],dtype=float)
    y=np.array([int(r[config['target']]) for r in records])
    groups=np.array([r[config['cross_validation_group']] for r in records])
    assert np.isfinite(X).all() and set(y)=={0,1}
    assert len({r['id'] for r in records})==len(records)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    oof=np.full(len(y),np.nan)
    base_oof=np.full(len(y),np.nan)
    fold_id=np.zeros(len(y),dtype=int)
    fold_rows=[]
    convergence=[]
    for fold,(tr,va) in enumerate(cv.split(X,y,groups),1):
        assert not set(groups[tr])&set(groups[va])
        assert not np.any(fold_id[va])
        model=build_pipeline(features)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(X[tr],y[tr])
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise RuntimeError(f'Fold {fold} did not converge')
        scaler=model.named_steps['preprocess'].named_transformers_['numeric']
        assert np.allclose(scaler.mean_,X[tr][:,[features.index(c) for c in ['age_years','bmi','ap_hi','ap_lo']]].mean(axis=0))
        p=model.predict_proba(X[va])[:,1]
        dummy=DummyClassifier(strategy='most_frequent').fit(X[tr],y[tr])
        bp=dummy.predict_proba(X[va])[:,1]
        oof[va],base_oof[va],fold_id[va]=p,bp,fold
        for name,prob in [('logistic_regression',p),('majority_baseline',bp)]:
            fold_rows.append({'model':name,'fold':fold,'training_records':len(tr),
                'validation_records':len(va),'validation_positive_rate':float(y[va].mean()),
                **metrics(y[va],prob)})
        convergence.append({'fold':fold,'iterations':int(model.named_steps['classifier'].n_iter_[0])})
        print(f'Fold {fold}: {metrics(y[va],p)}',flush=True)
    assert np.isfinite(oof).all() and np.all(fold_id>0)
    group_folds={}
    for group,fold in zip(groups,fold_id):
        if group in group_folds:
            assert group_folds[group]==fold
        group_folds[group]=fold
    pooled=metrics(y,oof)
    baseline=metrics(y,base_oof)
    summary_rows=[]
    keys=['accuracy','precision','recall','specificity','f1','roc_auc','average_precision']
    for name,pooled_metrics in [('logistic_regression',pooled),('majority_baseline',baseline)]:
        for key in keys:
            values=[r[key] for r in fold_rows if r['model']==name]
            summary_rows.append({'model':name,'metric':key,'pooled_oof':pooled_metrics[key],
                'fold_mean':float(np.mean(values)),'fold_std_sample':float(np.std(values,ddof=1))})
    write_csv(report/'cv_fold_metrics.csv',fold_rows)
    write_csv(report/'cv_summary.csv',summary_rows)
    write_csv(predictions/'development_oof_predictions.csv',[
        {'id':r['id'],'profile_group':r['profile_group'],'fold':int(fold_id[i]),
         'actual_cardio':int(y[i]),'predicted_score':float(oof[i]),
         'predicted_cardio':int(oof[i]>=THRESHOLD),
         'baseline_score':float(base_oof[i]),'baseline_prediction':int(base_oof[i]>=THRESHOLD)}
        for i,r in enumerate(records)])
    write_csv(report/'oof_confusion_matrix.csv',[
        {'actual_cardio':0,'predicted_0':pooled['tn'],'predicted_1':pooled['fp']},
        {'actual_cardio':1,'predicted_0':pooled['fn'],'predicted_1':pooled['tp']}])

    # Refit once on the complete development split for later final testing.
    final_model=build_pipeline(features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        final_model.fit(X,y)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    model_path=models/'logistic_pipeline.joblib'
    joblib.dump({'pipeline':final_model,'features':features,'threshold':THRESHOLD,
                 'target':config['target'],'feature_set':FEATURE_SET,
                 'input_units':'original units; gender uses original 1/2 codes',
                 'denormalization_params':params,
                 'training_source':str(DATA_DIR/'cardio_train_80.csv')},model_path)
    loaded=joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].predict_proba(X[:100]),final_model.predict_proba(X[:100]))
    names=['age_years_per_SD','bmi_per_SD','ap_hi_per_SD','ap_lo_per_SD',
           'gender_2_vs_1','cholesterol_2_vs_1','cholesterol_3_vs_1',
           'gluc_2_vs_1','gluc_3_vs_1','smoke_1_vs_0','alco_1_vs_0','active_1_vs_0']
    coefs=final_model.named_steps['classifier'].coef_[0]
    assert len(names)==len(coefs)
    write_csv(report/'coefficients.csv',[
        {'term':name,'coefficient':float(c),'conditional_odds_ratio':float(np.exp(c)),
         'interpretation':'conditional association; numeric terms per training SD; not causal effects'}
        for name,c in zip(names,coefs)])
    write_csv(report/'scaling_parameters.csv',[
        {'field':field,'development_mean':float(mean),'development_scale_sd':float(sd)}
        for field,mean,sd in zip(['age_years','bmi','ap_hi','ap_lo'],
            final_model.named_steps['preprocess'].named_transformers_['numeric'].mean_,
            final_model.named_steps['preprocess'].named_transformers_['numeric'].scale_)])

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                          'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fpr,tpr,_=roc_curve(y,oof)
    axes[0].plot(fpr,tpr,color='#23658A',lw=2,label=f'Logistic AUC={pooled["roc_auc"]:.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Chance / majority baseline')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate (recall)',title='Out-of-fold ROC curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=.2)
    cm=np.array([[pooled['tn'],pooled['fp']],[pooled['fn'],pooled['tp']]])
    axes[1].imshow(cm,cmap='Blues')
    axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Predicted 0','Predicted 1'],
                yticklabels=['Actual 0','Actual 1'],title='Out-of-fold confusion matrix (threshold=0.5)')
    for i in range(2):
        for j in range(2):
            axes[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=18,
                         color='white' if cm[i,j]>cm.max()*.65 else '#26313c')
    fig.text(.01,.015,f'5-fold StratifiedGroupKFold | Development n={len(y):,} | Test set not evaluated',fontsize=10,color='#666666')
    fig.savefig(figures/'logistic_oof_evaluation.png',dpi=180,bbox_inches='tight')
    fig.savefig(figures/'logistic_oof_evaluation.svg',bbox_inches='tight')
    plt.close(fig)

    metadata={'feature_set':FEATURE_SET,'features':features,'records':len(y),
        'target':'cardio','threshold':THRESHOLD,'cv':'5-fold StratifiedGroupKFold',
        'random_state':SEED,'pooled_oof_metrics':pooled,'majority_baseline_metrics':baseline,
        'fold_convergence':convergence,'final_refit_iterations':int(final_model.named_steps['classifier'].n_iter_[0]),
        'parameters':{'solver':'lbfgs','C':1.0,'max_iter':1000,'class_weight':None},
        'categorical_encoding':'gender, cholesterol and gluc: one-hot; reference level 1',
        'scaling':'age_years,bmi,ap_hi,ap_lo standardized inside each training fold',
        'training_source':str(DATA_DIR/'cardio_train_80.csv'),
        'data_version':'Caroline cleaned version; latest remote at retraining',
        'input_adapter':'Reverse stored MinMax mapping and decode one-hot columns; verify every training row against raw ID; use canonical original values.',
        'max_denormalization_error':max_restore_error,
        'denormalization_sha256':hashlib.sha256(params_path.read_bytes()).hexdigest(),
        'raw_source_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'training_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'feature_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'test_evaluated':False,'hyperparameter_tuning':False,'threshold_tuning':False,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
                    'sklearn':sklearn.__version__,'joblib':joblib.__version__}}
    (report/'model_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['逻辑回归基线模型报告','',
        f'开发集：{len(y):,}条；特征方案：{FEATURE_SET}；判断门槛：0.5。',
        '输入：'+', '.join(features),
        '输出：疾病状态标签cardio的模型概率估计与0/1分类；不是未来发病风险。','',
        '验证方法：五折StratifiedGroupKFold，按profile_group分组，尽量保持类别比例。',
        '使用新版cardio_train_80.csv，旧版结果已替换。',
        '连续字段先反归一化，逐条核对原始ID及测量值，再用原单位数据在训练折内拟合StandardScaler。',
        '不会沿用全数据拟合的MinMax映射作为训练预处理。',
        '新版one-hot胆固醇与血糖还原为1/2/3；性别0/1转回原始1/2，以保持原10项输入方案。',
        '每条记录的OOF预测都来自未用该记录或相同profile_group训练的模型。',
        '标准化只在每折训练部分拟合；类别编码采用预先定义的类别集合。',
        '没有调参、调整门槛、重采样或测试集评估。','',
        '合并全部OOF预测的结果（不是最终测试集结果）：']
    for key in keys:
        line=next(r for r in summary_rows if r['model']=='logistic_regression' and r['metric']==key)
        lines.append(f'{key}: {pooled[key]:.6f}；五折均值 {line["fold_mean"]:.6f}，折间样本标准差 {line["fold_std_sample"]:.6f}')
    lines+=['',f'混淆矩阵：TN={pooled["tn"]:,}, FP={pooled["fp"]:,}, FN={pooled["fn"]:,}, TP={pooled["tp"]:,}。',
        f'多数类别基线：Accuracy={baseline["accuracy"]:.2%}，ROC-AUC={baseline["roc_auc"]:.3f}，Recall={baseline["recall"]:.2%}。',
        f'逻辑回归相对基线的Accuracy差异：{(pooled["accuracy"]-baseline["accuracy"])*100:.2f}个百分点。','',
        '模型文件：保存了在全部开发数据上重拟合的预处理Pipeline和逻辑回归，用于后续最终评估；重拟合模型本身没有独立测试成绩。',
        '系数属于全开发集重拟合模型；条件优势比不是患病概率倍数，也不是因果效应。连续字段系数对应每一个标准差变化。',
        'gender仅比较编码2与1；随附README没有明确性别映射，未猜测男女标签。',
        '血压、胆固醇等与疾病标签存在关联，不能据此证明临床筛查有效性。',
        '结果只适用于通过既定清洗规则的记录；保留的极端BMI可能影响线性模型稳定性。',
        '五折标准差仅表示折间波动，不是置信区间。描述性分析曾看过整个开发集，因此本次属于内部开发验证。','',
        '当前新版外部train/test是普通分层随机划分，不保证同健康资料组合不跨两集；内部交叉验证仍强制分组。',
        '后续：在相同交叉验证折上比较基础特征与树模型；最终方案确定后再使用cardio_test_20.csv。','',
        '复现：python3 -m pip install -r requirements_logistic.txt；python3 src/train_logistic.py。',
        '模型加载：bundle = joblib.load("models/logistic_regression/logistic_pipeline.joblib")；',
        '按bundle["features"]顺序建立数值矩阵X，再用bundle["pipeline"].predict_proba(X)[:,1]。','',
        '来源：https://github.com/caravanuden/cardio',
        '方法：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html',
        '模型：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
        'ChatGPT使用：协助编写训练程序、运行验证、绘图和解释结果。']
    report_text='\n'.join(lines)+'\n'
    (report/'logistic_report_zh.txt').write_text(report_text,encoding='utf-8')
    img=base64.b64encode((figures/'logistic_oof_evaluation.png').read_bytes()).decode()
    (report/'logistic_report.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逻辑回归基线模型结果</title>'
        '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1150px;margin:40px auto;padding:20px;color:#26313c;line-height:1.8}img{width:100%}pre{white-space:pre-wrap;font-family:inherit}</style>'
        '<h1>逻辑回归基线模型结果</h1><img alt="OOF ROC and confusion matrix" src="data:image/png;base64,'+img+'"><pre>'+html.escape(report_text)+'</pre></html>',encoding='utf-8')

    # Re-load exported predictions and independently reconcile their classification counts.
    with (predictions/'development_oof_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        saved=list(csv.DictReader(f))
    assert len(saved)==len(records)
    observed={(a,p):sum(int(r['actual_cardio'])==a and int(r['predicted_cardio'])==p for r in saved)
              for a in [0,1] for p in [0,1]}
    assert [observed[(0,0)],observed[(0,1)],observed[(1,0)],observed[(1,1)]]==[pooled[k] for k in ['tn','fp','fn','tp']]
    assert all(0<=float(r['predicted_score'])<=1 for r in saved)
    (report/'validation.txt').write_text(
        'PASS: current cardio_train_80.csv used; test CSV not opened or evaluated.\n'
        'PASS: normalized values restored and verified against raw source by ID.\n'
        'PASS: feature whitelist excludes target and metadata.\n'
        'PASS: each development record receives exactly one held-out prediction.\n'
        'PASS: no predictor-profile group overlaps between training/validation folds.\n'
        'PASS: fold scalers match training-fold means; all six fits converged.\n'
        'PASS: exported prediction counts reconcile with confusion matrix.\n'
        'PASS: saved model reload reproduces predictions.\n',encoding='utf-8')
    print('POOLED_OOF',json.dumps(pooled),flush=True)
    print('Saved model:',model_path,flush=True)


if __name__=='__main__':
    main()
