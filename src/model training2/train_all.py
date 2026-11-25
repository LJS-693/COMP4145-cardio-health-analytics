"""Train the complete seven-feature model training2 protocol.

Folders: baseline (3 models), round1 (3 searches), round2 (2 searches),
round3 (XGBoost multi-metric search). Historical training1 is read-only.
"""
import argparse
import base64
import html
import json
import platform
import shutil
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer,roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import (FEATURES,NUMERIC,CATEGORIES,METRICS,PRIMARY,sha256,read_csv,
    write_csv,load_development,metrics,scorer_value)

ROOT=Path(__file__).resolve().parents[2]
RUN='model training2'
MODELS=['logistic_regression','random_forest','xgboost']
TITLES={'logistic_regression':'Logistic regression','random_forest':'Random forest','xgboost':'XGBoost'}


def build_pipeline(name,parameters):
    preprocessing=ColumnTransformer([
        ('numeric',StandardScaler() if name=='logistic_regression' else 'passthrough',[FEATURES.index(c) for c in NUMERIC]),
        ('categories',OneHotEncoder(categories=[[1,2],[1,2,3],[1,2,3]],drop='first',handle_unknown='error',sparse_output=False),
         [FEATURES.index(c) for c in CATEGORIES])])
    constructor={'logistic_regression':LogisticRegression,'random_forest':RandomForestClassifier,'xgboost':xgboost.XGBClassifier}[name]
    return Pipeline([('preprocess',preprocessing),('classifier',constructor(**parameters))])


def select_auc(results):return int(np.argmax(results['mean_test_roc_auc']))


def ranks(results):return np.mean([results['rank_test_'+m] for m in PRIMARY],axis=0)


def select_multi(results):
    mean_rank=ranks(results)
    return min(range(len(mean_rank)),key=lambda i:(float(mean_rank[i]),-float(results['mean_test_roc_auc'][i]),
        int(results['params'][i]['classifier__n_estimators']),float(results['params'][i]['classifier__learning_rate']),i))


def fit_checked(model,X,y,name):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y)
    assert np.array_equal(model.classes_,[0,1])
    if name=='logistic_regression':
        scaler=model.named_steps['preprocess'].named_transformers_['numeric']
        assert np.allclose(scaler.mean_,X[:,[FEATURES.index(c) for c in NUMERIC]].mean(axis=0))
    return model


def table_html(rows):
    headers=['model','stage','roc_auc','accuracy','precision','recall','f1']
    return '<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(r[h]))+'</td>' if isinstance(r[h],str) else '<td>'+f'{r[h]:.6f}'+'</td>' for h in headers)+'</tr>' for r in rows)+'</table>'


def save_html(path,title,text,images=(),table=''):
    pictures=''.join('<img alt="Model evaluation" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">' for p in images)
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
        '<style>body{font-family:-apple-system,Arial,sans-serif;max-width:1200px;margin:36px auto;padding:16px;line-height:1.8;color:#203040}'
        'img{width:100%}pre{white-space:pre-wrap;font-family:inherit}table{width:100%;border-collapse:collapse;font-size:14px}'
        'td,th{border:1px solid #ddd;padding:8px}th{background:#edf3f7}</style><h1>'+html.escape(title)+'</h1>'+table+pictures+'<pre>'+html.escape(text)+'</pre></html>',encoding='utf-8')


def stage_paths(out,stage,name):
    return {key:out/base/RUN/stage/name for key,base in [('report','reports'),('model','models'),('figures','outputs'),('prediction','outputs')]}


def run_stage(out,stage,name,parameters,grid,previous,context,plan,resume=False):
    X,y,records,adapter,cv,fold_ids,provenance=context
    report=out/'reports'/RUN/stage/name
    model_dir=out/'models'/RUN/stage/name
    figures=out/'outputs'/RUN/'figures'/stage/name
    predictions=out/'outputs'/RUN/'predictions'/stage/name
    metadata_path=report/'model_metadata.json'
    if resume and metadata_path.exists() and (report/'validation.txt').exists():
        meta=json.loads(metadata_path.read_text())
        assert meta['training_sha256']==provenance['training_sha256'] and meta['features']==FEATURES
        assert meta['training_plan_sha256']==sha256(out/'config'/RUN/'training_plan.json')
        saved=read_csv(predictions/'development_oof_predictions.csv')
        assert len(saved)==len(y) and all(saved[i]['id']==records[i]['id'] for i in range(len(y)))
        print('RESUME completed stage:',stage,name,flush=True)
        return {'metadata':meta,'probabilities':np.array([float(r['predicted_score']) for r in saved])}
    for folder in [report,model_dir,figures,predictions]:
        if folder.exists():raise FileExistsError(f'Output already exists: {folder}; use a fresh output or resume completed stages.')
    print('START',stage,name,'parameters=',json.dumps(parameters),flush=True)
    start=time.monotonic()
    base=build_pipeline(name,parameters)
    search=None;search_rows=[];all_grid_folds=[];reference_reproduced=False
    selected_parameters=parameters.copy()
    if grid:
        scoring={m:make_scorer(scorer_value,response_method='predict_proba',metric=m) for m in METRICS}
        prefixed={'classifier__'+k:v for k,v in grid.items()}
        search=GridSearchCV(base,prefixed,scoring=scoring,refit=select_multi if stage=='round3' else select_auc,
            cv=cv,n_jobs=1,return_train_score=True,error_score='raise',verbose=2)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            search.fit(X,y)
        result=search.cv_results_;best=int(search.best_index_)
        selected_parameters.update({k.removeprefix('classifier__'):v for k,v in search.best_params_.items()})
        mean_ranks=ranks(result)
        for i,p in enumerate(result['params']):
            values={'candidate_index':i+1,**{k.removeprefix('classifier__'):v for k,v in p.items()},
                    'selected':i==best,'rank_auc':int(result['rank_test_roc_auc'][i]),'mean_primary_rank':float(mean_ranks[i]),
                    'mean_fit_seconds':float(result['mean_fit_time'][i])}
            for metric in METRICS:
                values['rank_'+metric]=int(result['rank_test_'+metric][i])
                for phase,label in [('train','training'),('test','validation')]:
                    values[f'{label}_{metric}_mean']=float(result[f'mean_{phase}_{metric}'][i])
                    values[f'{label}_{metric}_std_population']=float(result[f'std_{phase}_{metric}'][i])
            search_rows.append(values)
            for fold in range(5):
                all_grid_folds.append({'candidate_index':i+1,'fold':fold+1,
                    **{k.removeprefix('classifier__'):v for k,v in p.items()},
                    **{'validation_'+m:float(result[f'split{fold}_test_{m}'][i]) for m in METRICS},
                    **{'training_'+m:float(result[f'split{fold}_train_{m}'][i]) for m in METRICS}})
        if previous:
            control=next((i for i,p in enumerate(result['params']) if all(parameters[k.removeprefix('classifier__')]==v for k,v in p.items())),None)
            if control is not None:
                for metric in METRICS:
                    assert np.isclose(result['mean_test_'+metric][control],previous['metadata']['cv_fold_means'][metric],atol=1e-10,rtol=0)
                reference_reproduced=True
        final=search.best_estimator_
        print('SELECTED',stage,name,json.dumps(selected_parameters),flush=True)
    oof=np.full(len(y),np.nan);fold_rows=[]
    for fold,(train,valid) in enumerate(cv,1):
        model=fit_checked(build_pipeline(name,selected_parameters),X[train],y[train],name)
        probability=model.predict_proba(X[valid])[:,1]
        assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
        oof[valid]=probability
        score=metrics(y[valid],probability)
        train_score=metrics(y[train],model.predict_proba(X[train])[:,1])
        row={'fold':fold,'training_records':len(train),'validation_records':len(valid),
             **score,**{'training_'+m:train_score[m] for m in METRICS}}
        fold_rows.append(row)
        if search:
            for metric in METRICS:
                assert np.isclose(score[metric],result[f'split{fold-1}_test_{metric}'][best],atol=1e-10,rtol=0)
                assert np.isclose(train_score[metric],result[f'split{fold-1}_train_{metric}'][best],atol=1e-10,rtol=0)
        print('OOF',stage,name,fold,'AUC',round(score['roc_auc'],6),'Acc',round(score['accuracy'],6),flush=True)
    assert np.isfinite(oof).all()
    means={m:float(np.mean([r[m] for r in fold_rows])) for m in METRICS}
    pooled=metrics(y,oof)
    if not search:
        print('REFIT',stage,name,flush=True)
        final=fit_checked(build_pipeline(name,selected_parameters),X,y,name)
    assert final.named_steps['preprocess'].transform(X[:2]).shape[1]==9
    if name=='logistic_regression':
        assert np.allclose(final.named_steps['preprocess'].named_transformers_['numeric'].mean_,X[:,[FEATURES.index(c) for c in NUMERIC]].mean(axis=0))
    for folder in [report,model_dir,figures,predictions]:folder.mkdir(parents=True)
    bundle={'pipeline':final,'features':FEATURES,'threshold':.5,'target':'cardio','feature_set':'seven_features',
            'denormalization_params':adapter,'input_units':'original units; original gender 1/2 and cholesterol/gluc 1/2/3',
            'parameters':selected_parameters,'run':RUN,'stage':stage}
    model_path=model_dir/'model_pipeline.joblib'
    joblib.dump(bundle,model_path,compress=('xz',3))
    loaded=joblib.load(model_path)
    assert np.allclose(loaded['pipeline'].predict_proba(X[:100]),final.predict_proba(X[:100]))
    for key,value in selected_parameters.items():assert loaded['pipeline'].named_steps['classifier'].get_params()[key]==value
    if name=='xgboost':
        booster_path=model_dir/'xgboost_booster.json'
        final.named_steps['classifier'].save_model(booster_path)
        native=xgboost.XGBClassifier();native.load_model(booster_path)
        assert np.allclose(native.predict_proba(final.named_steps['preprocess'].transform(X[:100])),final.predict_proba(X[:100]))
    if name=='logistic_regression':
        scaler=final.named_steps['preprocess'].named_transformers_['numeric']
        write_csv(report/'scaling_parameters.csv',[{'feature':c,'mean':float(a),'scale':float(b)} for c,a,b in zip(NUMERIC,scaler.mean_,scaler.scale_)])
    write_csv(report/'cv_fold_metrics.csv',fold_rows)
    write_csv(report/'cv_summary.csv',[{'metric':m,'fold_mean':means[m],'fold_std_sample':float(np.std([r[m] for r in fold_rows],ddof=1)),
                                       'pooled_oof':pooled[m]} for m in METRICS])
    write_csv(predictions/'development_oof_predictions.csv',[{'id':r['id'],'profile_group':r['profile_group'],'fold':int(fold_ids[i]),
        'actual_cardio':int(y[i]),'predicted_score':float(oof[i]),'predicted_cardio':int(oof[i]>=.5)} for i,r in enumerate(records)])
    write_csv(report/'oof_confusion_matrix.csv',[{'actual_cardio':0,'predicted_0':pooled['tn'],'predicted_1':pooled['fp']},
        {'actual_cardio':1,'predicted_0':pooled['fn'],'predicted_1':pooled['tp']}])
    full_public=final.named_steps['classifier'].get_params(deep=False)
    write_csv(report/'parameters.csv',[{'parameter':k,'value':'None' if v is None else str(v),
        'source':'explicit selected parameter' if k in selected_parameters else 'estimator default; None may delegate to backend'} for k,v in sorted(full_public.items())])
    if search:
        write_csv(report/'gridsearch_results.csv',search_rows)
        write_csv(report/'all_grid_fold_metrics.csv',all_grid_folds)
        winners=[]
        for metric in METRICS:
            maximum=max(r['validation_'+metric+'_mean'] for r in search_rows)
            for r in search_rows:
                if r['validation_'+metric+'_mean']==maximum:
                    winners.append({'metric':metric,'candidate_index':r['candidate_index'],
                        'score':maximum,'parameters':json.dumps({k:r[k] for k in grid}), 'selected':r['selected']})
        write_csv(report/'metric_winners.csv',winners)
    if previous:
        write_csv(report/'change_vs_previous.csv',[{'metric':m,'previous':previous['metadata']['cv_fold_means'][m],
            'current':means[m],'difference':means[m]-previous['metadata']['cv_fold_means'][m]} for m in METRICS])
    metadata={'run':RUN,'stage':stage,'model':name,**provenance,'features':FEATURES,'encoded_feature_count':9,
        'parameters':selected_parameters,'parameter_grid':grid,'selection_rule':plan['selection'][stage],
        'threshold':.5,'classification_rule':'predict_proba[:,1] >= 0.5','threshold_tuning':False,'early_stopping':False,
        'cv_fold_means':means,'pooled_oof_metrics':pooled,
        'mean_training_accuracy':float(np.mean([r['training_accuracy'] for r in fold_rows])),
        'mean_training_auc':float(np.mean([r['training_roc_auc'] for r in fold_rows])),
        'candidate_count':len(search_rows),'grid_cv_fits':len(search_rows)*5,'selected_oof_fits':5,'full_development_refits':1,
        'previous_control_reproduced':reference_reproduced,'test_evaluated':False,'test_csv_read':False,'fully_cleaned_csv_read':False,
        'training_plan_sha256':sha256(out/'config'/RUN/'training_plan.json'),
        'training_code_sha256':sha256(Path(__file__)),'data_utils_sha256':sha256(Path(__file__).parent/'data_utils.py'),
        'previous_stage':previous['metadata']['stage'] if previous else None,
        'previous_parameters':previous['metadata']['parameters'] if previous else None,
        'seconds':time.monotonic()-start,'limitations':plan['limits'],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__}}
    if name=='logistic_regression':metadata['full_refit_iterations']=final.named_steps['classifier'].n_iter_.tolist()
    if stage=='round3':metadata['selected_mean_primary_rank']=float(ranks(result)[best])
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained');fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fpr,tpr,_=roc_curve(y,oof)
    axes[0].plot(fpr,tpr,label=f'Pooled OOF AUC={pooled["roc_auc"]:.4f}',lw=2)
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].legend(loc='lower right')
    axes[0].set(title=TITLES[name]+' / '+stage,xlabel='False positive rate',ylabel='Recall')
    cm=np.array([[pooled['tn'],pooled['fp']],[pooled['fn'],pooled['tp']]])
    axes[1].imshow(cm,cmap='Blues');axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'],title='OOF confusion matrix; threshold=0.5')
    for i in range(2):
        for j in range(2):axes[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()*.65 else '#203040')
    fig.text(.01,.015,'7 original inputs / 9 encoded columns | Same grouped development folds | Test not evaluated',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(figures/('oof_evaluation.'+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)
    pictures=[figures/'oof_evaluation.png']
    if search:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for metric in PRIMARY:
            if metric in ['roc_auc','accuracy','f1']:ax.plot(range(1,len(search_rows)+1),[r['validation_'+metric+'_mean'] for r in search_rows],marker='o',label=metric)
        ax.axvline(best+1,color='#777777',ls='--',label='Selected candidate')
        ax.set(xticks=range(1,len(search_rows)+1),ylim=(0,1),xlabel='Candidate index (see CSV for parameters)',ylabel='Five-fold validation mean',title=TITLES[name]+' / '+stage+' grid search')
        ax.legend(loc='lower right');fig.savefig(figures/'gridsearch.png',dpi=160,bbox_inches='tight');plt.close(fig)
        pictures.append(figures/'gridsearch.png')
    text='\n'.join([f'{RUN} / {stage} / {name}','',f'七项特征：{", ".join(FEATURES)}。开发记录：{len(y):,}。',
        '参数：'+json.dumps(selected_parameters),'选择规则：'+plan['selection'][stage],
        '五折验证均值：',*[f'{m}: {means[m]:.6f}' for m in METRICS],
        '',f'各折训练准确率均值：{metadata["mean_training_accuracy"]:.3%}。',
        '所有阶段保持原始资料分组及同样五折；标准化仅用于逻辑回归且在训练折内拟合。',
        '网格中的 test 字样表示验证折；并非最终测试集。表格用折均值，ROC 图用合并 OOF AUC。',
        '选中配置的五折预测重新生成并与网格逐折核对；不是独立的新验证。',
        '分类阈值固定 0.5；未做阈值优化、提前停止或类别重采样。',
        '本次从七项输入重新选择每轮参数，没有照搬十项实验的最优解。',
        '保存模型在全部开发集重拟合；模型重载及原生 booster 预测已核对（如适用）。',
        '此前已查看全量描述图，且重复使用开发折选参，因此结果属于探索性开发比较；未做嵌套 CV。',
        '未读取最终测试 CSV 或全量清洗 CSV；不改动 model training1。',
        '预测目标是现有 cardio 状态标签；评分未被验证为临床紧急程度或未来发病概率。',
        'ChatGPT 协助编写与运行；所有统计由本地计算并核验。'])+'\n'
    (report/'report_zh.txt').write_text(text,encoding='utf-8')
    save_html(report/'report.html',f'{stage} / {TITLES[name]}',text,pictures)
    saved=read_csv(predictions/'development_oof_predictions.csv')
    assert len(saved)==len(y) and len({r['id'] for r in saved})==len(y)
    assert all(r['id']==records[i]['id'] and int(r['fold'])==fold_ids[i] for i,r in enumerate(saved))
    assert metrics(np.array([int(r['actual_cardio']) for r in saved]),np.array([float(r['predicted_score']) for r in saved]))==pooled
    (report/'validation.txt').write_text('PASS: canonical data, seven-feature whitelist, same profile groups and folds.\nPASS: fold metrics, OOF exports and saved models reconcile.\nPASS: selected parameters inherited from the immediately preceding seven-feature stage.\nPASS: LR scaling uses training fold only; all fits converged.\nPASS: no final test data read or evaluated; training1 preserved.\n',encoding='utf-8')
    print('DONE',stage,name,json.dumps(means),flush=True)
    return {'metadata':metadata,'probabilities':oof}


def summarize(out,states,context,plan):
    X,y,records,adapter,cv,fold_ids,provenance=context
    report=out/'reports'/RUN;figures=out/'outputs'/RUN/'figures'/'overall'
    report.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    summary=[];parameters=[];fold_rows=[];all_candidates=[];capacity=[]
    for (stage,name),state in states.items():
        meta=state['metadata'];m=meta['cv_fold_means']
        summary.append({'model':name,'stage':stage,'features':7,'training_accuracy':meta['mean_training_accuracy'],
            **m,'selection_rule':meta['selection_rule'],'parameters':json.dumps(meta['parameters'])})
        parameters.extend({'stage':stage,'model':name,**r} for r in read_csv(report/stage/name/'parameters.csv'))
        fold_rows.extend({'stage':stage,'model':name,**r} for r in read_csv(report/stage/name/'cv_fold_metrics.csv'))
        grid_path=report/stage/name/'gridsearch_results.csv'
        if grid_path.exists():
            for row in read_csv(grid_path):
                all_candidates.append({'stage':stage,'model':name,'candidate_index':int(row['candidate_index']),
                    'parameters':json.dumps({k:row[k] for k in plan[stage][name]}),'selected':row['selected'],
                    'rank_auc':int(row['rank_auc']),'mean_primary_rank':float(row['mean_primary_rank']),
                    **{f'{phase}_{metric}_{stat}':float(row[f'{phase}_{metric}_{stat}'])
                       for phase in ['training','validation'] for metric in METRICS for stat in ['mean','std_population']}})
        probability=state['probabilities']
        for fraction in [.1,.2,.3,.5]:
            mask=np.zeros(len(y),dtype=bool)
            for _,valid in cv:
                cutoff=np.sort(probability[valid])[::-1][int(np.ceil(len(valid)*fraction))-1]
                mask[valid]=probability[valid]>=cutoff
            selected=int(mask.sum());positive=int(y[mask].sum());precision=positive/selected
            capacity.append({'stage':stage,'model':name,'nominal_top_fraction':fraction,'selected_records':selected,
                'actual_fraction':selected/len(y),'positive_selected':positive,'precision_at_capacity':precision,
                'recall_at_capacity':positive/int(y.sum()),'lift_vs_random':precision/float(y.mean())})
    assert len(summary)==9 and len(fold_rows)==45 and len(all_candidates)==55
    write_csv(report/'all_rounds_comparison.csv',summary)
    write_csv(report/'all_parameters.csv',parameters)
    write_csv(report/'selected_model_fold_metrics.csv',fold_rows)
    write_csv(report/'all_grid_candidates.csv',all_candidates)
    write_csv(report/'ranking_capacity_comparison.csv',capacity)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    labels=[TITLES[r['model']]+' / '+r['stage'] for r in summary]
    colors=[{'logistic_regression':'#65768B','random_forest':'#2F7F86','xgboost':'#BE6546'}[r['model']] for r in summary]
    for ax,metric,title in zip(axes,['roc_auc','accuracy'],['Mean validation ROC-AUC','Mean validation accuracy']):
        values=[r[metric] for r in summary]
        ax.barh(labels,values,color=colors);ax.invert_yaxis();ax.set(xlim=(0,1),title=title)
        for i,value in enumerate(values):ax.text(value+.01,i,f'{value:.4f}',va='center',fontsize=9)
    fig.savefig(figures/'all_rounds_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for name,stage in [('logistic_regression','round1'),('random_forest','round2'),('xgboost','round3')]:
        state=states[(stage,name)];fpr,tpr,_=roc_curve(y,state['probabilities'])
        ax.plot(fpr,tpr,lw=2,label=f'{TITLES[name]} / {stage}: {state["metadata"]["pooled_oof_metrics"]["roc_auc"]:.4f}')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(title='Final-stage seven-feature models: pooled OOF ROC',xlabel='False positive rate',ylabel='Recall')
    ax.legend(loc='lower right');fig.savefig(figures/'final_stage_roc.png',dpi=160,bbox_inches='tight');plt.close(fig)
    best_auc=max(summary,key=lambda r:r['roc_auc']);best_acc=max(summary,key=lambda r:r['accuracy'])
    total=sum(s['metadata']['grid_cv_fits']+s['metadata']['selected_oof_fits']+s['metadata']['full_development_refits'] for s in states.values())
    assert total==329
    text='\n'.join(['model training2 七项特征完整训练报告','',f'开发集 {len(y):,} 条，cardio=1 共 {int(y.sum()):,} 条；与 training1 相同五折。',
        '输入：'+', '.join(FEATURES),'七项原始特征经类别编码后为九列。',
        'baseline：逻辑回归、随机森林、XGBoost 从固定初始参数开始训练。',
        'round1：逻辑回归搜索 C；随机森林搜索 max_depth/min_samples_leaf；XGBoost 搜索 max_depth/min_child_weight。',
        'round2：继承各自 round1 选中配置；随机森林搜索 max_features/max_samples；XGBoost 搜索 subsample/colsample_bytree。',
        'round3：继承 XGBoost round2 配置，搜索 learning_rate/n_estimators。',
        'round1/round2 按平均验证 AUC；round3 按五项指标平均名次，并列优先 AUC、较少树、较低学习率。',
        '每轮的参数值、网格候选、折间波动和混淆矩阵均保存在对应子文件夹。',
        '',f'本轮共 9 个已保存的阶段模型、55 个网格候选记录、275 次网格 CV 拟合，合计 {total} 次拟合。',
        '合计包含 baseline 的 15 次折拟合、调参后 30 次 OOF 重放和 9 次全开发集重拟合。',
        '',f'表中 AUC 最高：{best_auc["model"]} / {best_auc["stage"]}，{best_auc["roc_auc"]:.6f}。',
        f'表中 Accuracy 最高：{best_acc["model"]} / {best_acc["stage"]}，{best_acc["accuracy"]:.3%}。',
        '',*[' / '.join([r['model'],r['stage']])+': '+', '.join(f'{m}={r[m]:.6f}' for m in METRICS) for r in summary],
        '','解释与限制',
        '所有表格为五折算术均值；ROC 图为合并 OOF。训练 Accuracy 用于诊断过拟合，不参与选参。',
        '最终阶段候选不保证每一项指标均高于前一阶段；多指标选择可能有取舍，应查看所有候选。',
        '等权平均名次是项目的折中约定，指标互有重叠；名次差和小幅指标差均不代表统计显著。',
        '分类阈值固定为 0.5，没有优化阈值。排名容量分析用各折内部的 Top-K 比例，同分边界一并纳入。',
        '容量表是已有标签的回顾性富集模拟，不代表真实就诊、复查优先级或进一步评估获益。',
        '重复使用同一开发折选择参数，且此前已查看全量描述性图；结果属于探索性开发比较。',
        '未做嵌套交叉验证或最终测试预测评估；当前外部划分可能有相同健康资料跨开发集与测试集。',
        '本次仅读取开发 CSV、原始数据中对应开发记录及反归一化参数；未读取最终测试 CSV 或全量清洗 CSV。',
        'model training1 归档及原数据保持不变。','',
        '复现：python3 "src/model training2/train_all.py"。输出已存在时不会覆盖；--resume 只读取已完成且配置相同的阶段。',
        '预测：python3 "src/model training2/predict.py" --model MODEL.joblib --input INPUT.csv --output predictions.csv。',
        '默认输入是现有清洗 CSV 格式；--original-units 接收原单位七项特征和原始类别代码。',
        'ChatGPT 协助编写、运行和整理，所有指标由本地计算并核验。'])+'\n'
    (report/'overall_report_zh.txt').write_text(text,encoding='utf-8')
    links='<p>'+''.join(f'<a href="{stage}/{name}/report.html">{stage} / {TITLES[name]}</a><br>' for stage,name in states)+'</p>'
    save_html(report/'overall_report.html','model training2 七项特征全流程对比',text,
        [figures/'all_rounds_comparison.png',figures/'final_stage_roc.png'],links+table_html(summary))
    experiment={'run':RUN,'features':FEATURES,**provenance,'models_saved':9,'candidate_records':55,'grid_cv_fits':275,
        'selected_oof_fits':45,'full_development_refits':9,'total_fits':329,'test_evaluated':False,
        'stage_ids':[{'stage':stage,'model':name} for stage,name in states],
        'training_plan_sha256':sha256(out/'config'/RUN/'training_plan.json'),
        'training_code_sha256':sha256(Path(__file__)),'data_utils_sha256':sha256(Path(__file__).parent/'data_utils.py')}
    (report/'experiment_metadata.json').write_text(json.dumps(experiment,ensure_ascii=False,indent=2),encoding='utf-8')
    (report/'validation.txt').write_text('PASS: 9 stages, 55 grid candidates, 45 selected-fold rows and 329 total fits reconcile.\nPASS: all stages use identical seven features and grouped validation partitions.\nPASS: inherited parameters are tracked; individual OOF and saved-model checks passed.\nPASS: no final test data evaluated; training1 is not modified.\n',encoding='utf-8')
    print('ALL_STAGES_COMPLETE',json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--plan',type=Path,default=ROOT/'config'/RUN/'training_plan.json')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output_root or args.project_root
    plan=json.loads(args.plan.read_text());assert plan['features']==FEATURES
    config=out/'config'/RUN;config.mkdir(parents=True,exist_ok=True)
    target=config/'training_plan.json'
    if target.resolve()!=args.plan.resolve():
        if target.exists():assert target.read_bytes()==args.plan.read_bytes()
        else:shutil.copy2(args.plan,target)
    context=load_development(args.project_root,plan)
    (config/'data_provenance.json').write_text(json.dumps(context[-1],ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(config/'development_folds.csv',[{'id':r['id'],'profile_group':r['profile_group'],'fold':int(context[5][i])} for i,r in enumerate(context[2])])
    states={}
    for name in MODELS:states[('baseline',name)]=run_stage(out,'baseline',name,plan['baseline'][name],None,None,context,plan,args.resume)
    for stage,previous_stage in [('round1','baseline'),('round2','round1'),('round3','round2')]:
        for name,grid in plan[stage].items():
            previous=states[(previous_stage,name)]
            states[(stage,name)]=run_stage(out,stage,name,previous['metadata']['parameters'],grid,previous,context,plan,args.resume)
    summarize(out,states,context,plan)


if __name__=='__main__':main()
