"""Three descriptive extensions using development data only; no models or test-set reads."""
import argparse
import base64
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
AGES=['<30','30-39','40-49','50-59','60-69','70+']
FIELDS={'smoke':('Smoking','吸烟'),'alco':('Alcohol intake','饮酒'),'active':('Physical activity','身体活动')}
MIN_N=100
BP_BANDS=['<110','110-<120','120-<130','130-<140','140-<160','>=160']
BLUE,ORANGE='#23658A','#C26735'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def proportion(rows):
    return sum(int(r['cardio']) for r in rows)/len(rows) if rows else None


def bp_band(value):
    return ('<110' if value<110 else '110-<120' if value<120 else '120-<130' if value<130
            else '130-<140' if value<140 else '140-<160' if value<160 else '>=160')


def html_table(rows,columns):
    return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(r[c]))+'</td>' for c in columns)+'</tr>' for r in rows)+'</table>'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--output-root',type=Path)
    args=ap.parse_args(); project=args.project_root; out=args.output_root or project
    source=project/'data/processed/train.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    assert records and all(r['split']=='train' for r in records)
    assert all(r['age_group'] in AGES for r in records)
    n=len(records); overall=proportion(records)
    tables=out/'outputs/descriptive_deep/tables'
    figures=out/'outputs/figures/descriptive_deep'
    reports=out/'reports/descriptive_deep'
    for p in [tables,figures,reports]: p.mkdir(parents=True,exist_ok=True)
    present=[a for a in AGES if any(r['age_group']==a for r in records)]
    cells=defaultdict(list)
    for r in records:
        for field in FIELDS: cells[(field,r['age_group'],r[field])].append(r)
    # One common reference population for all three comparisons, with adequate support in both arms.
    common_ages=[a for a in present if all(len(cells[(f,a,v)])>=MIN_N for f in FIELDS for v in ['0','1'])]
    assert common_ages
    common=[r for r in records if r['age_group'] in common_ages]
    age_counts={a:sum(r['age_group']==a for r in common) for a in common_ages}
    weights={a:count/len(common) for a,count in age_counts.items()}
    strata=[]; mix=[]; contributions=[]; standard=[]
    for field in FIELDS:
        for value in ['0','1']:
            full_group=[r for r in records if r[field]==value]
            common_group=[r for r in common if r[field]==value]
            std=0.
            for a in present:
                subset=cells[(field,a,value)]; rate=proportion(subset)
                strata.append({'variable':field,'age_group':a,'category':value,'records':len(subset),
                    'cardio_1':sum(int(r['cardio']) for r in subset),
                    'positive_rate':rate if rate is not None else '',
                    'display_eligible':int(len(subset)>=MIN_N),'in_standard_population':int(a in common_ages)})
                mix.append({'variable':field,'category':value,'age_group':a,'records':len(subset),
                    'category_records_all_ages':len(full_group),'age_share_within_category':len(subset)/len(full_group)})
                if a in common_ages:
                    term=weights[a]*rate; std+=term
                    contributions.append({'variable':field,'category':value,'age_group':a,
                        'records':len(subset),'cardio_1':sum(int(r['cardio']) for r in subset),
                        'stratum_rate':rate,'reference_weight':weights[a],'weighted_contribution':term})
            standard.append({'variable':field,'category':value,'full_records':len(full_group),
                'crude_rate_full':proportion(full_group),'common_records':len(common_group),
                'crude_rate_common':proportion(common_group),'age_standardized_rate':std,
                'standardized_minus_common_crude_pp':(std-proportion(common_group))*100})
    weights_rows=[{'age_group':a,'reference_records':age_counts[a],
                   'reference_total':len(common),'weight':weights[a]} for a in common_ages]
    contrasts=[]
    for field in FIELDS:
        zero,one=[next(s for s in standard if s['variable']==field and s['category']==v) for v in ['0','1']]
        contrasts.append({'variable':field,'contrast':'1 minus 0',
            'crude_full_difference_pp':100*(one['crude_rate_full']-zero['crude_rate_full']),
            'crude_common_difference_pp':100*(one['crude_rate_common']-zero['crude_rate_common']),
            'standardized_difference_pp':100*(one['age_standardized_rate']-zero['age_standardized_rate'])})
    heat_groups=defaultdict(list)
    for r in records: heat_groups[(bp_band(float(r['ap_hi'])),r['cholesterol'])].append(r)
    heat=[]
    for band in BP_BANDS:
        for level in ['1','2','3']:
            subset=heat_groups[(band,level)]; rate=proportion(subset)
            heat.append({'systolic_band':band,'cholesterol':level,'records':len(subset),
                'cardio_1':sum(int(r['cardio']) for r in subset),
                'positive_rate':rate if rate is not None else '',
                'overall_development_positive_rate':overall,
                'difference_from_overall_pp':(rate-overall)*100 if rate is not None else '',
                'display_eligible':int(len(subset)>=MIN_N)})
    exports={'age_lifestyle_strata':strata,'lifestyle_age_composition':mix,
             'age_standardization':standard,'standard_population_weights':weights_rows,
             'standardization_contributions':contributions,'lifestyle_rate_differences':contrasts,
             'systolic_cholesterol':heat}
    for name,rows in exports.items(): write_csv(tables/f'{name}.csv',rows)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                          'axes.spines.top':False,'axes.spines.right':False})
    figure_list=[]
    def save(fig,name,title):
        fig.get_layout_engine().set(rect=(0,.09,1,.91))
        fig.text(.01,.015,f'Development data only | Overall n={n:,} | Descriptive associations, not causal effects',
                 fontsize=10,color='#616A74')
        fig.savefig(figures/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(figures/f'{name}.svg',bbox_inches='tight')
        plt.close(fig); figure_list.append({'file':name,'title':title})

    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,(field,(title,_)) in zip(axes,FIELDS.items()):
        x=np.arange(len(common_ages))
        for v,offset,color in [('0',-.19,BLUE),('1',.19,ORANGE)]:
            selected=[next(s for s in strata if s['variable']==field and s['category']==v and s['age_group']==a) for a in common_ages]
            heights=[s['positive_rate'] for s in selected]
            ax.bar(x+offset,heights,width=.36,label='No (0)' if v=='0' else 'Yes (1)',color=color)
            for xi,s in zip(x+offset,selected):
                other=next(t for t in strata if t['variable']==field and t['age_group']==s['age_group'] and t['category']=='0')
                label_y=max(s['positive_rate'],other['positive_rate'])+.085 if v=='1' else s['positive_rate']+.018
                ax.annotate(f'{s["positive_rate"]:.1%}\nn={s["records"]:,}',
                        xy=(xi,s['positive_rate']),xytext=(xi,label_y),
                        ha='center',va='bottom',fontsize=7.8,
                        arrowprops={'arrowstyle':'-','color':'#89939b','lw':.7,'shrinkA':3,'shrinkB':3})
        ax.set_xticks(x,common_ages); ax.set_ylim(0,.9)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=title,xlabel='Age group (years)',ylabel='Disease label proportion')
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.legend(loc='upper left',fontsize=9)
    save(fig,'01_age_stratified_lifestyle','年龄分层的生活习惯比较')

    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    for ax,(field,(title,_)) in zip(axes,FIELDS.items()):
        entries={r['category']:r for r in standard if r['variable']==field}
        for i,(metric,label,color) in enumerate([
            ('crude_rate_full','Crude: all ages','#9AB6C7'),
            ('crude_rate_common','Crude: common ages',BLUE),
            ('age_standardized_rate','Age-standardized',ORANGE)]):
            x=np.arange(2)+(i-1)*.24; heights=[entries[v][metric] for v in ['0','1']]
            ax.bar(x,heights,width=.22,color=color,label=label)
            for xi,h in zip(x,heights): ax.text(xi,h+.014,f'{h:.1%}',ha='center',fontsize=8.5)
        ax.set_xticks([0,1],['No (0)','Yes (1)']); ax.set_ylim(0,.75)
        ax.set(title=title,ylabel='Disease label proportion')
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True); ax.legend(loc='upper left',fontsize=8.5)
    save(fig,'02_crude_vs_age_standardized','原始比例与年龄标准化比例对照')

    lookup={(r['systolic_band'],r['cholesterol']):r for r in heat}
    grid=np.array([[lookup[(b,c)]['positive_rate'] if lookup[(b,c)]['display_eligible'] else np.nan
                    for c in ['1','2','3']] for b in BP_BANDS])
    diff=(grid-overall)*100; maxdiff=float(np.nanmax(np.abs(diff)))
    fig,axes=plt.subplots(1,2,figsize=(15,9),layout='constrained')
    for ax,data,cmap,vmin,vmax,title in [
        (axes[0],grid,'YlGnBu',0,1,'Disease label proportion'),
        (axes[1],diff,'RdBu_r',-maxdiff,maxdiff,'Difference from overall proportion (pp)')]:
        colors=plt.get_cmap(cmap).copy(); colors.set_bad('#E9EDF0')
        im=ax.imshow(data,cmap=colors,vmin=vmin,vmax=vmax,aspect='auto')
        ax.set_xticks([0,1,2],['1: Normal','2: Above\nnormal','3: Well above\nnormal'])
        ax.set_yticks(range(len(BP_BANDS)),BP_BANDS)
        ax.set(xlabel='Cholesterol category',ylabel='Systolic BP band (mmHg)',title=title)
        for i,b in enumerate(BP_BANDS):
            for j,c in enumerate(['1','2','3']):
                r=lookup[(b,c)]
                if not r['display_eligible']:
                    text=f'Suppressed\nn={r["records"]:,}'
                elif ax is axes[0]: text=f'{r["positive_rate"]:.1%}\nn={r["records"]:,}'
                else: text=f'{r["difference_from_overall_pp"]:+.1f} pp\nn={r["records"]:,}'
                dark=np.isfinite(data[i,j]) and (data[i,j]>.62 if ax is axes[0] else abs(data[i,j])>maxdiff*.65)
                ax.text(j,i,text,ha='center',va='center',fontsize=10,color='white' if dark else '#26313c')
        cb=fig.colorbar(im,ax=ax,shrink=.8)
        if ax is axes[0]: cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig,'03_systolic_cholesterol_heatmap','收缩压与胆固醇组合热力图')

    # Findings are calculated from the exported study population, never assumed in advance.
    detail=[]
    for field,(_,cn) in FIELDS.items():
        for a in common_ages:
            s0=next(s for s in strata if s['variable']==field and s['category']=='0' and s['age_group']==a)
            s1=next(s for s in strata if s['variable']==field and s['category']=='1' and s['age_group']==a)
            detail.append(f'{cn}，{a}岁：0组{s0["positive_rate"]:.1%}（n={s0["records"]:,}）；1组{s1["positive_rate"]:.1%}（n={s1["records"]:,}）；差值{(s1["positive_rate"]-s0["positive_rate"])*100:+.2f}个百分点。')
    directions=[]
    for field,(_,cn) in FIELDS.items():
        higher=[]; lower=[]
        for a in common_ages:
            difference=proportion(cells[(field,a,'1')])-proportion(cells[(field,a,'0')])
            if difference>0: higher.append(a)
            elif difference<0: lower.append(a)
        directions.append(f'{cn}：1组比例高于0组的年龄组为{", ".join(higher) if higher else "无"}；低于0组的年龄组为{", ".join(lower) if lower else "无"}。这些是观察方向，未检验显著性。')
    detail=directions+detail
    std_text=[]
    display_rows=[]
    for field,(_,cn) in FIELDS.items():
        zero,one=[next(s for s in standard if s['variable']==field and s['category']==v) for v in ['0','1']]
        d=next(r for r in contrasts if r['variable']==field)
        std_text.append(f'{cn}：全开发集粗比例（0/1）为{zero["crude_rate_full"]:.2%}/{one["crude_rate_full"]:.2%}；共同年龄范围粗比例为{zero["crude_rate_common"]:.2%}/{one["crude_rate_common"]:.2%}；年龄标准化后为{zero["age_standardized_rate"]:.2%}/{one["age_standardized_rate"]:.2%}。1减0差值由{d["crude_common_difference_pp"]:+.2f}变为{d["standardized_difference_pp"]:+.2f}个百分点。')
        for value,entry in [('0',zero),('1',one)]:
            display_rows.append({'变量':cn,'类别':value,'全龄粗比例':f'{entry["crude_rate_full"]:.2%}',
                '共同年龄粗比例':f'{entry["crude_rate_common"]:.2%}',
                '年龄标准化比例':f'{entry["age_standardized_rate"]:.2%}',
                '共同年龄样本量':f'{entry["common_records"]:,}'})
    std_text.append('统一年龄构成后，吸烟与饮酒的组间差距缩小，支持年龄构成是粗比例差异的部分解释；身体活动组间仍有差距。未控制其他因素，也未计算不确定区间，因此不能解释为独立效应或统计显著差异。')
    eligible=[r for r in heat if r['display_eligible']]
    low=min(eligible,key=lambda r:r['positive_rate']); high=max(eligible,key=lambda r:r['positive_rate'])
    heat_text=[f'开发集整体标签1比例为{overall:.2%}。颜色差值以此为参照，并非年龄调整后的差异。',
        f'可展示格中，最低比例出现在收缩压{low["systolic_band"]} mmHg、胆固醇等级{low["cholesterol"]}：{low["positive_rate"]:.1%}（n={low["records"]:,}）；最高出现在收缩压{high["systolic_band"]} mmHg、胆固醇等级{high["cholesterol"]}：{high["positive_rate"]:.1%}（n={high["records"]:,}）。',
        '组合分组反映未经调整的共现规律，不能证明两项指标有统计交互效应，也不是诊断阈值或转诊规则。']
    heat_text.insert(1,f'在同一收缩压120–<130 mmHg区间，胆固醇等级1的标签比例为{lookup[("120-<130","1")]["positive_rate"]:.1%}（n={lookup[("120-<130","1")]["records"]:,}），等级3为{lookup[("120-<130","3")]["positive_rate"]:.1%}（n={lookup[("120-<130","3")]["records"]:,}）。')
    heat_text.append('不是每个血压区间中，标签比例都随胆固醇等级严格递增；应展示完整热力图，不只选择最大和最小格作结论。')
    methods=[
        f'只读取train.csv，共{n:,}条；不读取test.csv，不修改现有数据、模型和原描述性输出。',
        f'共同标准人群包含年龄组：{", ".join(common_ages)}，共{len(common):,}条；排除共同支持外记录{n-len(common)}条。',
        '共同支持规则：三项生活习惯在每个纳入年龄组中，0/1两类均至少100条。这是预设展示和稳定性规则，不是统计显著性判断。',
        '原始分层CSV保留所有观察年龄组，包括空单元格和小样本组；图中只展示共同支持年龄组。',
        '直接年龄标准化：每类各年龄组的标签比例乘以开发集共同年龄范围内的统一年龄权重，再求和。',
        '权重来自全部共同人群，三项变量及0/1两类使用完全相同的权重；不是每一类各算各的权重。',
        '比较标准化效果时以共同年龄范围的粗比例为参照，避免把改变人群范围误认为年龄调整效果。',
        '年龄按十岁组调整，组内仍有残余年龄差异；未控制性别、其他习惯及选择、自报和测量偏差。',
        '标准化后的数字是统一年龄构成下的描述性标签比例，不是个人患病概率，不是总体患病率，也不是因果效应。',
        '血压热力图仅使用收缩压ap_hi，单位mmHg；区间为<110、110–<120、120–<130、130–<140、140–<160、>=160。',
        '血压区间是本项目固定展示区间，不命名为临床分级，不根据疾病比例调整边界；胆固醇是等级而非具体浓度。',
        '热力图n<100的格子隐藏比例，CSV保留计数及比例便于核查；百分点差异不是相对风险。',
        '不报告p值、显著性或因果结论。尤其不能把吸烟/饮酒组较低的粗比例解释成保护作用。']
    report_text='Cardio 深入描述性分析\n\n'
    report_text+='1 年龄分层的生活习惯比较\n'+'\n'.join(detail)+'\n\n'
    report_text+='2 原始比例与年龄标准化比例对照\n'+'\n'.join(std_text)+'\n\n'
    report_text+='3 收缩压 × 胆固醇热力图\n'+'\n'.join(heat_text)+'\n\n'
    report_text+='方法与限制\n'+'\n'.join('- '+s for s in methods)+'\n\n统一年龄权重\n'
    report_text+='\n'.join(f'{a}岁：{weights[a]:.6%}，n={age_counts[a]:,}' for a in common_ages)
    report_text+='\n\n来源：data/processed/train.csv；https://github.com/caravanuden/cardio\n'
    (reports/'deep_descriptive_zh.txt').write_text(report_text,encoding='utf-8')
    parts=[]
    for i,f in enumerate(figure_list):
        img=base64.b64encode((figures/f'{f["file"]}.png').read_bytes()).decode()
        paragraphs=[detail,std_text,heat_text][i]
        content='<h2>'+str(i+1)+' '+html.escape(f['title'])+'</h2><img alt="'+html.escape(f['title'])+'" src="data:image/png;base64,'+img+'">'
        content+='<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in paragraphs)+'</ul>'
        if i==1: content+=html_table(display_rows,list(display_rows[0]))
        parts.append('<section>'+content+'</section>')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cardio 深入描述性分析</title><style>body{font-family:-apple-system,"PingFang SC",Arial,sans-serif;max-width:1250px;margin:35px auto;padding:20px;color:#26313c;line-height:1.8}h1{font-size:30px}h2{margin-top:38px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d9d9d9;padding:9px;text-align:left}th{background:#edf2f6}section{margin-bottom:40px}</style><h1>Cardio 深入描述性分析</h1>'
    doc+=f'<p>开发集{n:,}条；共同年龄标准人群{len(common):,}条。探索生活习惯的分层关联、统一年龄构成后的比例，以及收缩压与胆固醇的组合分布。</p>'
    doc+=''.join(parts)+'<h2>方法与限制</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in methods)+'</ul>'
    doc+='<p>复现：python3 src/descriptive_deep.py。来源：data/processed/train.csv；<a href="https://github.com/caravanuden/cardio">原项目</a>。ChatGPT协助编写代码、解释结果，所有统计由本地脚本计算。</p></html>'
    (reports/'deep_descriptive_report.html').write_text(doc,encoding='utf-8')
    notes='深入Descriptive展示建议\n\n第1页 年龄分层比较\n先展示原粗比例，再说明分层后不同年龄组方向不完全一致，避免把生活习惯关联当成因果。\n'+ '\n'.join(detail)+'\n\n第2页 年龄标准化\n解释统一权重，报告0/1差异调整前后改变多少。\n'+'\n'.join(std_text)+'\n\n第3页 收缩压与胆固醇\n'+'\n'.join(heat_text)+'\n'
    (reports/'presentation_notes_zh.txt').write_text(notes,encoding='utf-8')

    # Reconcile exports and independently calculate standardized rates with weighted record sums.
    assert sum(r['records'] for r in heat)==n
    assert sum(r['cardio_1'] for r in heat)==sum(int(r['cardio']) for r in records)
    assert abs(sum(weights.values())-1)<1e-12
    for field in FIELDS:
        assert sum(s['records'] for s in strata if s['variable']==field)==n
        assert sum(s['cardio_1'] for s in strata if s['variable']==field)==sum(int(r['cardio']) for r in records)
        for value in ['0','1']:
            s=next(s for s in standard if s['variable']==field and s['category']==value)
            selected=[r for r in common if r[field]==value]
            weighted=sum(int(r['cardio'])*weights[r['age_group']]/len(cells[(field,r['age_group'],value)]) for r in selected)
            assert abs(weighted-s['age_standardized_rate'])<1e-11
            assert min(proportion(cells[(field,a,value)]) for a in common_ages)<=s['age_standardized_rate']<=max(proportion(cells[(field,a,value)]) for a in common_ages)
    for name,expected in exports.items():
        with (tables/f'{name}.csv').open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
        assert len(saved)==len(expected)
        for row,original in zip(saved,expected):
            assert all(row[k]==str(v) for k,v in original.items())
    from PIL import Image
    for f in figure_list:
        with Image.open(figures/f'{f["file"]}.png') as image: image.verify()
    manifest={'records':n,'common_records':len(common),'common_age_groups':common_ages,
        'excluded_from_standardization':n-len(common),'reference_weights':weights,
        'minimum_cell_n':MIN_N,'overall_positive_rate':overall,'contrasts':contrasts,
        'standardized_rates':standard,'figures':figure_list,'csv_tables':list(exports),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__}
    (reports/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (reports/'validation.txt').write_text('PASS: only development train.csv read.\nPASS: age strata and BP/cholesterol counts reconcile with the full development sample.\nPASS: one common standard population and weights sum to 1.\nPASS: six standardized rates independently verified using record-level weighting.\nPASS: exported CSV round-trip values match computed tables.\nPASS: three PNGs decode; visual inspection performed separately.\n',encoding='utf-8')
    print('\n'.join(std_text)); print('\n'.join(heat_text)); print('Common ages:',common_ages,'n=',len(common))


if __name__=='__main__': main()
