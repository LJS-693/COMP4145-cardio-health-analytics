from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy.stats import mannwhitneyu
from plotly.subplots import make_subplots

st.set_page_config(page_title="Cardio Health Analytics", layout="wide")
DATA = Path(__file__).resolve().parent.parent / "data/processed/caroline_cleaned version"
COHORT_FILES = {"Full Cohort (100%)": "cardio_fully_cleaned.csv", "Training Cohort (80%)": "cardio_train_80.csv"}
TEAL, RED, GRAY = "#247986", "#bd514d", "#788492"
AGE, BMI, BP = "Age group", "BMI category", "Blood pressure stage"
AGE_BINS = ["<40", "40–44", "45–49", "50–54", "55–59", "60–64"]
BMI_BINS = ["Underweight (<18.5)", "Normal [18.5,25)", "Overweight [25,30)", "Obese I [30,35)", "Obese II+ (≥35)"]
BP_BINS = ["<120", "120–129", "130–139", "140–159", "≥160"]
LEVELS = ["Normal", "Above normal", "Well above normal"]
BINARY = {"Gender": "gender", "Smoking": "smoke", "Alcohol": "alco", "Physical activity": "active"}
FEATURES = [BP, AGE, BMI, "Cholesterol", "Glucose", *BINARY]
MEASURES = {"Systolic BP (mmHg)": "ap_hi", "Mean arterial pressure (mmHg)": "map",
            "Diastolic BP (mmHg)": "ap_lo", "Pulse pressure (mmHg)": "pulse_pressure",
            "Age (years)": "age_years", "BMI (kg/m²)": "bmi", "Weight (kg)": "weight", "Height (cm)": "height"}


@st.cache_data
def load(filename):
    d = pd.read_csv(DATA / filename)
    for col, scale in json.loads((DATA / "denormalization_params.json").read_text()).items():
        d[col] = (d[col] * (scale["max"] - scale["min"]) + scale["min"]).round(10)
    for prefix, label in [("cholesterol", "Cholesterol"), ("gluc", "Glucose")]:
        indicators = d[[f"{prefix}_{i}" for i in (1, 2, 3)]].apply(
            lambda s: s.astype(str).str.lower().map({"true": 1, "false": 0, "0": 0, "1": 1}).astype(int))
        if not indicators.sum(axis=1).eq(1).all():
            raise ValueError(f"Invalid {label} encoding")
        d[prefix] = indicators.to_numpy() @ np.arange(1, 4)
        d[label] = pd.Categorical.from_codes(d[prefix]-1, LEVELS, ordered=True)
    for key, col in BINARY.items():
        d[key] = pd.Categorical.from_codes(d[col], ["Female", "Male"] if key == "Gender" else ["No", "Yes"])
    for key, col, cuts, labels in [(AGE, "age_years", [40,45,50,55,60], AGE_BINS),
                                   (BMI, "bmi", [18.5,25,30,35], BMI_BINS),
                                   (BP, "ap_hi", [120,130,140,160], BP_BINS)]:
        d[key] = pd.cut(d[col], [-np.inf,*cuts,np.inf], labels=labels, right=False)
    return d


def combo(t):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=t.index, y=t["size"], name="Records", marker_color="#aebbc9", opacity=.45, secondary_y=True)
    fig.add_scatter(x=t.index, y=t["mean"], name="Disease rate", mode="lines+markers+text",
                    text=t["mean"].map(lambda p: f"{p:.1%}" if pd.notna(p) else ""),
                    textposition="top center", line_color=RED, secondary_y=False)
    fig.update_yaxes(range=[0,1], tickformat=".0%", secondary_y=False)
    fig.update_yaxes(rangemode="tozero", secondary_y=True)
    fig.update_layout(yaxis=dict(layer="above traces"))
    return fig


def show(fig):
    fig.update_layout(margin=dict(t=45,b=55,l=55,r=35), legend=dict(orientation="h", y=1.14))
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def full_stats(filename):
    d = load(filename)
    labels = {"cardio":"Cardio disease","ap_hi":"Systolic BP","map":"MAP","pulse_pressure":"Pulse pressure",
              "ap_lo":"Diastolic BP","age_years":"Age","cholesterol":"Cholesterol","bmi":"BMI","weight":"Weight",
              "gluc":"Glucose","gender":"Gender (male)","alco":"Alcohol","height":"Height","smoke":"Smoking","active":"Physical activity"}
    corr = d[list(labels)].rename(columns=labels).corr(method="spearman")
    order = corr["Cardio disease"].sort_values(ascending=False).index
    corr = corr.loc[order,order]
    matrix = go.Figure(go.Heatmap(z=corr,x=order,y=order,colorscale="RdBu_r",zmin=-1,zmax=1,
                         texttemplate="%{z:.2f}",textfont_size=9,hovertemplate="%{x} / %{y}: %{z:.3f}<extra></extra>"))
    matrix.update_yaxes(autorange="reversed")
    matrix.update_layout(height=650)
    x = pd.DataFrame(index=d.index)
    for col, unit in [("ap_hi","mmHg"),("ap_lo","mmHg"),("age_years","years"),("bmi","kg/m²")]:
        sd = d[col].std()
        x[f"{labels[col]} (+{sd:.1f} {unit})"] = (d[col]-d[col].mean())/sd
    for col in ["cholesterol","gluc"]:
        for level in [2,3]:
            x[f"{labels[col]}: {LEVELS[level-1]} vs normal"] = d[col].eq(level).astype(int)
    for name,col in BINARY.items():
        x[name + (" (male vs female)" if name=="Gender" else " (yes vs no)")] = d[col]
    fit = sm.Logit(d.cardio, sm.add_constant(x)).fit(disp=False)
    odds = np.exp(fit.params.drop("const")).sort_values()
    ci = np.exp(fit.conf_int().loc[odds.index])
    forest = go.Figure()
    for label, value in odds.items():
        lo,hi = ci.loc[label]
        color = GRAY if fit.pvalues[label]>=.05 else RED if value>1 else TEAL
        forest.add_scatter(x=[value],y=[label],mode="markers",marker=dict(color=color,size=9),
                           error_x=dict(type="data",array=[hi-value],arrayminus=[value-lo],color=color),
                           showlegend=False,hovertemplate=f"{value:.3f} ({lo:.3f}–{hi:.3f})<extra></extra>")
        forest.add_annotation(x=1.01,xref="paper",y=label,text=f"{value:.2f} ({lo:.2f}–{hi:.2f})",
                              showarrow=False,xanchor="left",font_size=11)
    forest.add_vline(x=1,line_dash="dash",line_color=GRAY)
    forest.update_layout(height=580,margin=dict(l=230,r=125,t=30,b=45),xaxis_title="Adjusted odds ratio")
    return matrix,forest


st.title("Cardio Health Analytics")
st.markdown("""<style>[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;box-shadow:0 4px 16px #16344c12}
[data-testid="stMetric"]{background:linear-gradient(110deg,#e0f3f2,#edf0fa);padding:14px;border-radius:14px}</style>""",unsafe_allow_html=True)
ss = st.session_state
defaults = {"age":(29,65),"low":29,"high":65,**{name:"All" for name in BINARY},**{b:True for b in BMI_BINS}}
for key,value in defaults.items():
    ss.setdefault(key,value)


def sync_slider():
    ss.low,ss.high = ss.age


def sync_bounds():
    ss.low,ss.high = sorted((ss.low,ss.high))
    ss.age = (ss.low,ss.high)


with st.container(border=True):
    cohort = st.radio("Dataset", list(COHORT_FILES), horizontal=True, key="cohort")
    filename = COHORT_FILES[cohort]
    try:
        d = load(filename)
    except FileNotFoundError:
        st.error(f"Missing dataset: {filename}")
        st.stop()
    st.caption(f"Source: {filename} · {len(d):,} records before filters")
    st.subheader("Cohort filters")
    a,b,c = st.columns([1,4,1])
    a.number_input("Minimum age",29,65,key="low",on_change=sync_bounds)
    b.slider("Age range",29,65,key="age",on_change=sync_slider)
    c.number_input("Maximum age",29,65,key="high",on_change=sync_bounds)
    for col,name in zip(st.columns(5), BMI_BINS):
        col.checkbox(name,key=name)
    for col,name in zip(st.columns(4),BINARY):
        col.selectbox(name,["All",*d[name].cat.categories],key=name)
    st.button("Reset all filters",on_click=lambda: ss.update(defaults))
mask = d.age_years.between(*ss.age) & d[BMI].isin([b for b in BMI_BINS if ss[b]])
for name in BINARY:
    if ss[name]!="All":
        mask &= d[name].eq(ss[name])
f = d.loc[mask]
n = len(f)
rate = f.cardio.mean()
cols = st.columns(4)
cols[0].metric("Total records",f"{n:,}",f"{n/len(d):.1%} of selected cohort")
cols[1].metric("Disease cases",f"{f.cardio.sum():,}")
cols[2].metric("Disease prevalence",f"{rate:.1%}" if n else "—",f"{100*(rate-d.cardio.mean()):+.1f} pp" if n else None)
cols[3].metric("Average age / BMI",f"{f.age_years.mean():.1f} yrs" if n else "—",f"BMI {f.bmi.mean():.1f}" if n else None)
st.header("Filtered cohort exploration")
if not n:
    st.info("No records match these filters. Adjust the cohort filters or reset them.")
else:
    left,right = st.columns(2)
    with left.container(border=True):
        st.subheader("Single-category risk profile")
        key = st.selectbox("Select feature",FEATURES)
        split = st.selectbox("Optional split by",["None",*[s for s in [*BINARY,"Cholesterol","Glucose"] if s!=key]])
        if split=="None":
            fig = combo(f.groupby(key,observed=True).cardio.agg(["mean","size"]))
            fig.add_hline(y=rate,line_dash="dash",line_color=TEAL,secondary_y=False)
        else:
            fig = go.Figure()
            for val,g in f.groupby(split,observed=True):
                t=g.groupby(key,observed=True).cardio.agg(["mean","size"])
                fig.add_bar(name=str(val),x=t.index,y=t["mean"],texttemplate="%{y:.1%}",customdata=t["size"],
                            hovertemplate="%{x}: %{y:.1%}, n=%{customdata:,}")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
        show(fig)
        if {key,split} & {"Smoking","Alcohol"}:
            st.caption("Age, gender and changes in habits after diagnosis may influence crude rates. These data cannot establish protection.")
    with right.container(border=True):
        st.subheader("Clinical measurement comparator")
        name=st.selectbox("Clinical measurement",MEASURES)
        view=st.radio("Distribution",["Violin + boxplot","ECDF"],horizontal=True)
        samples=[f.loc[f.cardio.eq(i),MEASURES[name]] for i in (0,1)]
        a,b=samples
        if min(len(a),len(b))>=2:
            diff=b.mean()-a.mean()
            pooled=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
            effect=f"{diff/pooled:.2f}" if pooled else "—"
            st.caption(f"No disease mean (median): {a.mean():.1f} ({a.median():.1f}) · Disease: {b.mean():.1f} ({b.median():.1f}) · Difference: {diff:+.1f} · Cohen’s d: {effect} · Mann–Whitney p: {mannwhitneyu(a,b).pvalue:.3g}")
        else:
            st.caption("Group comparison needs at least two records in each group.")
        fig=go.Figure()
        for v,label,color in zip(samples,["No disease","Disease"],[TEAL,RED]):
            if view=="ECDF" and len(v):
                freq=v.value_counts().sort_index().cumsum()/len(v)
                fig.add_scatter(x=freq.index,y=freq,mode="lines",line_shape="hv",name=label,line_color=color)
            elif len(v):
                fig.add_violin(y=v,name=label,box_visible=True,meanline_visible=True,spanmode="hard",points=False,line_color=color)
        if view=="ECDF":
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(title=name)
        else:
            fig.update_yaxes(title=name)
        show(fig)
    left,right=st.columns(2)
    with left.container(border=True):
        st.subheader("Two-factor prevalence heatmap")
        presets=[(BP,AGE),("Cholesterol","Glucose"),(BP,"Physical activity"),("Gender",AGE)]
        ss.setdefault("xvar",BP)
        ss.setdefault("yvar",AGE)
        for i,(x,y) in enumerate(presets):
            st.button(f"{x} × {y}",key=f"preset{i}",on_click=lambda x=x,y=y:ss.update(xvar=x,yvar=y))
        x=st.selectbox("X-axis",FEATURES,key="xvar")
        y=st.selectbox("Y-axis",FEATURES,key="yvar")
        if x==y:
            st.info("Select two different variables.")
        else:
            table=f.groupby([y,x],observed=False).cardio.agg(["mean","size"])
            z=table["mean"].unstack()
            counts=table["size"].unstack()
            fig=go.Figure(go.Heatmap(z=z,x=z.columns,y=z.index,zmin=0,zmax=1,colorscale="RdBu_r",
                                      customdata=counts,colorbar_tickformat=".0%",hovertemplate="%{x} / %{y}: %{z:.1%}, n=%{customdata:,}<extra></extra>"))
            for row in z.index:
                for col in z.columns:
                    p,num=z.loc[row,col],counts.loc[row,col]
                    if num:
                        fig.add_annotation(x=col,y=row,text=f"{p:.1%}<br>(n={num:,})"+("*" if num<20 else ""),
                                           hovertext="Small cell: fewer than 20 records" if num<20 else None,
                                           font_color="white" if p<.23 or p>.77 else "black",showarrow=False)
            show(fig)
    with right.container(border=True):
        st.subheader("Cumulative risk factor builder")
        flags={"High BP (≥140 systolic or ≥90 diastolic)":(f.ap_hi>=140)|(f.ap_lo>=90),
               "Elevated cholesterol":f.cholesterol>=2,"Elevated glucose":f.gluc>=2,"BMI ≥30":f.bmi>=30,
               "Physically inactive":f.active==0,"Age ≥55":f.age_years>=55}
        active=st.multiselect("Factors included",list(flags),default=list(flags))
        if active:
            score=sum(flags[k].astype(int) for k in active)
            t=f.groupby(score).cardio.agg(["mean","size"]).reindex(range(len(active)+1))
            t["size"]=t["size"].fillna(0)
            fig=combo(t)
            fig.update_xaxes(title="Number of selected factors present",dtick=1)
            show(fig)
        else:
            st.info("Choose at least one factor to calculate a score.")
st.divider()
st.header(f"Dataset-wide statistical overview ({cohort}, unfiltered N = {len(d):,})")
st.caption("Not affected by top filters")
matrix,forest=full_stats(filename)
with st.container(border=True):
    st.subheader("Spearman rank correlations")
    show(matrix)
with st.container(border=True):
    st.subheader("Adjusted odds ratios and 95% confidence intervals")
    st.plotly_chart(forest,width="stretch")
    st.caption("Each odds ratio describes the association with the disease label while holding the other listed predictors constant. Red and teal indicate positive and negative associations with p < 0.05; gray indicates p ≥ 0.05.")
