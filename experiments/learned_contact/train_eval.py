"""Offline training/evaluation of contact gates (NO policy finetuning).

Only real public+eval JSONL pairs are accepted. Split by initial state before
fitting preprocessors/models; test results are evaluated once after selection.
No GPU/LIBERO claims are made by this script alone.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .dataset import load_pairs,group_split,labeled_data,FEATURE_NAMES
from .mixture import combine
from .stream import source_version

GRIP_IDX=[FEATURE_NAMES.index(k) for k in ('grip_q0','grip_q1','grip_delta','prev_action_6')]


def train_sklearn(train,val,seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    xt,yt,_=labeled_data(train)
    xv,yv,_=labeled_data(val)
    if len(np.unique(yt))!=2 or len(np.unique(yv))!=2:
        raise ValueError('Both contact classes must appear in train and val. Gather more diverse data.')
    out={}
    for name,indices,ctor in [
        ('gripper_lr',GRIP_IDX,lambda:LogisticRegression(max_iter=800,random_state=seed)),
        ('full_lr',list(range(len(FEATURE_NAMES))),lambda:LogisticRegression(max_iter=800,random_state=seed)),
        ('full_mlp',list(range(len(FEATURE_NAMES))),lambda:MLPClassifier(hidden_layer_sizes=(32,),max_iter=300,
            early_stopping=False,random_state=seed))]:
        xtr=xt[:,:,indices].reshape(len(xt),-1)
        xva=xv[:,:,indices].reshape(len(xv),-1)
        scaler=StandardScaler().fit(xtr)
        model=ctor();model.fit(scaler.transform(xtr),yt)
        val_prob=model.predict_proba(scaler.transform(xva))[:,1]
        out[name]=dict(model=model,scaler=scaler,indices=indices,
                       val_brier=float(np.mean((val_prob-yv)**2)))
    return out


def train_gru(train,val,seed,epochs=80):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    xt,yt,_=labeled_data(train);xv,yv,_=labeled_data(val)
    torch.manual_seed(seed)
    scaler=StandardScaler().fit(xt.reshape(-1,xt.shape[-1]))
    def norm(x): return scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
    X=torch.tensor(norm(xt),dtype=torch.float32)
    Y=torch.tensor(yt,dtype=torch.float32)
    XV=torch.tensor(norm(xv),dtype=torch.float32)
    YV=torch.tensor(yv,dtype=torch.float32)
    from .models import TinyGRU
    model=TinyGRU(X.shape[-1]);opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
    loss=nn.BCEWithLogitsLoss()
    best=None;best_score=np.inf;best_epoch=0
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(Y)),max(1,int(np.ceil(len(Y)/128)))):
            if not len(ids):continue
            opt.zero_grad();loss(model(X[ids]),Y[ids]).backward();opt.step()
        model.eval()
        with torch.no_grad():
            pv=torch.sigmoid(model(XV)).numpy()
            score=float(np.mean((pv-yv)**2))
        if score<best_score-1e-6:
            best_score=score;best_epoch=epoch;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=15:break
    model.load_state_dict(best);model.eval()
    return dict(model=model,scaler=scaler,val_brier=best_score,epoch=best_epoch,
                model_input_dim=int(X.shape[-1]))


def probabilities(desc,X,name):
    if name=='gru':
        import torch
        scaled=desc['scaler'].transform(X.reshape(-1,X.shape[-1])).reshape(X.shape)
        with torch.no_grad():
            return torch.sigmoid(desc['model'](torch.tensor(scaled,dtype=torch.float32))).numpy()
    xx=X[:,:,desc['indices']].reshape(len(X),-1)
    return desc['model'].predict_proba(desc['scaler'].transform(xx))[:,1]


def fixed_baselines(t,closed_aperture,open_aperture):
    """Observable-only M2-style fixed heuristic; requires independent calibration.

    Rule baseline is deliberately labeled 'M2_style' rather than asserting an
    exact match to an unconfigured original M2 checkpoint/run.
    """
    pub=t['public'];pred=[];last_vis=[];latched=None
    for row in pub:
        pix=row['target_pixel_visible'];eef=row['eef_pixel']
        if pix is not None and eef is not None:
            last_vis.append(row)
            last_vis=last_vis[-5:]
            latched=None
        if latched is None:
            if len(last_vis)>=5 and all(last_vis[i]['step']+1==last_vis[i+1]['step'] for i in range(4)):
                obj=np.array([r['target_pixel_visible'] for r in last_vis]);robot=np.array([r['eef_pixel'] for r in last_vis])
                q=np.array([r['gripper_qpos'] for r in last_vis]);aperture=q[:,0]-q[:,1]
                def diam(x): return float(np.linalg.norm(x[:,None]-x[None,:],axis=-1).max())
                if (diam(obj)>=1.5 and diam(robot)>=2. and diam(obj-robot)<=2. and
                    np.all(aperture<=closed_aperture)):
                    latched=1.
                elif (diam(obj)<=1.5 and diam(robot)>=2. and np.all(aperture>=open_aperture)):
                    latched=0.
        if row['target_pixel_missing'] and latched==1. and row['gripper_qpos'][0]-row['gripper_qpos'][1]>closed_aperture:
            latched=None
        pred.append(latched)
    return pred


def trajectory_predictions(t,p,*,method,model_seed,history,split_seed):
    """One PRE-action prediction per episode/step/method/seed, with full identity.

    All targets and privileged labels are named EVAL_ONLY and never passed to
    the estimator. Predicted state is RGB proxy centroid, not 3D center.
    """
    output=[]
    for i,row in enumerate(t['public']):
        pp=float(p[i]) if np.isfinite(p[i]) else None
        h=t['hyp'][i]
        mix=None
        if pp is not None and h['memory'] is not None:
            mix=combine(h['memory'],h['rigid'],pp,max_age=100,
                age=int(t['steps'][i])-int(h['anchor_step']) if h['anchor_step'] is not None else None)
        ref=t['target'][i]
        pixel_err=None;mean_err=None
        if row['target_pixel_missing'] and ref is not None and mix is not None:
            mean=np.asarray(h['memory'])*(1-pp)+np.asarray(h['rigid'])*pp
            mean_err=float(np.linalg.norm(mean-ref))
            if mix.cue_pixel is not None:
                pixel_err=float(np.linalg.norm(np.asarray(mix.cue_pixel)-ref))
        output.append(dict(episode_id=t['episode_id'],initial_state_id=t['group'],
            task_id=t['task_id'],trajectory_id=t['trajectory'],step=int(row['step']),
            method=method,seed=model_seed,collection_seed=t['collection_seed'],
            record_phase='pre_action_observation',code_version=t['code_version'],
            evaluation_code_version=source_version(),
            config=dict(collection=t['config'],model=dict(history=history,
                model_seed=model_seed,group_split_seed=split_seed)),
            target_pixel_missing=row['target_pixel_missing'],
            attached_probability=pp,
            memory_pixel=None if h['memory'] is None else list(map(float,h['memory'])),
            rigid_pixel=None if h['rigid'] is None else list(map(float,h['rigid'])),
            mixture_weights=None if mix is None else list(mix.weights),
            actual_cue_pixel=None if mix is None or mix.cue_pixel is None else list(mix.cue_pixel),
            actual_cue_abstained=bool(mix is None or mix.cue_pixel is None),
            state_estimate_decision='no_anchor' if mix is None else mix.decision,
            # Evaluation-only fields, never available to online inference.
            attached_label_EVAL_ONLY=(None if t['y'][i]<0 else int(t['y'][i])),
            target_pixel_reference_rgb_EVAL_ONLY=(None if ref is None else list(map(float,ref))),
            error_to_rgb_reference_px_EVAL_ONLY=pixel_err,
            mixture_mean_error_to_rgb_reference_px_EVAL_ONLY=mean_err))
    return output


def trace_metrics(t,p,*,strategy):
    """Score the exact online hard Memory/Rigid selector and its abstentions.

    All hidden frames with a clean-RGB reference are retained as a common
    denominator for state coverage, regardless of predicted validity. Brier
    is explicitly CONDITIONAL on a finite probability (not directly comparable
    across methods with different abstention).
    """
    labels=np.asarray(t['y']); p=np.asarray(p,dtype=float)
    hidden=np.asarray([r['target_pixel_missing'] for r in t['public']],bool)
    known=(labels>=0)&hidden
    p_valid=np.isfinite(p)
    mask=known&p_valid
    from collections import Counter
    from .mixture import combine
    errors={'actual_cue':[], 'mixture_mean_diagnostic':[], 'memory_expert':[], 'rigid_expert':[]}
    available_ref=0; output_count=0; abstain_reasons=Counter()
    false_abandon=0; attached=0; persistent=0
    for i,(pp,h,reference) in enumerate(zip(p,t['hyp'],t['target'])):
        if not hidden[i]:continue
        mix=None
        if np.isfinite(pp) and h['memory'] is not None and h['rigid'] is not None:
            mix=combine(h['memory'],h['rigid'],pp,max_age=100,
                age=int(t['steps'][i])-int(h['anchor_step']) if h['anchor_step'] is not None else None)
        if mix is None: abstain_reasons['missing_probability_or_anchor']+=1
        elif mix.cue_pixel is None: abstain_reasons[mix.decision]+=1
        if reference is not None:
            available_ref+=1
            ref=np.asarray(reference,float)
            if h['memory'] is not None:
                errors['memory_expert'].append(float(np.linalg.norm(np.asarray(h['memory'])-ref)))
            if h['rigid'] is not None:
                errors['rigid_expert'].append(float(np.linalg.norm(np.asarray(h['rigid'])-ref)))
            if mix is not None:
                mean=(1-float(pp))*np.asarray(h['memory'])+float(pp)*np.asarray(h['rigid'])
                errors['mixture_mean_diagnostic'].append(float(np.linalg.norm(mean-ref)))
                if mix.cue_pixel is not None:
                    output_count+=1
                    errors['actual_cue'].append(float(np.linalg.norm(np.asarray(mix.cue_pixel)-ref)))
        if np.isfinite(pp):
            if labels[i]==0 and pp>=.75:persistent+=1
            if labels[i]==1:
                attached+=1
                if pp<=.25:false_abandon+=1
    # Label transitions can be interval censored: 1 -> unknown ... -> 0.
    intervals=[]; last_one=None; between_unknown=0
    for i,label in enumerate(labels):
        if label==1:
            last_one=i;between_unknown=0
        elif label<0 and last_one is not None:
            between_unknown+=1
        elif label==0 and last_one is not None:
            if hidden[i]:
                intervals.append(dict(last_attached_step=int(t['steps'][last_one]),
                   first_detached_step=int(t['steps'][i]),unknown_between=int(between_unknown),
                   boundary_width_steps=int(t['steps'][i]-t['steps'][last_one]),
                   exact_transition_time_known=bool(between_unknown==0 and i==last_one+1)))
            last_one=None;between_unknown=0
    rmse=lambda x:float(np.sqrt(np.mean(np.square(x)))) if x else None
    counts=Counter(t.get('reason',[]))
    return dict(episode_id=t['episode_id'],initial_state_id=t['group'],
        trajectory_id=t['trajectory'],task_id=t['task_id'],method=strategy,
        collection_seed=t['collection_seed'],code_version=t['code_version'],
        evaluation_code_version=source_version(),step=None,aggregate_scope='trajectory',
        config=t['config'],record_phase='pre_action_observation',
        number_of_steps=int(len(labels)),missing_steps=int(hidden.sum()),
        hidden_labels_known=int(known.sum()),hidden_labels_unknown=int((hidden&~(labels>=0)).sum()),
        label_reasons=dict(counts),observed_labels=int(mask.sum()),
        brier_hidden=float(np.mean((p[mask]-labels[mask])**2)) if mask.any() else None,
        brier_comparison_scope='finite_prediction_only; not valid for cross-method comparison without shared mask',
        actual_output_rmse_to_rgb_proxy_px=rmse(errors['actual_cue']),
        mixture_mean_diagnostic_rmse_to_rgb_proxy_px=rmse(errors['mixture_mean_diagnostic']),
        memory_expert_rmse_to_rgb_proxy_px=rmse(errors['memory_expert']),
        rigid_expert_rmse_to_rgb_proxy_px=rmse(errors['rigid_expert']),
        state_rmse_to_same_rgb_proxy_px=rmse(errors['actual_cue']),
        error_geometry='224x224 RGB proxy centroid vs same detector on clean RGB; NOT 3D object center',
        hidden_reference_available=int(available_ref),valid_state_reference_steps=len(errors['actual_cue']),
        output_coverage_hidden_reference=(output_count/available_ref if available_ref else None),
        cue_abstain_reasons=dict(abstain_reasons),
        attached_false_abandon_rate=(false_abandon/attached if attached else None),
        detached_wrongly_rigid_steps=int(persistent),
        release_transition_intervals=intervals,
        release_delays=[], # old exact-delay estimate removed: unknown intervals make it unjustified
        coverage_confident=int(sum(p_valid&hidden&((p<=.25)|(p>=.75)))))


def compare_on_shared_frames(t,predictions):
    """All methods evaluated on exactly the SAME known hidden frames."""
    labels=np.asarray(t['y'],int)
    mask=(labels>=0)&np.array([r['target_pixel_missing'] for r in t['public']],bool)
    converted={k:np.asarray(v,float) for k,v in predictions.items()}
    coverage={k:dict(finite=int((mask&np.isfinite(v)).sum()),total=int(mask.sum()))
              for k,v in converted.items()}
    for v in converted.values(): mask &= np.isfinite(v)
    return {'trajectory_id':t['trajectory'],'initial_state_id':t['group'],
        'matched_hidden_frames':int(mask.sum()),'potential_known_hidden_frames':int(
         ((labels>=0)&np.array([r['target_pixel_missing'] for r in t['public']],bool)).sum()),
        'per_method_coverage':coverage,
        'matched_brier':{k:(float(np.mean((v[mask]-labels[mask])**2)) if mask.any() else None)
                         for k,v in converted.items()},
        'evaluation_scope':'development_only; matched observed frames; no selection on internal test'}


def save_figures(traces,pred,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    t=traces[0];r=pred[t['trajectory']]
    tt=t['steps'];objects=np.array([np.nan*np.ones(2) if v is None else v for v in t['target']])
    mem=np.array([np.nan*np.ones(2) if h['memory'] is None else h['memory'] for h in t['hyp']]);rigid=np.array([np.nan*np.ones(2) if h['rigid'] is None else h['rigid'] for h in t['hyp']])
    fig,ax=plt.subplots();ax.plot(objects[:,0],objects[:,1],label='Reference clean-RGB detector centroid')
    ax.plot(mem[:,0],mem[:,1],label='RGB-proxy memory');ax.plot(rigid[:,0],rigid[:,1],label='RGB-proxy rigid')
    ax.legend();ax.set_title('Same RGB proxy geometry; detector failure remains a limitation');ax.set_xlabel('u px');ax.set_ylabel('v px');fig.savefig(out/'01_trajectory_same_rgb_proxy.png',dpi=130,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots();ax.plot(tt,r,label='P(attached)');ax.plot(tt,np.where(t['y']>=0,t['y'],np.nan),'o',ms=3,label='Verified labels')
    ax.set_ylim(-.1,1.1);ax.set_xlabel('Step');ax.legend();fig.savefig(out/'02_attachment_probability.png',dpi=130,bbox_inches='tight');plt.close(fig)
    # Real data only; actual dropout length grouping, not synthetic interpolation.
    sizes=[];errs=[]
    for tr in traces:
        m=np.array([row['target_pixel_missing'] for row in tr['public']]);
        if not m.any():continue
        p0=pred[tr['trajectory']];good=m&(tr['y']>=0)&np.isfinite(p0)
        if good.any():sizes.append(int(m.sum()));errs.append(float(np.mean((p0[good]-tr['y'][good])**2)))
    fig,ax=plt.subplots();ax.scatter(sizes,errs);ax.set_xlabel('Total withheld-observation steps / trace');ax.set_ylabel('Hidden Brier score');fig.savefig(out/'03_dropout_duration_vs_brier.png',dpi=130,bbox_inches='tight');plt.close(fig)
    # Reliability diagram for verified held-out labels; no invented confidence bins.
    probs=[];ys=[]
    for tr in traces:
        pr=pred[tr['trajectory']]
        good=(tr['y']>=0)&np.isfinite(pr)
        probs.extend(pr[good]);ys.extend(tr['y'][good])
    fig,ax=plt.subplots();ax.plot([0,1],[0,1],'--',label='perfect calibration')
    if len(probs):
        probs=np.asarray(probs);ys=np.asarray(ys)
        for left,right in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            mask=(probs>=left)&(probs<right if right<1 else probs<=right)
            if mask.any():ax.scatter(float(probs[mask].mean()),float(ys[mask].mean()),s=max(20,int(mask.sum())))
    ax.set(xlabel='Mean predicted P(attached)',ylabel='Empirical attached frequency',xlim=(0,1),ylim=(0,1))
    fig.savefig(out/'05_contact_calibration.png',dpi=130,bbox_inches='tight');plt.close(fig)
    breakdown={}
    for tr in traces:
        p0=pred[tr['trajectory']]; idx=(tr['y']>=0)&np.isfinite(p0)
        if idx.any():breakdown.setdefault(tr['intervention'],[]).append(float(np.mean((p0[idx]-tr['y'][idx])**2)))
    fig,ax=plt.subplots();keys=list(breakdown);ax.bar(keys,[np.mean(breakdown[k]) for k in keys]);ax.set_ylabel('Brier score by intervention');ax.set_title('Diagnosed categories; not natural-failure rates');fig.savefig(out/'04_failure_categories.png',dpi=130,bbox_inches='tight');plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Directory containing public/ and eval_only/')
    p.add_argument('--output',required=True)
    p.add_argument('--history',type=int,default=6)
    p.add_argument('--seed',type=int,default=2026,help='Group split fixed independently of model seeds')
    p.add_argument('--train-seeds',default='7,17,37')
    p.add_argument('--calibrated-closed-aperture',type=float,default=None)
    p.add_argument('--calibrated-open-aperture',type=float,default=None)
    args=p.parse_args(argv)
    traces=load_pairs(Path(args.data)/'public',Path(args.data)/'eval_only',history=args.history)
    split=group_split(traces,args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest={k:sorted(set(t['group'] for t in v)) for k,v in split.items()}
    (out/'split_groups.json').write_text(json.dumps(manifest,indent=2))
    (out/'experiment_config.json').write_text(json.dumps(vars(args),indent=2))
    # All three partitions must have labels; do not report success on absent truth.
    for k in split:labeled_data(split[k])
    for k in ('train','val'):
        if len(np.unique(labeled_data(split[k])[1]))!=2:
            raise ValueError(f'{k} needs attached and detached verified labels')
    import joblib
    seeds=[int(v) for v in args.train_seeds.split(',')]
    if len(set(seeds))<3:raise ValueError('At least three DISTINCT training seeds required')
    candidates={}
    for s in seeds:
        for name,d in train_sklearn(split['train'],split['val'],s).items():
            candidates[(name,s)]=d
        candidates[('gru',s)]=train_gru(split['train'],split['val'],s)
    # Selection uses VAL ONLY. Do not examine test to select seed/model.
    best_name,best_seed=min(candidates,key=lambda x:candidates[x]['val_brier'])
    best=candidates[(best_name,best_seed)]
    if best_name=='gru':
        import torch
        torch.save(best['model'].state_dict(),out/'selected_gru_weights.pt')
        joblib.dump(dict(scaler=best['scaler'],feature_names=FEATURE_NAMES,
                         hidden_dim=32, input_dim=best['model_input_dim']),out/'selected_gru_scaler.joblib')
    else:
        joblib.dump(dict(model=best_name,seed=best_seed,fit=best,
                         feature_names=FEATURE_NAMES),out/'selected_model.joblib')
    # DEVELOPMENT decisions must use validation groups, not the touched internal test.
    comparisons=[]
    for tr in split['val']:
        predictions={
            'selected_'+best_name: probabilities(best,tr['X'],best_name),
            'gripper_lr': probabilities(candidates[('gripper_lr',best_seed)],tr['X'],'gripper_lr'),
            'memory':np.zeros(len(tr['y'])),
            'rigid':np.ones(len(tr['y']))}
        if args.calibrated_closed_aperture is not None and args.calibrated_open_aperture is not None:
            r=fixed_baselines(tr,args.calibrated_closed_aperture,args.calibrated_open_aperture)
            predictions['M2_style']=np.array([np.nan if v is None else v for v in r],float)
        comparisons.append(compare_on_shared_frames(tr,predictions))
    (out/'validation_comparison.json').write_text(json.dumps({
        'schema':'contact_validation_comparison_v2','source':'validation_development_only',
        'used_to_select_model':True,'not_independent_confirmation':True,
        'comparisons':comparisons,'selected_model':best_name,'seed':best_seed},indent=2))

    results=[];per_step=[];chosen_p={}
    for name in ('gripper_lr','full_lr','full_mlp','gru'):
        for s in seeds:
            d=candidates[(name,s)]
            for t in split['test']:
                pr=probabilities(d,t['X'],name)
                results.append({**trace_metrics(t,pr,strategy=name),
                    'seed':s,'selected':(name,s)==(best_name,best_seed),'val_brier':d['val_brier']})
                per_step.extend(trajectory_predictions(t,pr,method=name,model_seed=s,
                    history=args.history,split_seed=args.seed))
                if (name,s)==(best_name,best_seed):chosen_p[t['trajectory']]=pr
    for t in split['test']:
        for name,v in [('memory',0.),('rigid',1.)]:
            pr=np.full(len(t['y']),v)
            results.append({**trace_metrics(t,pr,strategy=name),'seed':0})
            per_step.extend(trajectory_predictions(t,pr,method=name,model_seed=0,
                history=args.history,split_seed=args.seed))
        if args.calibrated_closed_aperture is not None and args.calibrated_open_aperture is not None:
            raw=fixed_baselines(t,args.calibrated_closed_aperture,args.calibrated_open_aperture)
            pr=np.asarray([np.nan if v is None else v for v in raw])
            results.append({**trace_metrics(t,pr,strategy='M2_style'), 'seed':0})
            per_step.extend(trajectory_predictions(t,pr,method='M2_style',model_seed=0,
                history=args.history,split_seed=args.seed))
    if args.calibrated_closed_aperture is None or args.calibrated_open_aperture is None:
        print('WARNING: no calibrated M2-style rule baseline: incomplete comparative study')
    (out/'test_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    with (out/'test_predictions_EVAL_ONLY.jsonl').open('w',encoding='utf-8') as f:
        for row in per_step:f.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'selection.json').write_text(json.dumps(dict(selected_model=best_name,seed=best_seed,
        validation_brier=best['val_brier'],evaluation_code_version=source_version(),
        feature_names=FEATURE_NAMES,history=args.history,test_is_untouched_for_selection=True),indent=2))
    if chosen_p:save_figures(split['test'],chosen_p,out/'figures')
    print('Offline training finished:',out,'selected',best_name,best_seed,
          '(not a GPU closed-loop success measurement)')

if __name__=='__main__':main()
