"""Scientific result catalog over unchanged saved runs, not directory-name order."""
from . import artifacts as a

ANIMALS={'rhino':'קרנף · Rhino','camel':'גמל · Camel','flamingo':'פלמינגו · Flamingo','cows':'פרה · Cows'}
EXPERIMENTS={'baseline':'1 · LIFT4D baseline','2.1':'2.1 · Frozen mesh · final deformation','2.2':'2.2 · Frozen mesh · geometry deformation','2.3':'2.3 · Mesh fine-tuning','2.4':'2.4 · Mesh ARAP projection','2.5':'2.5 · Surface-aware frozen mesh','demo':'DEMO · Synthetic data'}
STAGES={'node':'geometry','node_delta':'appearance','geometry':'geometry','appearance':'appearance','mesh_finetune':'mesh_finetune'}
STAGE_LABELS={'geometry':'Geometry','appearance':'Appearance','mesh_finetune':'Fine-tuning','arap':'ARAP · source','surface_transfer':'Surface weights · source'}


def build(root, include_auxiliary=False):
    entries=[];excluded=[];issues=[]
    for run_id in a.runs(root):
        if run_id.startswith('results/'):continue
        try:
            detail=a.describe(root,run_id);meta=detail['metadata'];config=detail['config']
            experiment=str(meta.get('experiment',config.get('experiment',config.get('id','unknown'))))
            if detail['demo']:experiment='demo'
            technical=bool(detail['demo'] or config.get('smoke_check') or not meta.get('selected_objects')
                           or config.get('status')=='not_implemented')
            by_sequence={sequence:a.views(root/run_id,sequence) for sequence in detail['sequences']}
            # Unlaunched baseline stubs have an inventory but no produced artifacts.
            has_producer=bool(list((root/run_id/'logs').glob('*geometry.log')) or (root/run_id/'logs/losses.jsonl').is_file() or (root/run_id/'logs/projection.jsonl').is_file() or (root/run_id/'logs/transfer.jsonl').is_file())
            if not any(by_sequence.values()) and not has_producer:technical=True
            if technical and not include_auxiliary:
                excluded.append(run_id);continue
            for sequence,views in by_sequence.items():
                groups={}
                for view in views:
                    stage=STAGES.get(view.get('stage'),view.get('stage') or 'saved')
                    try:iteration=int(view['checkpoint'])
                    except (ValueError,TypeError,KeyError):iteration=None
                    key=(stage,iteration)
                    # One choice per scientific checkpoint; exact RGB/alpha manifest
                    # takes precedence over the original multi-panel comparison.
                    groups.setdefault(key,[]).append(view)
                if experiment=='2.3':
                    # Atomic saved fine-tuning checkpoints can exist before their
                    # renders. Expose them with N/A rather than hiding the state.
                    import time
                    for checkpoint_file in (root/run_id/'checkpoints').glob('delta-*.pt'):
                        try:step=int(checkpoint_file.stem.split('-')[1])
                        except (ValueError,IndexError):continue
                        if time.time()-checkpoint_file.stat().st_mtime<2:continue
                        groups.setdefault(('mesh_finetune',step),[{'id':'','kind':'missing','label':'Saved checkpoint; renders unavailable'}])
                if not groups:
                    source_stage={'2.4':'arap','2.5':'surface_transfer'}.get(experiment)
                    pending=(source_stage,int(config['source_checkpoint'])) if source_stage and config.get('source_checkpoint') else ('pending',None)
                    groups[pending]=[{'id':'','kind':'missing','label':'No settled saved render yet'}]
                for (stage,iteration),candidates in groups.items():
                    def preference(view):
                        complete=False
                        if view['kind']=='manifest':
                            complete=bool(a.read_json(a.inside(root/run_id,view['id'])).get('complete'))
                        return (view['kind']=='manifest',complete,view['id'])
                    chosen=max(candidates,key=preference)
                    manifest=a.read_json(a.inside(root/run_id,chosen['id'])) if chosen['kind']=='manifest' else None
                    complete=(manifest.get('complete') if detail['demo'] else bool(manifest.get('complete'))) if manifest else None
                    label=STAGE_LABELS.get(stage,stage.title())
                    checkpoint=f'{stage}:{iteration}'
                    checkpoint_label=f'{label} · {iteration:,}' if iteration is not None else ('Not available · No saved render' if chosen['kind']=='missing' else 'Saved render · checkpoint ID unavailable')
                    if complete is False:checkpoint_label+=' · partial'
                    if iteration is not None and chosen['kind']=='missing':checkpoint_label+=' · renders unavailable'
                    execution_label=meta.get('display_label')
                    if experiment=='2.4' and config.get('iterations'):
                        execution_label=f'{execution_label or "ARAP"} · {config["iterations"]} iterations'
                    entries.append({'animal':sequence,'animal_label':ANIMALS.get(sequence,sequence),'experiment':experiment,
                                    'experiment_label':EXPERIMENTS.get(experiment,experiment),'checkpoint':checkpoint,
                                    'checkpoint_label':checkpoint_label,'stage':stage,'iteration':iteration,
                                    'run':run_id,'view':chosen['id'],'kind':chosen['kind'],'complete':complete,
                                    'created_utc':meta.get('created_utc'),'execution_label':execution_label,'technical':technical,'demo':detail['demo'],
                                    'alternative_views':[v['id'] for v in candidates if v!=chosen]})
        except (OSError,ValueError,TypeError,KeyError) as error:
            issues.append({'run':run_id,'reason':str(error)})
    entries.sort(key=lambda e:(list(ANIMALS).index(e['animal']) if e['animal'] in ANIMALS else 100,e['animal'],list(EXPERIMENTS).index(e['experiment']) if e['experiment'] in EXPERIMENTS else 100,e['experiment'],
                               e['iteration'] if e['iteration'] is not None else -1,e['stage'],e['created_utc'] or '',e['run']))
    return {'entries':entries,'excluded_auxiliary_runs':len(excluded),'issues':issues,'root':str(root)}
