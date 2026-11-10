"""Fixed internal validation panels and scalar-only experiment tracking."""
from collections import defaultdict
import random


def validation_positions(dataset, *, per_stratum, seed):
    """Stratify by original cohort and endpoint; never consult external rows."""
    if dataset.split != 'validation':
        raise ValueError('Monitoring panel must be internal validation only')
    if per_stratum <= 0:
        raise ValueError('Positive finite per-stratum panel budget required')
    db = dataset.connect()
    try:
        rows = db.execute('SELECT i.position,e.cohort,e.endpoint FROM task_index i '
            'JOIN examples e USING(row_id) WHERE i.task=? AND i.split=? ORDER BY i.position',
            (dataset.task, 'validation'))
        groups = defaultdict(list)
        for original, cohort, endpoint in rows:
            position = dataset.logical_position(original)
            if position is not None:
                groups[(cohort, endpoint)].append(position)
    finally:
        db.close()
    chosen = []
    for key in sorted(groups):
        values = groups[key]
        chosen.extend(random.Random(str(seed)+repr(key)).sample(values, min(len(values), per_stratum)))
    return sorted(chosen)


def scalar_report(report, prefix):
    return {f'{prefix}/{group}/{key}':value
            for group, values in report.items() for key, value in values.items()
            if isinstance(value, (int, float))}


def start_tracking(settings, output, *, task, variant, contract, config, resume=False):
    if not settings.get('enabled', False):
        return None
    import wandb
    run = wandb.init(entity=settings['entity'], project=settings['project'],
        id=settings['run_ids'][task], name=f'{task}-{variant}-seed{config["seed"]}',
        group=settings['group'], job_type=task, dir=str(output),
        mode='online', resume='must' if resume else 'never',
        config=dict(task=task, variant=variant, training_contract_sha256=contract, **config),
        settings=wandb.Settings(init_timeout=60, disable_git=True, save_code=False, console='off'))
    run.define_metric('update')
    run.define_metric('train/*', step_metric='update')
    run.define_metric('validation_panel/*', step_metric='update')
    run.define_metric('validation_full/*', step_metric='update')
    return run
