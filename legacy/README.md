# legacy/ — the 2021 originals

These are the nine files exactly as they were written in 2021 (verified
byte-for-byte when they were moved here). They are kept for reference only; the
maintained implementation lives in `dso/` + `experiments/`.

**They do not produce valid results.** What went wrong, at a glance:

| Problem | Where | Effect |
| --- | --- | --- |
| Several "independent" buffers are the same tensor | `d_sgt.py:116-122`, `our_dsgt.py:118-126`, `d_saga_tracking.py:144-150` | `y -= last_y` zeroes the tracker; `x_prev` restore is a no-op. `d_sgt` and `d_saga_tracking` never train after step 1; `our_dsgt` uses the parameter vector as a gradient. Loss stays at ln 10 ≈ 2.303. |
| Async `broadcast` without `wait()` | every training loop | send buffer may be freed while in flight |
| `torch.save` into missing directories | end of every epoch | crashes after the first epoch even on a working cluster |
| Dataset loaded at import, in float64 | `load.py:13,35` | ~18 GiB of RAM for 8 workers |
| No evaluation, no seeds | everywhere | results are neither measured nor reproducible |

Run one of them only to reproduce the *old* behaviour, from inside this
directory (they use bare `import load as cifar`):

```bash
cd legacy
python d_sgd.py        # needs cifar-10-batches-py/ here and 8 GPUs
```

The refactored equivalents are:

| legacy | refactor |
| --- | --- |
| `d_sgd.py` | `dso/algos/dsgd.py` |
| `d_saga.py` | `dso/algos/dsaga.py` |
| `d_sgt.py` | `dso/algos/dsgt.py` |
| `our_dsgt.py` | `dso/algos/dsgt_reval.py` (canonical name `dsgt_reval`) |
| `d_saga_tracking.py` | `dso/algos/dsaga_gt.py` |
| `local_sgd.py` | `dso/algos/local_sgd.py` |
| `Period_global_average_sgd.py` | `dso/algos/periodic_avg.py` |
| `gragh.py` | `dso/common/topology.py` + `dso/common/comm.py` |
| `load.py` | `dso/common/data.py` |
