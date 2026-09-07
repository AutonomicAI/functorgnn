# Functor GNN

Commit-indexed, reproducible graph inference with a runnable fraud/back-GNN benchmark.

Functor GNN is a proof-of-architecture implementation of the **Functor Model** for Graph Neural Networks (GNNs). It separates model state from model execution by storing trained weight states in a commit-addressed store, then makes the selected commit part of the inference input. The repository also demonstrates deterministic replay, rollback, graph canonicalization, prepared steady-state execution, projection-aware inference, and probe-induced equivalence classes.

The primary runnable project is [`gnn/`](gnn/). The most prominent benchmark is the fraud-oriented back-GNN workflow in [`gnn/back_gnn/`](gnn/back_gnn/).

> **Scope:** This repository demonstrates an architecture and its execution properties. The included synthetic, chemistry, ESOL, BACE, projection, and fraud examples must not be interpreted as validated clinical, biological, molecular-activity, or production fraud-detection models.

## Repository layout

```text
functorgnn/
├── README.md
├── .gitignore
└── gnn/
    ├── run_all.sh                 # Canonical end-to-end workflow
    ├── cli.py                     # Dataset, training, inference, and replay CLI
    ├── requirements.txt
    ├── back_gnn/
    │   └── run_back_bench.sh      # Fraud/back-GNN benchmark entry point
    ├── functor_gnn/
    │   ├── canonicalize.py        # Canonicalization and adjacency preparation
    │   ├── chemistry.py           # Restricted atom/bond feature encoding
    │   ├── commit_store.py        # Commit ID to immutable weight state
    │   ├── dataset.py             # Dataset generation/loading
    │   ├── inference.py           # Commit-indexed inference and prepared runtime
    │   ├── logger.py              # Structured JSONL inference log
    │   ├── model.py               # Message-passing GNN
    │   ├── projection.py          # Projection lenses and probe responses
    │   └── train.py               # Training and checkpoint commits
    ├── data/                      # Generated locally; not source-controlled
    ├── commits/                   # Generated model commits/checkpoints
    └── logs/                      # Generated inference records
```

Generated directories and filenames may vary by workflow. The important boundary is that source and configuration belong in Git; generated datasets, checkpoints, caches, logs, and benchmark outputs do not.

## Quick start

From a fresh clone:

```bash
git clone <repository-url> functorgnn
cd functorgnn/gnn

# Optional when run_all.sh does not create/manage the environment itself:
python3 -m pip install -r requirements.txt

./run_all.sh
```

`run_all.sh` is the canonical end-to-end entry point. Run it from `gnn/` so that its relative paths resolve against the runnable project rather than a machine-specific directory.

## Fraud / back-GNN benchmark

The back-GNN benchmark exposes the fraud-oriented configuration selected as the best all-around benchmark workflow during development:

```bash
cd functorgnn/gnn
./back_gnn/run_back_bench.sh
```

The benchmark script is the supported entry point for that experiment. Keeping it behind one command makes the environment, data preparation, model configuration, and measurement sequence inspectable and repeatable. Benchmark reports should always record the code revision, dataset/configuration, seed, dependency versions, operating system, and hardware; do not compare results across environments without recording those differences.

No benchmark score is claimed in this README because results depend on the checked-out revision and execution environment. Use the output produced by the script as the result for the current run.

## Generated artifacts and `.pt` files

PyTorch `.pt` files in this project are generated artifacts—for example serialized datasets, trained weights, commit snapshots, and checkpoints. They should **not** be committed to Git.

The repository `.gitignore` should include at least:

```gitignore
*.pt
__pycache__/
*.py[cod]
```

The intended lifecycle is:

```text
clone source
  -> install dependencies
  -> generate or obtain permitted source data
  -> train/run the benchmark
  -> create local .pt artifacts
```

Before committing, verify that no generated binary artifacts are staged:

```bash
git status
git ls-files '*.pt'
```

If a future release requires a deliberately distributed pretrained model, document its provenance, license, integrity hash, size, and retrieval mechanism explicitly rather than silently adding it to ordinary source control.

## Core architecture

### Canonicalized inference

Let \(\mathcal C\) be an input category and \(\mathcal D\) an output category. A model is treated as a structure-preserving map

\[
F : \mathcal C \to \mathcal D.
\]

Before evaluation, an input is mapped to a canonical representative:

\[
X \longmapsto C(X) \longmapsto F(C(X)).
\]

If \(X \sim Y\) and canonicalization is consistent on the equivalence class, then

\[
C(X)=C(Y)
\quad\Longrightarrow\quad
F(C(X))=F(C(Y)).
\]

This removes irrelevant representational differences before inference. In the implementation, graph preparation includes deterministic feature handling, canonicalization, adjacency normalization, and consistent tensor ordering.

### Commit-indexed model state

Let \(K\) be a set of model commits and \(W\) the space of weight states. The commit store implements

\[
\pi : K \to W,
\]

where \(\pi(c)\) resolves a commit identifier \(c\) to one serialized model state. Commit-indexed inference is

\[
\widetilde F(c,G,\omega)=S(\pi(c),G,\omega),
\]

with graph \(G\), execution controls \(\omega\) (including the seed), and evaluator \(S\).

The commit is therefore an explicit argument rather than hidden mutable state. Subject to a fixed software/hardware environment and deterministic operators, this provides:

1. **Referential transparency** — execution is determined by explicit inputs rather than the current training process.
2. **Rollback** — selecting an earlier state means resolving an earlier commit.
3. **Replay** — a recorded commit, graph, seed, and configuration can be executed again.
4. **Auditability** — an inference record can identify the exact model state used.

Bit-identical replay is an empirical property of a controlled runtime, not a universal promise across arbitrary PyTorch versions, hardware, numerical kernels, or nondeterministic operators.

### Projection-aware inference

The inference tuple can include a deterministic projection context \(p\):

\[
\widetilde F(c,G,\omega,p)
=S\!\left(\pi(c),C(P_p(G)),\omega\right).
\]

Here \(P_p\) is a feature lens carrying metadata such as task, assay context, or reference frame. The demo registry includes contexts such as `assay_polarity`, `assay_aromaticity`, `assay_size`, `assay_hydrophobicity`, `assay_charge`, and `assay_topology`.

These projections are deterministic, hand-authored feature transformations. They demonstrate how contextual interpretation can be made explicit and replayable; they are not learned assays or validated biochemical models.

### Probe-induced equivalence

For a fixed suite of graphs, each projection can be mapped to a response vector \(Y(p)\). The demo declares two projections empirically equivalent at tolerance \(\tau\) when

\[
\lVert Y(p_i)-Y(p_j)\rVert_\infty \le \tau.
\]

Clustering under this rule illustrates response-behavior compression. The resulting classes depend on the probe suite and tolerance and are not theorem-level equivalence or molecular validation.

## Prepared steady-state runtime

The project separates the audit-oriented inference path from repeated steady-state execution.

`run_inference(...)` resolves a commit, fixes seeds, prepares a graph, performs inference, creates reproducibility metadata, and appends an audit record. `PreparedFunctorRuntime` moves invariant work out of the repeated forward path:

```python
runtime = PreparedFunctorRuntime.from_commit(store, commit_id)
prepared_graphs = runtime.prepare_graphs(graphs)
batch = runtime.prepare_batch(prepared_graphs)

logits = runtime.forward(batch)
```

Model construction, commit loading, projection application, canonicalization, adjacency normalization, edge-tensor permutation, and batch construction occur before repeated `forward` calls. The underlying message-passing, edge scaling, pooling, and prediction mathematics remain the same.

Use the prepared path for steady-state performance measurement. Use the complete logged path when measuring audit-oriented execution. A benchmark must state which path it measures.

## CLI workflow

The shell scripts are the recommended starting points. The CLI remains useful for focused experiments from `functorgnn/gnn`.

### Generate a dataset

```bash
python3 cli.py generate --dataset synthetic
```

Other repository-supported dataset selectors may include:

```bash
python3 cli.py generate --dataset chemistry
python3 cli.py generate --dataset esol
python3 cli.py generate --dataset bace
```

Availability can depend on the checked-out revision and optional data dependencies. The generated dataset is local and should remain untracked.

### Train and create model commits

```bash
python3 cli.py train --epochs 30 --commit-every 10
python3 cli.py commits
```

Training writes commit metadata and serialized weight states. A commit ID is derived from serialized model state and identifies the corresponding checkpoint within the local commit store.

### Infer and replay

```bash
python3 cli.py infer --graph cycle_00 --seed 42
python3 cli.py equiv-demo --graph cycle_00 --seed 42
python3 cli.py rollback-demo --graph cycle_00 --seed 42
```

To run against an earlier model state:

```bash
python3 cli.py infer --graph cycle_00 --commit <commit-id> --seed 42
```

### Projection and probe demonstrations

```bash
python3 cli.py infer --graph ethanol --projection assay_polarity --seed 42
python3 cli.py projection-demo --graph ethanol --seed 42
python3 cli.py probe-equiv-demo --tau 0.05
```

### Inspect the inference log

```bash
python3 cli.py log --tail 10
```

## Dataset demonstrations

### Synthetic cycle/tree task

The synthetic task classifies cyclic and acyclic graphs. Node features are derived deterministically from graph structure:

\[
f_v=
\left[
\deg(v),
\deg(v)^2,
\frac{1}{\deg(v)},
\mathbf 1_{\{\deg(v)=1\}}
\right].
\]

| Graph type | Label | Structural property |
|---|---:|---|
| Cycle \(C_n\) | 1 | Every node has degree 2 |
| Tree | 0 | A nontrivial finite tree has leaves |

The deterministic structural features support the canonicalization demonstration: isomorphic inputs represented consistently yield the same feature construction.

### Restricted chemistry task

The chemistry example is intentionally narrow and dependency-light. It uses explicit atom and bond fields rather than providing a production cheminformatics pipeline.

Supported atom types are `C`, `H`, `N`, `O`, `S`, `F`, `Cl`, and `Br`. Supported bond types are `single`, `double`, `triple`, and `aromatic`. Invalid self-bonds, duplicate bonds, unsupported fields, and formal charges outside `{-1, 0, +1}` are rejected.

Each atom is encoded as:

```text
[atom-type one-hot (8), formal-charge one-hot (3), aromatic flag, explicit degree]
```

This produces 13 node features. Bond types use one-hot edge features and deterministic demonstration-scale strengths:

| Bond | Strength |
|---|---:|
| Single | 1.0 |
| Double | 2.0 |
| Triple | 3.0 |
| Aromatic | 1.5 |

The demo does not parse SMILES, infer valence, add implicit hydrogens, generate conformers, model pharmacokinetics, or establish chemical feasibility. No personalized-medicine or human-body-chemistry conclusion should be drawn from its outputs.

### ESOL and BACE

ESOL and BACE entry points provide additional graph-data workflows when their required data and dependencies are available. They should be treated as reproducibility and architecture demonstrations unless a separate, documented validation protocol establishes endpoint quality, split methodology, uncertainty calibration, and domain applicability.

## Inference records

The audit-oriented path appends structured JSONL records under `gnn/logs/`. A record can include:

```json
{
  "run_id": "3fa8c12d",
  "commit_id": "f51e3b0c9a724d88",
  "seed": 42,
  "graph_id": "cycle_00",
  "projection_id": "default",
  "predicted_class": 1,
  "probs": [0.08241, 0.91759],
  "logits": [-1.20341, 1.40221],
  "timestamp": "2025-04-14T10:00:00.000000+00:00"
}
```

Treat numeric values above as schema examples, not benchmark claims.

## Reliability and reproducibility guidance

- Run entry-point scripts from `gnn/`.
- Pin and record dependency versions for published benchmark results.
- Record the Git revision, model commit, dataset identity/hash, seed, configuration, execution mode, hardware, and operating system.
- Keep generated model/data artifacts out of Git; regenerate them through documented workflows.
- Validate clean-clone execution before publishing a release.
- Test canonicalization, commit resolution, rollback, replay, CLI error handling, and generated-artifact exclusions.
- Do not silently compare audit-path timing with prepared-runtime timing.
- Do not claim cross-platform bit identity without testing the exact environments involved.

## Limitations

- This is a proof of architecture, not a production decision system.
- Fraud results require evaluation on representative, properly governed data before operational use.
- Projection contexts are hand-authored deterministic lenses.
- Probe-equivalence classes are empirical and suite-dependent.
- Chemistry examples do not validate mQSAR, molecular activity, personalized medicine, or physiological effects.
- Generated commits are local artifacts unless a separate artifact registry and retention policy are configured.
- CPU execution is supported; runtime and memory requirements depend on the selected dataset and benchmark configuration.

## Legal notice

This session-generated README is designated **“Not a Contribution”** under the Apache License 2.0. Repository licensing and contribution terms should be stated separately in the repository's `LICENSE` and contribution documentation.

