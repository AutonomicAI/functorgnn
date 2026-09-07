"""
Training loop for FunctorGNN.

Training creates a sequence of committed weight snapshots that form
the commit chain in CommitStore. Each checkpoint corresponds to a node
in the commit DAG (directed acyclic graph), and the projection π maps
each commit to the corresponding weight state.

This directly implements the Compatible Commit Sequence from
GNNEquiv Definition 13:
    π(c_{t+1}) = U(π(c_t))
where U is one gradient update step.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from . import maintenance, materialize, prepare_graph
from .prepared_store import PreparedStore
from .projection import list_projection_ids

def _task_and_problem_type(graphs: list) -> tuple[str, str]:
    """Return (task, problem_type) using the first graph as dataset metadata."""
    task = graphs[0].get("task", "cycle_detection")
    problem_type = "regression" if task == "esol" else "classification"
    return task, problem_type


def train(
    model: torch.nn.Module,
    graphs: list,
    commit_store,
    epochs: int = 30,
    lr: float = 0.01,
    commit_every: int = 10,
    seed: int = 0,
    prepared_dir: str = "prepared",
) -> list[tuple[int, str]]:
    """
    Train the model and commit weight snapshots at regular intervals.

    Parameters
    ----------
    model        : FunctorGNN to train (mutated in-place)
    graphs       : list of graph dicts from dataset.load_dataset
    commit_store : CommitStore — receives commits at each checkpoint
    epochs       : total training epochs
    lr           : learning rate
    commit_every : commit a snapshot every N epochs (and at final epoch)
    seed         : torch manual seed for reproducible training

    Returns
    -------
    List of (epoch, commit_id) tuples for every committed checkpoint.
    """
    torch.manual_seed(seed)

    raw_graphs = prepare_graph.prepare(graphs)
    task, problem_type = _task_and_problem_type(raw_graphs)

    # Backloaded path: construct and persist deterministic prepared artifacts
    # before training/inference.  The default projection preserves legacy
    # training behavior; registered projections are also materialized so
    # projection-aware demos can replay without frontloaded inference work.
    store = PreparedStore(prepared_dir)
    prepared = materialize.prepare(raw_graphs, projection=None)
    prepared_ids = [store.save(graph) for graph in prepared]
    all_projection_ids = ["default", *list_projection_ids()]
    for projection_id in all_projection_ids[1:]:
        for graph in materialize.prepare(raw_graphs, projection=projection_id):
            store.save(graph)

    optimizer = Adam(model.parameters(), lr=lr)
    committed: list[tuple[int, str]] = []
    is_regression = problem_type == "regression"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        squared_error = 0.0

        for g in prepared:
            optimizer.zero_grad()
            output = model(g.x, g.adj_norm, g.edge_attr)            # (num_classes,) or (1,)
            if is_regression:
                target = torch.tensor([float(g.label)], dtype=torch.float32)
                loss = F.mse_loss(output.view(1), target)
                squared_error += float((output.view(1) - target).pow(2).item())
            else:
                target = torch.tensor([int(g.label)], dtype=torch.long)
                loss = F.cross_entropy(output.unsqueeze(0), target)
                correct += int(output.argmax().item() == int(g.label))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(prepared)
        acc      = correct / len(prepared) if not is_regression else None
        rmse     = (squared_error / len(prepared)) ** 0.5 if is_regression else None

        is_checkpoint = (epoch % commit_every == 0) or (epoch == epochs)
        if is_checkpoint:
            metrics = {"loss": round(avg_loss, 6)}
            if is_regression:
                metrics["rmse"] = round(float(rmse), 6)
            else:
                metrics["acc"] = round(float(acc), 6)

            metadata = maintenance.update(
                epoch=epoch,
                metrics=metrics,
                task=task,
                problem_type=problem_type,
                prepared_count=len(prepared),
                prepared_ids=prepared_ids,
                canonical_hashes=[g.canonical_hash for g in prepared],
                projection_id="default",
            )

            cid = commit_store.commit(
                model,
                message  = f"epoch {epoch}",
                metadata = metadata,
            )
            committed.append((epoch, cid))
            metric = f"rmse={rmse:.4f}" if is_regression else f"acc={acc:.1%}"
            print(f"  epoch {epoch:3d} | loss={avg_loss:.4f} | {metric} | commit={cid}")
        elif epoch % 5 == 0:
            metric = f"rmse={rmse:.4f}" if is_regression else f"acc={acc:.1%}"
            print(f"  epoch {epoch:3d} | loss={avg_loss:.4f} | {metric}")

    return committed
