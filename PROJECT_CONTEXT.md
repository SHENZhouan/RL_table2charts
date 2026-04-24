# Project Context

## Goal

This repository is a course-project adaptation of Table2Charts focused on reproducible experiments over the RL search-sampling stage rather than on large architectural rewrites.

The immediate objective is to make the current Plotly-based workflow reproducible and easy to manage before adding more algorithms or broader sweeps.

## Dataset Constraint

The original Excel training corpus used by Table2Charts is not public. As a result, this project trains and evaluates in a Plotly-only setting built from the public Plotly corpus and local preprocessing scripts.

This means:

- experiment conclusions are for the Plotly-only adaptation, not the original private Excel setup;
- branch discipline and experiment metadata matter because multiple variants are being compared on the same public substitute dataset;
- old logs and checkpoints remain useful references and should be preserved.

## Current Implemented Variants

The current codebase and run scripts in this branch include:

- SFT checkpoint plus original greedy evaluation;
- RL fine-tuned original greedy baseline;
- updated policy with epsilon top-M exploration;
- dense reward shaping;
- dense reward plus updated policy;
- actor-critic-style policy head experiments.

These are not yet organized under one normalized experiment-management layer, which is the reason for the scaffold added in this step.

## Planned Experiment Families

Planned near-term studies include:

- epsilon sweeps over more than one epsilon/top-M setting;
- alternative exploration methods such as Boltzmann or softmax sampling, Gumbel or noisy greedy, and possibly UCB-inspired count-based exploration;
- reward-weight sweeps to study why dense reward plus epsilon-greedy can underperform either change alone;
- actor-only, critic-only, and blended-score diagnostics from the same trained actor-critic checkpoint without retraining.

If an experiment family is not supported by the current code, the scaffold should represent it explicitly as planned work rather than pretending it is runnable.

## Reproducibility Priorities

Important reproducibility concerns for this project:

- avoid machine-specific hard-coded paths;
- keep configs versioned and readable;
- separate local editing from remote execution;
- preserve existing run logs, checkpoints, and teammate files;
- normalize results into one comparable schema;
- use dry-run command generation before launching expensive jobs on the remote GPU host.

## Workflow

Current expected workflow:

- local machine: WSL plus VS Code for reading, planning, editing, and lightweight validation;
- remote GPU host: installing dependencies, running training and evaluation, collecting outputs, and keeping long jobs alive with `tmux`;
- Git: integrate work on dedicated branches rather than detached HEAD or direct edits to unrelated experiment branches.

The final dependency snapshot is intentionally not generated in this step. When the environment is stable, export it as `project_requirements_groupID.txt`.
