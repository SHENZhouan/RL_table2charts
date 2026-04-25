import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results" / "raw_logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run-first experiment runner")
    parser.add_argument("--config", help="Path to one JSON config file")
    parser.add_argument("--config-dir", help="Directory containing JSON configs")
    parser.add_argument("--only", help="Only include configs whose filename contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def require_config_selection(args) -> None:
    if bool(args.config) == bool(args.config_dir):
        raise SystemExit("Use exactly one of --config or --config-dir.")


def load_configs(args) -> List[Path]:
    if args.config:
        return [Path(args.config).resolve()]

    config_dir = Path(args.config_dir).resolve()
    configs = sorted(config_dir.glob("*.json"))
    if args.only:
        configs = [path for path in configs if args.only in path.name]
    if not configs:
        raise SystemExit("No config files matched the current selection.")
    return configs


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def env_default(name: str, fallback: Optional[str]) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    return fallback


def resolve_defaults(config: Dict) -> Dict:
    paths = config.setdefault("paths", {})
    runtime = config.setdefault("runtime", {})
    root = env_default("ROOT", paths.get("root") or str(REPO_ROOT))
    root_path = Path(root)

    resolved = {
        "root": root,
        "python_bin": env_default("PYTHON_BIN", paths.get("python_bin") or "python"),
        "corpus_path": env_default("CORPUS_PATH", paths.get("corpus_path") or str(root_path / "Data" / "PlotlyTable2Charts")),
        "sft_ckpt": env_default("SFT_CKPT", paths.get("sft_ckpt")),
        "actor_critic_ckpt": env_default("ACTOR_CRITIC_CKPT", paths.get("checkpoint")),
        "model_save_path": env_default("MODEL_SAVE_PATH", paths.get("model_save_path") or str(root_path / "Results" / "Models")),
        "summary_path": env_default("SUMMARY_PATH", paths.get("summary_path") or str(root_path / "Results" / "summary")),
        "gpu_ids": env_default("GPU_IDS", runtime.get("gpu_ids")),
    }

    if not paths.get("code_dir"):
        resolved["code_dir"] = str(Path(resolved["root"]) / "Table2Charts")
    else:
        resolved["code_dir"] = paths["code_dir"]

    resolved["checkpoint"] = paths.get("checkpoint")
    resolved["output_dir"] = paths.get("output_dir")
    resolved["run_id"] = runtime.get("run_id")
    resolved["rl_nprocs"] = int(env_default("RL_NPROCS", str(runtime.get("rl_nprocs"))) or runtime.get("rl_nprocs") or 1)
    resolved["eval_nprocs"] = int(env_default("EVAL_NPROCS", str(runtime.get("eval_nprocs"))) or runtime.get("eval_nprocs") or 1)
    resolved["master_port"] = int(env_default("MASTER_PORT", str(runtime.get("master_port"))) or runtime.get("master_port") or 29500)
    runtime["rl_nprocs"] = resolved["rl_nprocs"]
    runtime["eval_nprocs"] = resolved["eval_nprocs"]
    runtime["master_port"] = resolved["master_port"]
    return resolved


def quote_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def mk_env_exports(resolved: Dict) -> Dict[str, str]:
    exports = {}
    for key in ("root", "python_bin", "corpus_path", "sft_ckpt", "actor_critic_ckpt", "model_save_path", "summary_path", "gpu_ids"):
        value = resolved.get(key)
        if value:
            exports[key.upper()] = value
    return exports


def build_eval_command(config: Dict, resolved: Dict, program: str, model_dir: str, checkpoint_name: str) -> List[str]:
    search = config["search"]
    runtime = config["runtime"]
    return [
        resolved["python_bin"],
        program,
        "-m",
        model_dir,
        "-f",
        checkpoint_name,
        "--model_name",
        "cp",
        "--model_size",
        "small",
        "--features",
        "all-fast",
        "--log_save_path",
        config["paths"]["output_dir"],
        "--search_type",
        search["search_type"],
        "--input_type",
        search["input_type"],
        "--previous_type",
        search["previous_type"],
        "--nprocs",
        str(runtime["eval_nprocs"]),
        "--nagents",
        "64",
        "--nthreads",
        "5",
        "--search_limits",
        search["search_limits"],
        "--corpus_path",
        resolved["corpus_path"],
        "--lang",
        "en",
    ] + (["--limit_search_group"] if search.get("limit_search_group") else [])


def build_command_plan(config: Dict, resolved: Dict) -> Dict:
    name = config["name"]
    variant = config["base_variant"]
    search = config["search"]
    sampling = config["sampling"]
    reward = config["reward"]
    actor_critic = config["actor_critic"]
    runtime = config["runtime"]
    output_dir = config["paths"]["output_dir"]

    if sampling["strategy"] in {"boltzmann", "ucb"}:
        return {
            "status": "planned_not_implemented",
            "todo": f"{sampling['strategy']} sampling is planned in configs but not implemented in the current codebase.",
            "commands": [],
        }

    if variant == "baseline_sft_eval":
        if not resolved["sft_ckpt"]:
            return {
                "status": "planned_not_implemented",
                "todo": "SFT_CKPT is required for the SFT greedy baseline evaluation config.",
                "commands": [],
            }
        model_dir = str(Path(resolved["sft_ckpt"]).resolve().parent)
        command = build_eval_command(config, resolved, "test_agent_mp.py", model_dir, Path(resolved["sft_ckpt"]).name)
        return {"status": "runnable", "todo": None, "commands": [command]}

    if variant == "baseline_rl_greedy":
        if not resolved["sft_ckpt"]:
            return {
                "status": "planned_not_implemented",
                "todo": "SFT_CKPT is required to warm-start the RL greedy baseline run.",
                "commands": [],
            }
        train_cmd = [
            resolved["python_bin"],
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(runtime["rl_nprocs"]),
            "--master_port",
            str(runtime["master_port"]),
            "script.py",
            "--corpus_path",
            resolved["corpus_path"],
            "--model_size=small",
            "--model_name=cp",
            "--features=all-fast",
            "--negative_weight=0.8",
            f"--search_limits={search['search_limits']}",
            "--epochs=1",
            "-m",
            resolved["model_save_path"],
            "-p",
            resolved["sft_ckpt"],
            "--summary_path",
            resolved["summary_path"],
            "--search_type",
            search["search_type"],
            "--input_type",
            search["input_type"],
            "--previous_type",
            search["previous_type"],
            "--lang=en",
            "--queue_mode=local",
            "--log_freq_agent=500",
            "--log_freq_batch=100",
            "--max_tables=64",
            "--min_memory=1000",
            "--memory_sample_size=64",
            "--memory_sample_rounds=2",
        ]
        eval_cmd = [
            "# TODO: helper-managed final evaluation: discover the produced RL checkpoint directory after training, then run test_agent_mp.py on the test split."
        ]
        return {"status": "runnable", "todo": "Training command shape is implemented; helper scripts must discover the produced RL model directory and run test_agent_mp.py for final test-set evaluation.", "commands": [train_cmd, eval_cmd]}

    if variant == "updated_policy":
        if not resolved["sft_ckpt"]:
            return {
                "status": "planned_not_implemented",
                "todo": "SFT_CKPT is required for updated policy experiments.",
                "commands": [],
            }
        train_cmd = [
            resolved["python_bin"],
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(runtime["rl_nprocs"]),
            "--master_port",
            str(runtime["master_port"]),
            "--module",
            "reinforce.updated_policy_learn_dist",
            "--corpus_path",
            resolved["corpus_path"],
            "--model_size=small",
            "--model_name=cp",
            "--features=all-fast",
            "--negative_weight=0.8",
            f"--search_limits={search['search_limits']}",
            "--epochs=1",
            "--model_save_path",
            resolved["model_save_path"],
            "-p",
            resolved["sft_ckpt"],
            "--summary_path",
            resolved["summary_path"],
            "--search_type",
            search["search_type"],
            "--input_type",
            search["input_type"],
            "--previous_type",
            search["previous_type"],
            "--lang=en",
            "--queue_mode=local",
            "--log_freq_agent=500",
            "--log_freq_batch=100",
            "--max_tables=64",
            "--min_memory=1000",
            "--memory_sample_size=64",
            "--memory_sample_rounds=2",
            f"--policy_epsilon_start={sampling['epsilon_start']}",
            f"--policy_epsilon_end={sampling['epsilon_end']}",
            f"--policy_epsilon_decay={sampling['epsilon_decay']}",
            f"--policy_explore_top_m={sampling['top_m']}",
        ]
        eval_cmd = [
            "# TODO: helper-managed final evaluation: discover the produced RL checkpoint directory after training, then run test_agent_mp.py on the test split."
        ]
        return {"status": "runnable", "todo": "Training command shape is implemented; helper scripts must discover the produced RL model directory and run test_agent_mp.py for final test-set evaluation.", "commands": [train_cmd, eval_cmd]}

    if variant == "update_reward":
        if not resolved["sft_ckpt"]:
            return {
                "status": "planned_not_implemented",
                "todo": "SFT_CKPT is required for dense reward experiments.",
                "commands": [],
            }
        train_cmd = [
            resolved["python_bin"],
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(runtime["rl_nprocs"]),
            "--master_port",
            str(runtime["master_port"]),
            "--module",
            "reinforce.update_reward_learn_dist",
            "--corpus_path",
            resolved["corpus_path"],
            "--model_size=small",
            "--model_name=cp",
            "--features=all-fast",
            "--negative_weight=0.8",
            f"--search_limits={search['search_limits']}",
            "--epochs=1",
            "--model_save_path",
            resolved["model_save_path"],
            "-p",
            resolved["sft_ckpt"],
            "--summary_path",
            resolved["summary_path"],
            "--search_type",
            search["search_type"],
            "--input_type",
            search["input_type"],
            "--previous_type",
            search["previous_type"],
            "--lang=en",
            "--queue_mode=local",
            "--log_freq_agent=500",
            "--log_freq_batch=100",
            "--max_tables=64",
            "--min_memory=1000",
            "--memory_sample_size=64",
            "--memory_sample_rounds=2",
            f"--update_reward_exact={reward['exact_reward']}",
            f"--update_reward_default={reward['default_reward']}",
            f"--update_reward_same_token={reward['same_token_reward']}",
            f"--update_reward_field={reward['field_reward']}",
            f"--update_reward_same_field_type={reward['same_field_type_reward']}",
            f"--update_reward_positive_threshold={reward['positive_threshold']}",
        ]
        eval_cmd = [
            "# TODO: helper-managed final evaluation: discover the produced RL checkpoint directory after training, then run test_agent_mp.py on the test split."
        ]
        return {"status": "runnable", "todo": "Training command shape is implemented; helper scripts must discover the produced RL model directory and run test_agent_mp.py for final test-set evaluation.", "commands": [train_cmd, eval_cmd]}

    if variant == "update_reward_policy":
        if not resolved["sft_ckpt"]:
            return {
                "status": "planned_not_implemented",
                "todo": "SFT_CKPT is required for dense reward + updated policy experiments.",
                "commands": [],
            }
        train_cmd = [
            resolved["python_bin"],
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(runtime["rl_nprocs"]),
            "--master_port",
            str(runtime["master_port"]),
            "--module",
            "reinforce.update_reward_policy_learn_dist",
            "--corpus_path",
            resolved["corpus_path"],
            "--model_size=small",
            "--model_name=cp",
            "--features=all-fast",
            "--negative_weight=0.8",
            f"--search_limits={search['search_limits']}",
            "--epochs=1",
            "--model_save_path",
            resolved["model_save_path"],
            "-p",
            resolved["sft_ckpt"],
            "--summary_path",
            resolved["summary_path"],
            "--search_type",
            search["search_type"],
            "--input_type",
            search["input_type"],
            "--previous_type",
            search["previous_type"],
            "--lang=en",
            "--queue_mode=local",
            "--log_freq_agent=500",
            "--log_freq_batch=100",
            "--max_tables=64",
            "--min_memory=1000",
            "--memory_sample_size=64",
            "--memory_sample_rounds=2",
            f"--update_reward_exact={reward['exact_reward']}",
            f"--update_reward_default={reward['default_reward']}",
            f"--update_reward_same_token={reward['same_token_reward']}",
            f"--update_reward_field={reward['field_reward']}",
            f"--update_reward_same_field_type={reward['same_field_type_reward']}",
            f"--update_reward_positive_threshold={reward['positive_threshold']}",
            f"--policy_epsilon_start={sampling['epsilon_start']}",
            f"--policy_epsilon_end={sampling['epsilon_end']}",
            f"--policy_epsilon_decay={sampling['epsilon_decay']}",
            f"--policy_explore_top_m={sampling['top_m']}",
        ]
        eval_cmd = [
            "# TODO: helper-managed final evaluation: discover the produced RL checkpoint directory after training, then run test_agent_mp.py on the test split."
        ]
        return {"status": "runnable", "todo": "Training command shape is implemented; helper scripts must discover the produced RL model directory and run test_agent_mp.py for final test-set evaluation.", "commands": [train_cmd, eval_cmd]}

    if variant == "actor_critic":
        score_mode = actor_critic["score_mode"]
        checkpoint = resolved["checkpoint"] or resolved["actor_critic_ckpt"]
        if not checkpoint:
            return {
                "status": "planned_not_implemented",
                "todo": "Actor-critic evaluation requires paths.checkpoint or ACTOR_CRITIC_CKPT.",
                "commands": [],
            }
        model_dir = str(Path(checkpoint).resolve().parent)
        command = [
            resolved["python_bin"],
            "update_actor_test_agent_mp.py",
            "-m",
            model_dir,
            "-f",
            Path(checkpoint).name,
            "--model_name",
            "cp",
            "--model_size",
            "small",
            "--features",
            "all-fast",
            "--log_save_path",
            output_dir,
            "--search_type",
            search["search_type"],
            "--input_type",
            search["input_type"],
            "--previous_type",
            search["previous_type"],
            "--nprocs",
            str(runtime["eval_nprocs"]),
            "--nagents",
            "64",
            "--nthreads",
            "5",
            "--search_limits",
            search["search_limits"],
            "--corpus_path",
            resolved["corpus_path"],
            "--lang",
            "en",
            "--score_mode",
            score_mode,
            "--critic_score_weight",
            str(actor_critic["critic_score_weight"]),
        ] + (["--limit_search_group"] if search.get("limit_search_group") else [])
        return {"status": "runnable", "todo": None, "commands": [command]}

    return {
        "status": "planned_not_implemented",
        "todo": f"Unknown or unsupported base_variant={variant}.",
        "commands": [],
    }


def print_plan(config_path: Path, config: Dict, resolved: Dict, plan: Dict) -> None:
    print(f"=== {config['name']} ===")
    print(f"config: {config_path}")
    print(f"status: {plan['status']}")
    if plan["todo"]:
        print(f"todo: {plan['todo']}")
    print("resolved_env:")
    for key, value in mk_env_exports(resolved).items():
        print(f"  {key}={value}")
    if plan["commands"]:
        print("commands:")
        for idx, command in enumerate(plan["commands"], start=1):
            if len(command) == 1 and command[0].startswith("# TODO:"):
                print(f"  {idx}. {command[0]}")
            else:
                print(f"  {idx}. {quote_command(command)}")
    print()


def run_plan(config: Dict, resolved: Dict, plan: Dict) -> int:
    if plan["status"] != "runnable":
        print(f"Skipping {config['name']}: {plan['todo']}")
        return 0

    log_dir = Path(config["paths"]["output_dir"])
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / f"{config['name']}.runner.log"

    env = os.environ.copy()
    env.update(mk_env_exports(resolved))

    with run_log.open("a", encoding="utf-8") as log_file:
        for command in plan["commands"]:
            if len(command) == 1 and command[0].startswith("# TODO:"):
                log_file.write(command[0] + "\n")
                continue
            log_file.write(quote_command(command) + "\n")
            log_file.flush()
            result = subprocess.run(
                command,
                cwd=resolved["code_dir"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return result.returncode
    return 0


def main():
    args = parse_args()
    require_config_selection(args)
    config_paths = load_configs(args)

    exit_code = 0
    for config_path in config_paths:
        config = load_json(config_path)
        resolved = resolve_defaults(config)
        plan = build_command_plan(config, resolved)
        print_plan(config_path, config, resolved, plan)
        if not args.dry_run:
            exit_code = run_plan(config, resolved, plan)
            if exit_code != 0:
                break
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
