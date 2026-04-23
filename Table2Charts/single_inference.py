# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import torch
from data import SpecialTokens
from model import CopyNet, get_cp_config
from search.agent import BeamDrillDownAgent
from test_agent_mp import construct_data_config, construct_search_config, feed_batch_nn
from util import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent Test Search Agents")

    parser.add_argument("--df_path", required=True,
                        help='The path saved data feature.')
    parser.add_argument("--emb_path", required=True,
                        help='The path saved embedding info.')
    parser.add_argument("--model_path", required=True,
                        help='The path saved model.')
    parser.add_argument("--output_path", default="result.json",
                        help='The path to write inference results.')
    parser.add_argument("--max_steps", type=int, default=60,
                        help="Max search steps for single-table inference (CPU-friendly).")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto (CUDA if available else CPU), cpu, cuda, or cuda:N. "
             "On shared servers set CUDA_VISIBLE_DEVICES to one free GPU and use --device cuda.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


class pretend_args:
    def __init__(self):
        self.model_name = "cp"
        self.model_size = "medium"
        self.model_save_path = ""
        self.features = "all-fast"
        self.search_type = "allCharts"
        self.search_all_types = True
        self.previous_type = "all"
        self.input_type = "allCharts"
        self.test_type = None
        self.unified_ana_token = False
        self.field_permutation = False
        self.num_workers = 1
        self.nprocs = 1
        self.valid_batch_size = 1
        self.corpus_path = ""
        self.model_load_path = ""
        self.mode = 'FULL'
        self.lang = 'en'
        # Tighter limits keep single-table demo runs tractable on CPU (raise to e100-b4-na to match paper README).
        self.search_limits = "e50-b4-na"
        self.empirical_study = True
        self.empirical_corpus_path = self.corpus_path
        self.empirical_log_path = None
        self.test_field_selections = False
        self.test_design_choices = False
        self.limit_search_group = False
        self.bing = True
        self.web_table = False
        # Dict passed to BeamDrillDownAgent (not a corpus tUID): build DataTable from JSON in memory.
        self.inline_table_inference = True


class single_inference:
    def __init__(self, args_ori):
        args = pretend_args()
        args.model_load_path = args_ori.model_path
        self.max_steps = getattr(args_ori, "max_steps", 60)
        self.data_config = construct_data_config(args)
        self.special_tokens = SpecialTokens(self.data_config)
        self.search_config = construct_search_config(args, self.data_config)
        self.device = resolve_device(getattr(args_ori, "device", "auto"))
        cp_config = get_cp_config(self.data_config, args.model_size)
        self.model = CopyNet(cp_config)
        load_checkpoint(args.model_load_path, self.model, device="cpu")
        self.model.to(self.device)
        self.model.eval()

    def inference(self, info):
        agent = BeamDrillDownAgent(info, self.data_config, self.special_tokens, self.search_config)
        agent.is_single_inference = True
        result = []
        steps = 0
        while (not agent.done()) and (steps < self.max_steps):
            chosen = agent.step()
            funure = feed_batch_nn(chosen, self.model, self.device, self.data_config)
            result = agent.update(chosen, funure)
            steps += 1
        return result


if __name__ == '__main__':
    args = parse_args()
    df_path = args.df_path
    emb_path = args.emb_path
    with open(df_path, 'r', encoding='utf-8-sig') as f:
        table_dicts = json.load(f)
    with open(emb_path, "r", encoding="utf-8-sig") as f:
        embedding = json.load(f)
    info = {"table": table_dicts, "embeddings": embedding}
    inference = single_inference(args)
    result = inference.inference(info)
    with open(args.output_path, 'w', encoding='utf-8-sig') as f:
        json.dump(result, f)
