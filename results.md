## Plotly Updated Policy Only 20260422T082259Z

- started_utc: 2026-04-22T08:23:00Z
- finished_utc: 2026-04-22T12:47:03Z
- gpu_ids: 3,5,6,7
- sft_ckpt: /ssd/shenzhouan/Table2Charts/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt
- policy: epsilon_top_m
- policy_epsilon_start: 0.2
- policy_epsilon_end: 0.02
- policy_epsilon_decay: 0.8
- policy_explore_top_m: 5
- rl_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260422162508-2el192fd128.128GRUh-allCharts-RL
- eval_log_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260422162508-2el192fd128.128GRUh-allCharts-RL/evaluations/test-updated-policy-rl-plotly-small-20260422T082259Z

```text
04/22/2026 11:43:24 - INFO - Student 0(3776631) -   EP-0 train SUMMARY: elapsed=11845.8s | avg_loss=18.912955 (tn, fp, fn, tp)=[2069263   44529   94568   51792] precision=0.537702 recall=0.353867 f1=0.426832 | total_cnt=15232success_cnt=45883 #states(expanded, reached, cut, dropped, complete)=(41.13, 99.90, 0.00, 4.37, 22.36) t(perf, process)=(61.90s, 9.16s) final_stage_cnt=45883 R@1=0.2024497090425648 R@3=0.39007911426890135 R@5=0.49833271582067434 R@10=0.650284419065885
04/22/2026 12:02:15 - INFO - Student 0(3776631) -   EP-0 test/valid SUMMARY: elapsed=1129.1s | avg_loss=98.475365 (tn, fp, fn, tp)=[596555  16507  33543  19433] precision=0.540707 recall=0.366826 f1=0.437109 | total_cnt=1093success_cnt=6430 #states(expanded, reached, cut, dropped, complete)=(41.27, 107.58, 0.00, 5.05, 22.90) t(perf, process)=(42.41s, 6.54s) final_stage_cnt=6430 R@1=0.4416796267496112 R@3=0.7528771384136859 R@5=0.8188180404354588 R@10=0.8905132192846035
04/22/2026 12:47:00 - INFO - summary(@153333) -   Complete recall info: {'recall': {'@03': 0.7913641279159586, '@05': 0.8513054225243318, '@10': 0.9068438127606983, '@20': 0.955739224470879, '@01': 0.4196663061949637, 'all': 0.955970956279932}, 'first_rank': '2.78*12376', 'reached': 20.45566198053453, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
```

## Plotly Update Reward Experiments 20260422T163833Z

- started_utc: 2026-04-22T16:38:33Z
- gpu_ids: 3,4,5,6
- sft_ckpt: /ssd/shenzhouan/Table2Charts/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt
- reward: exact=0.95, default=0.05, same_token=0.10, field=0.15, same_field_type=0.35
- combo_policy: epsilon_start=0.2, epsilon_end=0.02, epsilon_decay=0.8, explore_top_m=5
- log_file: /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_update_reward_then_combo_20260422T163833Z.log


### Update Reward Only, Original Greedy Policy

- finished_utc: 2026-04-22T20:28:24Z
- eval_log_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260423003933-2el192fd128.128GRUh-allCharts-RL/evaluations/test-update-reward-only-plotly-small-20260422T163833Z

```text
04/22/2026 19:44:36 - INFO - Student 0(836687) -   EP-0 train SUMMARY: elapsed=11066.6s | avg_loss=39.375889 (tn, fp, fn, tp)=[2083491   36878  100535   42475] precision=0.535266 recall=0.297007 f1=0.382033 | total_cnt=15216success_cnt=45883 #states(expanded, reached, cut, dropped, complete)=(41.13, 100.11, 0.00, 4.50, 22.09) t(perf, process)=(57.81s, 9.01s) final_stage_cnt=45883 R@1=0.16411307020029206 R@3=0.3294248414445437 R@5=0.43835407449382124 R@10=0.6051478761196958
04/22/2026 20:00:13 - INFO - Student 0(836687) -   EP-0 test/valid SUMMARY: elapsed=936.1s | avg_loss=176.763018 (tn, fp, fn, tp)=[580822  15999  33403  18464] precision=0.535763 recall=0.355987 f1=0.427754 | total_cnt=1106success_cnt=6430 #states(expanded, reached, cut, dropped, complete)=(41.25, 104.88, 0.00, 3.39, 22.59) t(perf, process)=(35.31s, 6.17s) final_stage_cnt=6430 R@1=0.43157076205287714 R@3=0.7122861586314152 R@5=0.7822706065318819 R@10=0.8751166407465008
04/22/2026 20:28:23 - INFO - summary(@1294312) -   Complete recall info: {'recall': {'all': 0.9521087594623822, '@20': 0.9289355785570833, '@01': 0.3960296616715588, '@10': 0.8595705237138884, '@03': 0.6632164375096555, '@05': 0.7752973891549514}, 'first_rank': '3.76*12326', 'reached': 20.177274833925537, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
```

### Update Reward + Updated Policy, Greedy Evaluation

- finished_utc: 2026-04-22T23:52:44Z
- eval_log_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260423042902-2el192fd128.128GRUh-allCharts-RL/evaluations/test-update-reward-policy-plotly-small-20260422T163833Z

```text
04/22/2026 19:44:36 - INFO - Student 0(836687) -   EP-0 train SUMMARY: elapsed=11066.6s | avg_loss=39.375889 (tn, fp, fn, tp)=[2083491   36878  100535   42475] precision=0.535266 recall=0.297007 f1=0.382033 | total_cnt=15216success_cnt=45883 #states(expanded, reached, cut, dropped, complete)=(41.13, 100.11, 0.00, 4.50, 22.09) t(perf, process)=(57.81s, 9.01s) final_stage_cnt=45883 R@1=0.16411307020029206 R@3=0.3294248414445437 R@5=0.43835407449382124 R@10=0.6051478761196958
04/22/2026 20:00:13 - INFO - Student 0(836687) -   EP-0 test/valid SUMMARY: elapsed=936.1s | avg_loss=176.763018 (tn, fp, fn, tp)=[580822  15999  33403  18464] precision=0.535763 recall=0.355987 f1=0.427754 | total_cnt=1106success_cnt=6430 #states(expanded, reached, cut, dropped, complete)=(41.25, 104.88, 0.00, 3.39, 22.59) t(perf, process)=(35.31s, 6.17s) final_stage_cnt=6430 R@1=0.43157076205287714 R@3=0.7122861586314152 R@5=0.7822706065318819 R@10=0.8751166407465008
04/22/2026 20:28:23 - INFO - summary(@1294312) -   Complete recall info: {'recall': {'all': 0.9521087594623822, '@20': 0.9289355785570833, '@01': 0.3960296616715588, '@10': 0.8595705237138884, '@03': 0.6632164375096555, '@05': 0.7752973891549514}, 'first_rank': '3.76*12326', 'reached': 20.177274833925537, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
04/22/2026 19:44:36 - INFO - Student 0(836687) -   EP-0 train SUMMARY: elapsed=11066.6s | avg_loss=39.375889 (tn, fp, fn, tp)=[2083491   36878  100535   42475] precision=0.535266 recall=0.297007 f1=0.382033 | total_cnt=15216success_cnt=45883 #states(expanded, reached, cut, dropped, complete)=(41.13, 100.11, 0.00, 4.50, 22.09) t(perf, process)=(57.81s, 9.01s) final_stage_cnt=45883 R@1=0.16411307020029206 R@3=0.3294248414445437 R@5=0.43835407449382124 R@10=0.6051478761196958
04/22/2026 20:00:13 - INFO - Student 0(836687) -   EP-0 test/valid SUMMARY: elapsed=936.1s | avg_loss=176.763018 (tn, fp, fn, tp)=[580822  15999  33403  18464] precision=0.535763 recall=0.355987 f1=0.427754 | total_cnt=1106success_cnt=6430 #states(expanded, reached, cut, dropped, complete)=(41.25, 104.88, 0.00, 3.39, 22.59) t(perf, process)=(35.31s, 6.17s) final_stage_cnt=6430 R@1=0.43157076205287714 R@3=0.7122861586314152 R@5=0.7822706065318819 R@10=0.8751166407465008
04/22/2026 20:28:23 - INFO - summary(@1294312) -   Complete recall info: {'recall': {'all': 0.9521087594623822, '@20': 0.9289355785570833, '@01': 0.3960296616715588, '@10': 0.8595705237138884, '@03': 0.6632164375096555, '@05': 0.7752973891549514}, 'first_rank': '3.76*12326', 'reached': 20.177274833925537, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
04/22/2026 23:09:15 - INFO - Student 0(1353928) -   EP-0 train SUMMARY: elapsed=9555.8s | avg_loss=39.094407 (tn, fp, fn, tp)=[2085706   36252   98767   42334] precision=0.538696 recall=0.300026 f1=0.385403 | total_cnt=15240success_cnt=45883 #states(expanded, reached, cut, dropped, complete)=(41.13, 99.98, 0.00, 4.43, 22.21) t(perf, process)=(49.96s, 8.74s) final_stage_cnt=45883 R@1=0.16247847786761982 R@3=0.3265479589390406 R@5=0.4338643942200815 R@10=0.6005056338949066
04/22/2026 23:24:49 - INFO - Student 0(1353928) -   EP-0 test/valid SUMMARY: elapsed=933.1s | avg_loss=171.194962 (tn, fp, fn, tp)=[572279  12791  34884  16561] precision=0.564220 recall=0.321917 f1=0.409941 | total_cnt=1113success_cnt=6430 #states(expanded, reached, cut, dropped, complete)=(41.26, 102.99, 0.00, 3.77, 22.59) t(perf, process)=(35.16s, 6.14s) final_stage_cnt=6430 R@1=0.42954898911353034 R@3=0.7219284603421462 R@5=0.7925349922239502 R@10=0.8660964230171073
04/22/2026 23:52:43 - INFO - summary(@1722292) -   Complete recall info: {'recall': {'@05': 0.7108759462382203, 'all': 0.9368917040012359, '@03': 0.6595087285648077, '@01': 0.3999691024254596, '@10': 0.8295226324733509, '@20': 0.9043720067974664}, 'first_rank': '4.11*12129', 'reached': 19.983469797620888, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
```

- all_finished_utc: 2026-04-22T23:52:44Z


## Plotly Update Actor Actor-Critic 20260423T120014Z

- started_utc: 2026-04-23T12:00:15Z
- gpu_ids: 3,4,5,6
- sft_ckpt: /ssd/shenzhouan/Table2Charts/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt
- actor_loss_weight: 0.1
- entropy_weight: 0.001
- critic_score_weight: 0.5
- log_file: /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_update_actor_actor_critic_rl_eval_20260423T120014Z.log

### Update Actor eval rerun 20260423T1623Z_evalfix

- finished_utc: 2026-04-23T16:53:32Z
- model_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260423200116-update_actor-2el192fd128.128GRUh-allCharts-actor-critic-RL
- eval_log_dir: /ssd/shenzhouan/Table2Charts/Results/Models/20260423200116-update_actor-2el192fd128.128GRUh-allCharts-actor-critic-RL/evaluations/test-update_actor-actor-critic-plotly-small-20260423T1623Z_evalfix
- log_file: /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_update_actor_eval_only_20260423T1623Z_evalfix.log

```text
04/23/2026 16:53:31 - INFO - summary(@4052863) -   Complete recall info: {'recall': {'@10': 0.6923374015139812, '@01': 0.347674957515835, '@05': 0.6198825892167464, '@03': 0.6137803182450178, 'all': 0.7315773211802874, '@20': 0.7100262629383594}, 'first_rank': '3.54*9471', 'reached': 17.172253978062724, 'targets': 2.396647613162367, 'top': [1, 3, 5, 10, 20], 't_cnt': 12946}
```
