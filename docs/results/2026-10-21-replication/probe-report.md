# Decision-model probes

Generated 2026-10-23T02:19:46+00:00. What each probe tests and what each readout design predicts: [docs/replication.md](../../replication.md).

| target | description |
|---|---|
| decision-1 | Microsoft-Decision-1 on Microsoft Foundry (post-trained Qwen3.5-9B). |
| qwen3.5-9b | Frozen Qwen3.5-9B (Decision-1's base model), option-letter readout, no training. |
| qwen3.5-9b-sorted | Frozen Qwen3.5-9B, choice options shown in byte-sorted order like Decision-1 (R18). |
| r9-qwen9b | R9: Qwen3.5-9B decision-trained (LoRA, code labels, CE+Brier, KL to base, interface augmentations). |
| r9b-kl | R9b: Qwen3.5-9B decision-trained with a 'none' option, KL 1.0 to the frozen base. |
| r9b-nokl | R9b: Qwen3.5-9B decision-trained with a 'none' option, no KL. |
| r15-typed | R15: Qwen3.5-9B decision-trained on all three question types. |
| r17-routing | R17: R9b-kl recipe + Bitext banking/support routing data. |
| r19-candidate | R19 candidate replica: R9b-kl recipe retrained, byte-sorted options. |
| r20-scale | R20: R17 recipe at 5x steps, LoRA r64, KL 0.3, byte-sorted options. |
| qwen3.5-2b | Frozen Qwen3.5-2B (base of the Part 3 training arms), option-letter readout. |
| r3-ce-shuffle | Qwen3.5-2B + LoRA arm `ce_shuffle` (cross-entropy, options shuffled every example). |
| r3-ce-fixed | Qwen3.5-2B + LoRA arm `ce_fixed` (cross-entropy, options in canonical label order). |
| r3-brier-shuffle | Qwen3.5-2B + LoRA arm `brier_shuffle` (Brier score, options shuffled every example). |

## Forensics

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answers | 1624 | 854 | 900 | 984 | 984 | 984 | 1144 | 1144 | 1144 | 1144 | 712 | 712 | 712 | 712 |
| logit_grid_share / 0.125 | 0.7746 | 0.0094 | 0.0067 | 0.002 | 0.002 | 0.0061 | 0.0026 | 0.0009 | 0.0017 | 0.0017 | 0.0042 | 0.0056 | 0.0056 | 0.0042 |
| logit_grid_share / 0.25 | 0.1367 | 0.0035 | 0.0022 | 0.003 | 0.0041 | 0.0051 | 0.0017 | 0.0009 | 0.0035 | 0.0026 | 0.0056 | 0.0056 | 0.0028 | 0.0112 |
| logit_grid_share / 0.5 | 0.0326 | 0.0035 | 0.0044 | 0.0061 | 0.001 | 0.003 | 0.0017 | 0.0009 | 0.0009 | – | 0.0014 | 0.0028 | 0.007 | 0.0056 |
| logit_grid_share / 0.0625 | 0.0308 | – | – | – | 0.001 | 0.0051 | 0.0009 | 0.0017 | 0.0009 | 0.0052 | 0.007 | 0.0028 | 0.0042 | 0.0084 |
| logit_grid_share / 1 | 0.024 | 0.0059 | 0.0056 | 0.0091 | 0.0091 | 0.0122 | 0.014 | 0.0052 | 0.0044 | 0.007 | 0.007 | 0.014 | 0.0112 | 0.0042 |
| logit_grid_share / none | 0.0012 | 0.9707 | 0.9711 | 0.9746 | 0.9787 | 0.9614 | 0.9747 | 0.9869 | 0.9773 | 0.9729 | 0.9649 | 0.9593 | 0.9635 | 0.9579 |
| logit_grid_main_step | 0.125 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| bf16_logit_magnitude_if_main_step | [16, 32) | – | – | – | – | – | – | – | – | – | – | – | – | – |
| confidence_is_jev_formula | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| confidence_is_max_prob | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0105 | 0 | 0 | 0 | 0 |
| max_abs_sum_error | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| choice_is_argmax | 0.9982 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| output_tokens / 1 | 1592 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| output_tokens / 2 | 32 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| latency_p50_ms | 485 | 77.4 | 65.4 | 93 | 95.8 | 102.9 | 94.5 | 97.6 | 91.2 | 93.5 | 50.1 | 65.3 | 67.2 | 69.4 |
| latency_p95_ms | 886.3 | 337.4 | 335.8 | 456.5 | 467.5 | 466.5 | 463.7 | 465.8 | 461.8 | 495.5 | 163 | 198.5 | 199.8 | 197.8 |
| logit_grid_share / 0.03125 | – | 0.0035 | 0.0056 | 0.002 | 0.002 | 0.002 | 0.0035 | – | 0.0052 | 0.0017 | 0.0042 | 0.0042 | – | 0.0028 |
| logit_grid_share / 0.015625 | – | 0.0035 | 0.0044 | 0.003 | 0.002 | 0.0051 | 0.0009 | 0.0035 | 0.0061 | 0.0087 | 0.0056 | 0.0056 | 0.0056 | 0.0056 |
| output_tokens / 0 | – | 854 | 900 | 984 | 984 | 984 | 1144 | 1144 | 1144 | 1144 | 712 | 712 | 712 | 712 |

## order

_Does reordering the options change the decision? Is there a position bias?_ (decision-1: 176 requests, 0 errors, qwen3.5-9b: 176 requests, 0 errors, qwen3.5-9b-sorted: 176 requests, 0 errors, r9-qwen9b: 176 requests, 0 errors, r9b-kl: 176 requests, 0 errors, r9b-nokl: 176 requests, 0 errors, r15-typed: 176 requests, 0 errors, r17-routing: 176 requests, 0 errors, r19-candidate: 176 requests, 0 errors, r20-scale: 176 requests, 0 errors, qwen3.5-2b: 176 requests, 0 errors, r3-ce-shuffle: 176 requests, 0 errors, r3-ce-fixed: 176 requests, 0 errors, r3-brier-shuffle: 176 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / accuracy | 0.8125 | 0.7917 | 0.8125 | 0.6875 | 0.8125 | 0.8542 | 0.7292 | 0.7812 | 0.75 | 0.625 | 0.5104 | 0.6458 | 0.6771 | 0.625 |
| ace / flip_rate_vs_original | 0 | 0.075 | 0 | 0 | 0.075 | 0.1 | 0.05 | 0.0625 | 0 | 0 | 0.3125 | 0.1125 | 0.0625 | 0.075 |
| ace / items_consistent_all_orders | 1 | 0.875 | 1 | 1 | 0.8125 | 0.875 | 0.875 | 0.8125 | 1 | 1 | 0.0625 | 0.75 | 0.875 | 0.875 |
| ace / mean_tv_vs_original | 0.014 | 0.0874 | 0 | 0.038 | 0.0469 | 0.0477 | 0.046 | 0.0498 | 0 | 0 | 0.2629 | 0.0806 | 0.0697 | 0.0538 |
| ace / mean_chosen_position | 0.5 | 0.5052 | 0.5 | 0.5 | 0.4948 | 0.4896 | 0.4688 | 0.5 | 0.5 | 0.5 | 0.651 | 0.5104 | 0.5208 | 0.4896 |
| ace / mean_gold_position | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| ace / first_vs_last_log_lift | -0.0048 | -0.0247 | 0 | -0.3103 | -0.1499 | -0.0602 | -0.0506 | -0.2996 | 0 | 0 | -1.183 | -0.1178 | 0.0282 | -0.1927 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / accuracy | 1 | 0.9625 | 0.9375 | 1 | 0.9375 | 0.9875 | 1 | 0.95 | 0.9375 | 1 | 0.9625 | 0.9625 | 0.925 | 0.9375 |
| banking77 / flip_rate_vs_original | 0 | 0.0312 | 0 | 0 | 0 | 0.0156 | 0 | 0.0156 | 0 | 0 | 0.0312 | 0.0312 | 0.0625 | 0 |
| banking77 / items_consistent_all_orders | 1 | 0.9375 | 1 | 1 | 1 | 0.9375 | 1 | 0.9375 | 1 | 1 | 0.9375 | 0.9375 | 0.9375 | 1 |
| banking77 / mean_tv_vs_original | 0.0035 | 0.0418 | 0 | 0.0218 | 0.0277 | 0.0132 | 0.0205 | 0.025 | 0 | 0 | 0.1361 | 0.0303 | 0.0392 | 0.0245 |
| banking77 / mean_chosen_position | 0.4675 | 0.47 | 0.465 | 0.4675 | 0.465 | 0.465 | 0.4675 | 0.4675 | 0.465 | 0.4675 | 0.465 | 0.46 | 0.475 | 0.465 |
| banking77 / mean_gold_position | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 |
| banking77 / first_vs_last_log_lift | 0.0216 | -0.5527 | 0 | -0.3227 | -0.2905 | -0.6179 | -0.1211 | 0.3457 | 0 | 0 | 2.863 | 0.5928 | 1.172 | 1.183 |

## letter_names

_Do option names that look like answer letters confuse the readout?_ (decision-1: 128 requests, 0 errors, qwen3.5-9b: 128 requests, 0 errors, qwen3.5-9b-sorted: 128 requests, 0 errors, r9-qwen9b: 128 requests, 0 errors, r9b-kl: 128 requests, 0 errors, r9b-nokl: 128 requests, 0 errors, r15-typed: 128 requests, 0 errors, r17-routing: 128 requests, 0 errors, r19-candidate: 128 requests, 0 errors, r20-scale: 128 requests, 0 errors, qwen3.5-2b: 128 requests, 0 errors, r3-ce-shuffle: 128 requests, 0 errors, r3-ce-fixed: 128 requests, 0 errors, r3-brier-shuffle: 128 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / accuracy_letters_aligned | 0.8125 | 0.75 | 0.75 | 0.6875 | 0.875 | 0.875 | 0.6875 | 0.75 | 0.75 | 0.6875 | 0.5 | 0.6875 | 0.625 | 0.6875 |
| ace / accuracy_letters_shifted | 0.75 | 0.75 | 0.6875 | 0.6875 | 0.875 | 0.875 | 0.75 | 0.8125 | 0.8125 | 0.6875 | 0.5 | 0.6875 | 0.625 | 0.625 |
| ace / accuracy_neutral | 0.8125 | 0.75 | 0.6875 | 0.6875 | 0.875 | 0.9375 | 0.6875 | 0.8125 | 0.8125 | 0.6875 | 0.4375 | 0.6875 | 0.625 | 0.6875 |
| ace / accuracy_original | 0.8125 | 0.8125 | 0.8125 | 0.6875 | 0.875 | 0.875 | 0.6875 | 0.75 | 0.75 | 0.625 | 0.4375 | 0.6875 | 0.625 | 0.6875 |
| ace / agree_with_neutral_letters_aligned | 1 | 1 | 0.875 | 1 | 1 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.8125 | 0.9375 | 1 | 1 | 1 |
| ace / agree_with_neutral_letters_shifted | 0.9375 | 1 | 1 | 1 | 1 | 0.9375 | 0.9375 | 1 | 1 | 0.9375 | 0.9375 | 0.9375 | 1 | 0.9375 |
| ace / agree_with_neutral_original | 1 | 0.9375 | 0.8125 | 1 | 1 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.875 | 1 | 1 | 1 | 1 |
| ace / shifted_disagreements | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| ace / collision_pull | 0 | – | – | – | – | 1 | 0 | – | – | 0 | 0 | 0 | – | 0 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / accuracy_letters_aligned | 1 | 1 | 1 | 1 | 0.9375 | 1 | 1 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / accuracy_letters_shifted | 1 | 1 | 0.9375 | 1 | 0.9375 | 1 | 1 | 1 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.6875 | 0.9375 |
| banking77 / accuracy_neutral | 0.9375 | 1 | 0.9375 | 1 | 0.9375 | 1 | 1 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / accuracy_original | 0.9375 | 1 | 0.9375 | 1 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / agree_with_neutral_letters_aligned | 0.9375 | 1 | 0.9375 | 1 | 1 | 1 | 1 | 0.9375 | 1 | 0.9375 | 1 | 1 | 1 | 1 |
| banking77 / agree_with_neutral_letters_shifted | 0.9375 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.9375 | 1 | 1 | 0.625 | 1 |
| banking77 / agree_with_neutral_original | 0.875 | 1 | 1 | 1 | 1 | 0.9375 | 1 | 0.9375 | 1 | 1 | 1 | 1 | 1 | 1 |
| banking77 / shifted_disagreements | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 6 | 0 |
| banking77 / collision_pull | 0 | – | – | – | – | – | – | – | – | 0 | – | – | 0 | – |

## duplicate

_How is probability split between two identical options?_ (decision-1: 96 requests, 0 errors, qwen3.5-9b: 96 requests, 0 errors, qwen3.5-9b-sorted: 96 requests, 0 errors, r9-qwen9b: 96 requests, 0 errors, r9b-kl: 96 requests, 0 errors, r9b-nokl: 96 requests, 0 errors, r15-typed: 96 requests, 0 errors, r17-routing: 96 requests, 0 errors, r19-candidate: 96 requests, 0 errors, r20-scale: 96 requests, 0 errors, qwen3.5-2b: 96 requests, 0 errors, r3-ce-shuffle: 96 requests, 0 errors, r3-ce-fixed: 96 requests, 0 errors, r3-brier-shuffle: 96 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / earlier_copy_share_dup_after | 0.8326 | 0.9674 | 0.9674 | 0.9651 | 0.9838 | 0.9982 | 0.9702 | 0.9956 | 0.9847 | 1 | 0.6887 | 0.8149 | 0.7931 | 0.7468 |
| ace / earlier_copy_share_dup_before | 0.1594 | 0.5943 | 0.0326 | 0.0179 | 0.0266 | 0.0118 | 0.1122 | 0.0113 | 0.0153 | 0 | 0.5646 | 0.7448 | 0.7112 | 0.661 |
| ace / original_name_share_dup_after | 0.8326 | 0.9674 | 0.9674 | 0.9651 | 0.9838 | 0.9982 | 0.9702 | 0.9956 | 0.9847 | 1 | 0.6887 | 0.8149 | 0.7931 | 0.7468 |
| ace / original_name_share_dup_before | 0.8406 | 0.4057 | 0.9674 | 0.9821 | 0.9734 | 0.9882 | 0.8878 | 0.9887 | 0.9847 | 1 | 0.4354 | 0.2552 | 0.2888 | 0.339 |
| ace / median_abs_log_ratio | 1.875 | 2.56 | 3.661 | 4.158 | 4.52 | 6.222 | 3.497 | 5.482 | 4.457 | 11.24 | 0.9537 | 1.587 | 1.194 | 0.9856 |
| ace / mass_conservation | 1.088 | 0.8553 | 0.8497 | 1.904 | 1.398 | 1.309 | 1.194 | 1.242 | 1.174 | 4.191 | 1.654 | 1.202 | 1.215 | 1.227 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / earlier_copy_share_dup_after | 0.8535 | 0.8803 | 0.8809 | 0.9851 | 0.9916 | 0.9972 | 0.9874 | 0.9975 | 0.9918 | 1 | 0.8714 | 0.9168 | 0.9352 | 0.8448 |
| banking77 / earlier_copy_share_dup_before | 0.1458 | 0.2969 | 0.1191 | 0.0021 | 0.0043 | 0.0025 | 0.0113 | 0.0049 | 0.0082 | 0 | 0.563 | 0.5037 | 0.4875 | 0.4385 |
| banking77 / original_name_share_dup_after | 0.8535 | 0.8803 | 0.8809 | 0.9851 | 0.9916 | 0.9972 | 0.9874 | 0.9975 | 0.9918 | 1 | 0.8714 | 0.9168 | 0.9352 | 0.8448 |
| banking77 / original_name_share_dup_before | 0.8542 | 0.7031 | 0.8809 | 0.9979 | 0.9957 | 0.9975 | 0.9887 | 0.9951 | 0.9918 | 1 | 0.437 | 0.4963 | 0.5125 | 0.5615 |
| banking77 / median_abs_log_ratio | 1.875 | 1.572 | 2.546 | 5.325 | 5.694 | 6.4 | 5.2 | 7.003 | 4.892 | 13.08 | 1.912 | 1.208 | 1.476 | 1.284 |
| banking77 / mass_conservation | 1.021 | 0.9978 | 1.037 | 1.026 | 1.145 | 1.09 | 1.029 | 1.049 | 1.132 | 1.117 | 1.055 | 1.009 | 0.9747 | 0.9808 |

## cardinality

_How do accuracy, tokens and latency scale with the number of options?_ (decision-1: 132 requests, 2 errors, qwen3.5-9b: 132 requests, 0 errors, r9-qwen9b: 132 requests, 0 errors, r9b-kl: 132 requests, 0 errors, r9b-nokl: 132 requests, 0 errors, r15-typed: 132 requests, 0 errors, r17-routing: 132 requests, 0 errors, r19-candidate: 132 requests, 0 errors, r20-scale: 132 requests, 0 errors, qwen3.5-2b: 132 requests, 0 errors, r3-ce-shuffle: 132 requests, 0 errors, r3-ce-fixed: 132 requests, 0 errors, r3-brier-shuffle: 132 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per_k / 2 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 2 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 2 / accuracy | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| per_k / 2 / mean_p_gold | 0.9931 | 0.9855 | 0.9846 | 0.9774 | 0.9827 | 0.9848 | 0.9782 | 0.9707 | 0.9818 | 0.9656 | 0.9981 | 0.9971 | 0.9927 |
| per_k / 2 / input_tokens | 64.75 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 |
| per_k / 2 / latency_p50_ms | 453.7 | 64 | 81.9 | 84.4 | 92 | 83.6 | 86.9 | 81.4 | 80.8 | 45.1 | 60.8 | 61.2 | 60.5 |
| per_k / 4 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 4 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 4 / accuracy | 0.9375 | 1 | 1 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 4 / mean_p_gold | 0.9381 | 0.9731 | 0.9699 | 0.9505 | 0.9605 | 0.9627 | 0.9615 | 0.9503 | 0.9539 | 0.9295 | 0.9416 | 0.9395 | 0.9438 |
| per_k / 4 / input_tokens | 83.12 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 |
| per_k / 4 / latency_p50_ms | 471.7 | 65.1 | 82.9 | 87 | 93.2 | 83.7 | 87.2 | 81.5 | 81.4 | 46.9 | 61.5 | 61.4 | 60.9 |
| per_k / 8 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 8 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 8 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 1 | 0.8125 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 8 / mean_p_gold | 0.9225 | 0.9288 | 0.9361 | 0.9279 | 0.9626 | 0.9381 | 0.9306 | 0.9451 | 0.9561 | 0.722 | 0.9365 | 0.9065 | 0.9166 |
| per_k / 8 / input_tokens | 125.5 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 |
| per_k / 8 / latency_p50_ms | 459.9 | 68 | 85.4 | 89.6 | 96 | 85.9 | 90 | 84.5 | 83.5 | 48.7 | 64 | 65.4 | 63.8 |
| per_k / 16 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 16 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 16 / accuracy | 1 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.875 | 0.9375 | 0.9375 | 0.875 |
| per_k / 16 / mean_p_gold | 0.928 | 0.9133 | 0.9335 | 0.9139 | 0.9165 | 0.9177 | 0.912 | 0.93 | 0.9226 | 0.7767 | 0.9001 | 0.9032 | 0.8781 |
| per_k / 16 / input_tokens | 206.4 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 |
| per_k / 16 / latency_p50_ms | 478.5 | 72.8 | 89.6 | 92 | 98.5 | 90 | 94.9 | 87.6 | 86.5 | 51.8 | 66.8 | 66.8 | 66.7 |
| per_k / 26 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 26 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 26 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.625 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 26 / mean_p_gold | 0.929 | 0.9213 | 0.9228 | 0.9177 | 0.9316 | 0.9284 | 0.9254 | 0.9099 | 0.9171 | 0.5578 | 0.908 | 0.8953 | 0.8896 |
| per_k / 26 / input_tokens | 306.9 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 |
| per_k / 26 / latency_p50_ms | 471.9 | 77.6 | 94 | 98.1 | 105.8 | 96.9 | 98.9 | 93.5 | 94.8 | 55.8 | 71.4 | 71.5 | 71.1 |
| per_k / 27 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 27 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 27 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.875 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 27 / mean_p_gold | 0.9104 | 0.9176 | 0.9132 | 0.9153 | 0.9159 | 0.9118 | 0.9167 | 0.909 | 0.9201 | 0.7677 | 0.9036 | 0.8933 | 0.879 |
| per_k / 27 / input_tokens | 317.5 | 466.1 | 275.8 | 275.8 | 275.8 | 275.8 | 275.8 | 275.8 | 275.8 | 466.4 | 466.2 | 466.4 | 466.5 |
| per_k / 27 / latency_p50_ms | 452.2 | 203.6 | 94.8 | 97.3 | 104.9 | 96.8 | 100.3 | 93.1 | 95.4 | 142.8 | 194 | 195 | 189.9 |
| per_k / 40 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 40 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 40 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.8125 | 0.875 | 0.875 | 0.9375 |
| per_k / 40 / mean_p_gold | 0.9201 | 0.9181 | 0.907 | 0.8994 | 0.9098 | 0.9081 | 0.9191 | 0.897 | 0.9118 | 0.6992 | 0.8388 | 0.8458 | 0.8343 |
| per_k / 40 / input_tokens | 447.4 | 555.1 | 365 | 365 | 365 | 365 | 365 | 365 | 365 | 556.2 | 555.2 | 555.8 | 555.8 |
| per_k / 40 / latency_p50_ms | 480.6 | 212.7 | 100.9 | 103.3 | 110.8 | 101 | 103.9 | 100.8 | 106.5 | 151.9 | 199 | 200.2 | 196.9 |
| per_k / 77 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 77 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 77 / accuracy | 0.8125 | 0.9375 | 0.9375 | 0.875 | 0.8125 | 0.9375 | 0.9375 | 0.875 | 0.875 | 0.8125 | 0.875 | 0.875 | 0.9375 |
| per_k / 77 / mean_p_gold | 0.8485 | 0.8693 | 0.8244 | 0.7705 | 0.8226 | 0.8226 | 0.8503 | 0.8183 | 0.8186 | 0.6773 | 0.8214 | 0.8226 | 0.7992 |
| per_k / 77 / input_tokens | 816.9 | 903.9 | 618.1 | 618.1 | 618.1 | 618.1 | 618.1 | 618.1 | 618.1 | 904.6 | 903.4 | 903.9 | 903.7 |
| per_k / 77 / latency_p50_ms | 470.8 | 296.7 | 151.8 | 153.7 | 153.4 | 152.6 | 153.2 | 152.1 | 164.1 | 211.6 | 272.8 | 278.9 | 274.2 |
| tokens_per_option | 10.04 | 11.11 | 6.88 | 6.88 | 6.88 | 6.88 | 6.88 | 6.88 | 6.88 | 11.13 | 11.11 | 11.12 | 11.11 |
| tokens_at_k0 | 45.1 | 66.4 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 66.3 | 66.5 | 66.4 | 66.5 |
| tokens_per_option_16_to_26 | 10.05 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 |
| tokens_per_option_26_to_27 | 10.62 | 198.2 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 198.6 | 198.4 | 198.6 | 198.7 |
| latency_ms_per_option | 0.14 | 3.38 | 0.89 | 0.87 | 0.79 | 0.89 | 0.84 | 0.91 | 1.09 | 2.41 | 3.1 | 3.17 | 3.11 |
| limits / k256 | error: HTTPError: HTTP 400: {"error":{"code":"unsupported_request_argument","message":"Model does not support request argument  | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted |
| limits / k255 | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted | accepted |

## question

_Does the model read the question, or only the state and options?_ (decision-1: 160 requests, 0 errors, qwen3.5-9b: 160 requests, 0 errors, qwen3.5-9b-sorted: 160 requests, 0 errors, r9-qwen9b: 160 requests, 0 errors, r9b-kl: 160 requests, 0 errors, r9b-nokl: 160 requests, 0 errors, r15-typed: 160 requests, 0 errors, r17-routing: 160 requests, 0 errors, r19-candidate: 160 requests, 0 errors, r20-scale: 160 requests, 0 errors, qwen3.5-2b: 160 requests, 0 errors, r3-ce-shuffle: 160 requests, 0 errors, r3-ce-fixed: 160 requests, 0 errors, r3-brier-shuffle: 160 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77 / accuracy_listed_first | 0.5 | 0.875 | 0.5 | 0.625 | 0.9375 | 1 | 0.9375 | 0.9062 | 0.5 | 0.5 | 0.5625 | 0.5312 | 0.5312 | 0.5312 |
| banking77 / accuracy_listed_last | 0.5 | 1 | 0.5 | 1 | 1 | 0.9688 | 1 | 1 | 0.5 | 0.5 | 1 | 0.5312 | 0.5 | 0.4688 |
| banking77 / accuracy_match | 1 | 0.9688 | 0.9375 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| banking77 / accuracy_not_match | 1 | 0.9688 | 0.9375 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.5 | 0.7812 | 0.625 | 0.8438 |
| banking77 / same_as_match_listed_first | 0.5 | 0.6562 | 0.625 | 0.875 | 0.5625 | 0.5 | 0.5625 | 0.5938 | 0.5 | 0.9375 | 0.9375 | 0.9688 | 0.9688 | 0.9688 |
| banking77 / same_as_match_listed_last | 0.5 | 0.4688 | 0.4375 | 0.5 | 0.5 | 0.4688 | 0.5 | 0.5 | 0.5 | 0.6875 | 0.5 | 0.9688 | 0.9375 | 0.9688 |
| banking77 / same_as_match_not_match | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 | 0.2188 | 0.375 | 0.1562 |
| ace / accuracy_listed_first | 1 | 0.8125 | 0.8125 | 0.4375 | 0.1875 | 0.3125 | 0.75 | 0.0625 | 0.3125 | 0 | 0 | 0.1875 | 0.25 | 0.1875 |
| ace / accuracy_listed_last | 0.75 | 1 | 1 | 0.3125 | 1 | 1 | 1 | 1 | 1 | 1 | 0.5625 | 0.3125 | 0.1875 | 0.5 |

## empty_state

_What does the model answer when the state carries no information?_ (decision-1: 20 requests, 0 errors, qwen3.5-9b: 20 requests, 0 errors, qwen3.5-9b-sorted: 20 requests, 0 errors, r9-qwen9b: 20 requests, 0 errors, r9b-kl: 20 requests, 0 errors, r9b-nokl: 20 requests, 0 errors, r15-typed: 20 requests, 0 errors, r17-routing: 20 requests, 0 errors, r19-candidate: 20 requests, 0 errors, r20-scale: 20 requests, 0 errors, qwen3.5-2b: 20 requests, 0 errors, r3-ce-shuffle: 20 requests, 0 errors, r3-ce-fixed: 20 requests, 0 errors, r3-brier-shuffle: 20 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77 / empty / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / empty / norm_entropy | 0.9533 | 0.7487 | 0.7694 | 0.9908 | 0.9823 | 0.9804 | 0.9868 | 0.9763 | 0.9957 | 0.9991 | 0.6734 | 0.8245 | 0.8507 | 0.8753 |
| banking77 / empty / p_first_position | 0.1612 | 0.2942 | 0.121 | 0.12 | 0.1442 | 0.1468 | 0.1209 | 0.0935 | 0.1292 | 0.1224 | 0.3986 | 0.2166 | 0.286 | 0.2166 |
| banking77 / empty / p_last_position | 0.161 | 0.341 | 0.121 | 0.1115 | 0.1764 | 0.1845 | 0.1861 | 0.1974 | 0.1292 | 0.1224 | 0.0271 | 0.1096 | 0.094 | 0.097 |
| banking77 / empty / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / empty / top_value | pending_top_up | pending_top_up | order_physical_card | transfer_fee_charged | pending_top_up | pending_top_up | pending_top_up | pending_top_up | order_physical_card | pending_top_up | transfer_timing | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge |
| banking77 / empty / top_value_p | 0.2006 | 0.4108 | 0.3084 | 0.1749 | 0.1838 | 0.1953 | 0.1754 | 0.1708 | 0.1519 | 0.1365 | 0.2172 | 0.2956 | 0.2505 | 0.244 |
| banking77 / na / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / na / norm_entropy | 0.9083 | 0.7762 | 0.8092 | 0.9915 | 0.9799 | 0.98 | 0.9859 | 0.9763 | 0.9925 | 0.9994 | 0.6821 | 0.8454 | 0.7424 | 0.907 |
| banking77 / na / p_first_position | 0.1392 | 0.2648 | 0.1059 | 0.1176 | 0.1547 | 0.1499 | 0.1321 | 0.1071 | 0.1268 | 0.1227 | 0.4332 | 0.2902 | 0.4858 | 0.2118 |
| banking77 / na / p_last_position | 0.1488 | 0.3307 | 0.1059 | 0.1114 | 0.1834 | 0.1806 | 0.19 | 0.1949 | 0.1268 | 0.1227 | 0.0426 | 0.0709 | 0.0379 | 0.0945 |
| banking77 / na / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / na / top_value | beneficiary_not_allowed | pending_top_up | order_physical_card | transfer_fee_charged | pending_top_up | pending_top_up | pending_top_up | pending_top_up | beneficiary_not_allowed | pending_top_up | transfer_timing | top_up_by_bank_transfer_charge | pending_top_up | transfer_fee_charged |
| banking77 / na / top_value_p | 0.2926 | 0.3999 | 0.3045 | 0.173 | 0.1926 | 0.1883 | 0.1817 | 0.1758 | 0.1583 | 0.1345 | 0.2434 | 0.2176 | 0.2997 | 0.2261 |
| banking77 / no_info / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / no_info / norm_entropy | 0.8958 | 0.7396 | 0.7342 | 0.9914 | 0.9874 | 0.9845 | 0.9887 | 0.9762 | 0.9961 | 0.9994 | 0.6716 | 0.8805 | 0.8441 | 0.9104 |
| banking77 / no_info / p_first_position | 0.1575 | 0.2966 | 0.0995 | 0.1171 | 0.129 | 0.144 | 0.1203 | 0.091 | 0.1266 | 0.1229 | 0.423 | 0.2357 | 0.3303 | 0.2186 |
| banking77 / no_info / p_last_position | 0.1653 | 0.3452 | 0.0995 | 0.1113 | 0.1702 | 0.1774 | 0.1777 | 0.1881 | 0.1266 | 0.1229 | 0.0317 | 0.0776 | 0.0536 | 0.0879 |
| banking77 / no_info / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / no_info / top_value | beneficiary_not_allowed | pending_top_up | order_physical_card | transfer_fee_charged | pending_top_up | pending_top_up | pending_top_up | pending_top_up | order_physical_card | pending_top_up | pending_top_up | top_up_by_bank_transfer_charge | pending_top_up | transfer_fee_charged |
| banking77 / no_info / top_value_p | 0.287 | 0.4574 | 0.439 | 0.1749 | 0.1656 | 0.1808 | 0.1695 | 0.1638 | 0.1463 | 0.1349 | 0.23 | 0.2366 | 0.2415 | 0.2214 |
| banking77 / lorem / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / lorem / norm_entropy | 0.9081 | 0.7637 | 0.8971 | 0.9951 | 0.9894 | 0.9851 | 0.98 | 0.9695 | 0.9974 | 0.9994 | 0.5907 | 0.8703 | 0.8669 | 0.916 |
| banking77 / lorem / p_first_position | 0.1533 | 0.2694 | 0.1637 | 0.1204 | 0.1641 | 0.1398 | 0.1313 | 0.1143 | 0.1265 | 0.1225 | 0.549 | 0.2029 | 0.2901 | 0.1927 |
| banking77 / lorem / p_last_position | 0.1533 | 0.3475 | 0.1637 | 0.1107 | 0.154 | 0.1893 | 0.206 | 0.2215 | 0.1265 | 0.1225 | 0.0178 | 0.0563 | 0.0399 | 0.057 |
| banking77 / lorem / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / lorem / top_value | beneficiary_not_allowed | pending_top_up | beneficiary_not_allowed | transfer_fee_charged | pending_top_up | pending_top_up | pending_top_up | pending_top_up | beneficiary_not_allowed | pending_top_up | pending_top_up | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge | transfer_fee_charged |
| banking77 / lorem / top_value_p | 0.27 | 0.4174 | 0.327 | 0.1603 | 0.1738 | 0.1829 | 0.1946 | 0.2008 | 0.1523 | 0.1327 | 0.2845 | 0.256 | 0.2078 | 0.2366 |
| ace / empty / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / empty / norm_entropy | 0.8623 | 0.2084 | 0.1914 | 0.991 | 0.0877 | 0.0851 | 0.1942 | 0.0835 | 0.3444 | 0.1185 | 0.8739 | 0.6301 | 0.5326 | 0.8358 |
| ace / empty / p_first_position | 0.3163 | 0.3556 | 0.3333 | 0.3248 | 0.3385 | 0.3388 | 0.3409 | 0.3342 | 0.3333 | 0.3333 | 0.2474 | 0.3342 | 0.3106 | 0.3499 |
| ace / empty / p_last_position | 0.3432 | 0.3365 | 0.3333 | 0.3406 | 0.3285 | 0.3252 | 0.3274 | 0.3335 | 0.3333 | 0.3333 | 0.4154 | 0.3466 | 0.3255 | 0.3409 |
| ace / empty / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / empty / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Compliant | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / empty / top_value_p | 0.5998 | 0.9477 | 0.9537 | 0.3687 | 0.9829 | 0.983 | 0.9548 | 0.9841 | 0.9051 | 0.9759 | 0.5444 | 0.7688 | 0.8166 | 0.6251 |
| ace / na / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / na / norm_entropy | 0.0428 | 0.2118 | 0.1923 | 0.9986 | 0.0352 | 0.0841 | 0.1644 | 0.0589 | 0.046 | 0.0415 | 0.9112 | 0.3496 | 0.2739 | 0.5678 |
| ace / na / p_first_position | 0.3332 | 0.3437 | 0.3333 | 0.3206 | 0.335 | 0.3357 | 0.3405 | 0.3336 | 0.3333 | 0.3333 | 0.3338 | 0.3211 | 0.324 | 0.3315 |
| ace / na / p_last_position | 0.3334 | 0.3408 | 0.3333 | 0.342 | 0.3334 | 0.3317 | 0.3266 | 0.335 | 0.3333 | 0.3333 | 0.3629 | 0.3444 | 0.3291 | 0.345 |
| ace / na / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / na / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Compliant | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / na / top_value_p | 0.9928 | 0.9483 | 0.9541 | 0.3445 | 0.9943 | 0.984 | 0.9632 | 0.9895 | 0.9923 | 0.9931 | 0.5326 | 0.9012 | 0.9289 | 0.8061 |
| ace / no_info / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / no_info / norm_entropy | 0.0156 | 0.1514 | 0.1591 | 0.994 | 0.0409 | 0.0961 | 0.1493 | 0.0519 | 0.0433 | 0.0754 | 0.5943 | 0.2884 | 0.2411 | 0.5252 |
| ace / no_info / p_first_position | 0.3334 | 0.3423 | 0.3333 | 0.3217 | 0.3351 | 0.3404 | 0.3388 | 0.3329 | 0.3333 | 0.3333 | 0.3412 | 0.3238 | 0.3238 | 0.3349 |
| ace / no_info / p_last_position | 0.333 | 0.334 | 0.3333 | 0.3454 | 0.3329 | 0.3251 | 0.3276 | 0.3343 | 0.3333 | 0.3333 | 0.3548 | 0.3418 | 0.3259 | 0.3413 |
| ace / no_info / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / no_info / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Compliant | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / no_info / top_value_p | 0.9978 | 0.9665 | 0.9644 | 0.359 | 0.9932 | 0.9804 | 0.9675 | 0.991 | 0.9928 | 0.9861 | 0.7868 | 0.9235 | 0.9386 | 0.8271 |
| ace / lorem / errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / lorem / norm_entropy | 0.0331 | 0.0795 | 0.0804 | 0.993 | 0.061 | 0.0329 | 0.1541 | 0.0405 | 0.0623 | 0.0326 | 0.6469 | 0.2794 | 0.2786 | 0.3496 |
| ace / lorem / p_first_position | 0.3337 | 0.338 | 0.3333 | 0.3288 | 0.3382 | 0.335 | 0.3359 | 0.3341 | 0.3333 | 0.3333 | 0.3354 | 0.3196 | 0.3207 | 0.325 |
| ace / lorem / p_last_position | 0.333 | 0.3337 | 0.3333 | 0.3323 | 0.332 | 0.3323 | 0.3315 | 0.3345 | 0.3333 | 0.3333 | 0.3514 | 0.3393 | 0.3281 | 0.3421 |
| ace / lorem / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / lorem / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / lorem / top_value_p | 0.9946 | 0.9848 | 0.9845 | 0.3722 | 0.9888 | 0.9947 | 0.9666 | 0.9933 | 0.9889 | 0.9948 | 0.7575 | 0.9253 | 0.9255 | 0.9023 |

## multi_question

_Is the state encoded once and shared across questions in one request?_ (decision-1: 64 requests, 0 errors, qwen3.5-9b: 64 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b |
|---|---|---|
| items | 16 | 16 |
| shared_tokens_mean | 718.1 | 0 |
| shared_fraction_of_q1_request | 0.8227 | 0 |
| marginal_tokens_same_question_again | 116 | 884.5 |
| max_abs_dp_q1_alone_vs_combined | 0.0476 | 0 |
| max_abs_dp_q2_alone_vs_combined | 0.04913 | 0 |
| max_abs_dp_duplicate_question | 0 | 0 |
| latency_ratio_combined_vs_q1 | 1.112 | 1.913 |

## determinism

_Are repeated identical requests answered identically?_ (decision-1: 60 requests, 0 errors, qwen3.5-9b: 60 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b |
|---|---|---|
| banking77 / items | 6 | 6 |
| banking77 / identical_all_repeats | 0 | 1 |
| banking77 / max_abs_dp | 0.04344 | 0 |
| banking77 / choice_flips | 0 | 0 |
| ace / items | 6 | 6 |
| ace / identical_all_repeats | 0 | 1 |
| ace / max_abs_dp | 0.02553 | 0 |
| ace / choice_flips | 0 | 0 |

## tokens

_Which tokenizer is billed, and how large is the hidden prompt?_ (decision-1: 18 requests, 0 errors, qwen3.5-9b: 18 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b |
|---|---|---|
| billed_input_tokens / hello | 40 | 89 |
| billed_input_tokens / japanese | 45 | 94 |
| billed_input_tokens / apple_x02 | 41 | 90 |
| billed_input_tokens / digits | 49 | 98 |
| billed_input_tokens / sql | 50 | 99 |
| billed_input_tokens / spaces | 42 | 91 |
| billed_input_tokens / apple_x32 | 71 | 120 |
| billed_input_tokens / hello_world | 43 | 92 |
| billed_input_tokens / code | 51 | 99 |
| billed_input_tokens / apple_x16 | 55 | 104 |
| billed_input_tokens / x | 40 | 89 |
| billed_input_tokens / russian | 46 | 95 |
| billed_input_tokens / apple_x04 | 43 | 92 |
| billed_input_tokens / pi | 55 | 104 |
| billed_input_tokens / apple_x01 | 40 | 89 |
| billed_input_tokens / url | 48 | 97 |
| billed_input_tokens / emoji | 46 | 95 |
| billed_input_tokens / apple_x08 | 47 | 96 |
| tokens_per_apple | 1 | 1 |
| tokenizer_fits / Qwen/Qwen3.5-9B / hidden_prompt_tokens | 39 | 88 |
| tokenizer_fits / Qwen/Qwen3.5-9B / exact_fit_fraction | 0.667 | 0.611 |
| tokenizer_fits / openai/gpt-oss-20b / hidden_prompt_tokens | 38 | 87 |
| tokenizer_fits / openai/gpt-oss-20b / exact_fit_fraction | 0.333 | 0.333 |
| best_tokenizer | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B |

## duplicate_names

_What makes a model prefer one of two identical options?_ (decision-1: 160 requests, 0 errors, qwen3.5-9b: 160 requests, 0 errors, qwen3.5-9b-sorted: 160 requests, 0 errors, r9-qwen9b: 160 requests, 0 errors, r9b-kl: 160 requests, 0 errors, r9b-nokl: 160 requests, 0 errors, r15-typed: 160 requests, 0 errors, r17-routing: 160 requests, 0 errors, r19-candidate: 160 requests, 0 errors, r20-scale: 160 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale |
|---|---|---|---|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / first_share_neutral | 0.4505 | 0.9222 | 0.349 | 0.4219 | 0.4335 | 0.499 | 0.7289 | 0.3599 | 0.3992 | 0.3988 |
| ace / original_share_suffix_copy | 0.8499 | 0.9645 | 0.9645 | 0.9735 | 0.9885 | 0.9985 | 0.9725 | 0.9934 | 0.9887 | 1 |
| ace / original_share_unrelated | 0.7784 | 0.9788 | 0.976 | 0.9656 | 0.9851 | 0.9987 | 0.9727 | 0.9876 | 0.9795 | 0.998 |
| ace / pair_mass_neutral | 0.7656 | 0.7007 | 0.6899 | 0.7137 | 0.8973 | 0.9207 | 0.7312 | 0.8282 | 0.7897 | 0.0271 |
| ace / pair_mass_suffix_copy | 0.7901 | 0.7323 | 0.7323 | 0.675 | 0.829 | 0.8867 | 0.7256 | 0.7676 | 0.7656 | 0.9999 |
| ace / pair_mass_swap_after | 0.7926 | 0.6995 | 0.7109 | 0.95 | 0.9398 | 0.9803 | 0.8296 | 0.9562 | 0.7675 | 1 |
| ace / pair_mass_swap_before | 0.7965 | 0.7109 | 0.7109 | 0.6643 | 0.819 | 0.8998 | 0.7168 | 0.7755 | 0.7675 | 1 |
| ace / pair_mass_unrelated | 0.7774 | 0.7234 | 0.7291 | 0.6683 | 0.7959 | 0.8478 | 0.7202 | 0.7698 | 0.7333 | 0.8587 |
| ace / plain_name_share_swap_after | 0.8413 | 0.4057 | 0.9674 | 0.9821 | 0.9734 | 0.9882 | 0.8878 | 0.9887 | 0.9847 | 1 |
| ace / plain_name_share_swap_before | 0.8412 | 0.9674 | 0.9674 | 0.9651 | 0.9838 | 0.9982 | 0.9702 | 0.9956 | 0.9847 | 1 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / first_share_neutral | 0.4249 | 0.3894 | 0.4943 | 0.3442 | 0.4394 | 0.3773 | 0.3996 | 0.4463 | 0.4929 | 0.502 |
| banking77 / original_share_suffix_copy | 0.911 | 0.8294 | 0.8372 | 0.9947 | 0.997 | 0.9994 | 0.9924 | 0.998 | 0.9964 | 1 |
| banking77 / original_share_unrelated | 0.994 | 0.9962 | 0.9896 | 0.9971 | 0.9967 | 0.997 | 0.9977 | 0.9974 | 0.9975 | 0.9971 |
| banking77 / pair_mass_neutral | 0.838 | 0.6135 | 0.5937 | 0.59 | 0.4791 | 0.5036 | 0.4633 | 0.4532 | 0.4608 | 0.2058 |
| banking77 / pair_mass_suffix_copy | 0.9481 | 0.9676 | 0.9371 | 0.9803 | 0.9955 | 0.9997 | 0.969 | 0.9838 | 0.9952 | 1 |
| banking77 / pair_mass_swap_after | 0.9465 | 0.9646 | 0.9455 | 0.9983 | 0.9944 | 0.9986 | 0.988 | 0.9893 | 0.9881 | 1 |
| banking77 / pair_mass_swap_before | 0.9475 | 0.9663 | 0.9455 | 0.9767 | 0.9882 | 0.9991 | 0.9676 | 0.9795 | 0.9881 | 1 |
| banking77 / pair_mass_unrelated | 0.9433 | 0.9808 | 0.9495 | 0.9698 | 0.9518 | 0.9672 | 0.9635 | 0.9625 | 0.9563 | 0.9562 |
| banking77 / plain_name_share_swap_after | 0.8525 | 0.7031 | 0.8809 | 0.9979 | 0.9957 | 0.9975 | 0.9887 | 0.9951 | 0.9918 | 1 |
| banking77 / plain_name_share_swap_before | 0.8567 | 0.8803 | 0.8809 | 0.9851 | 0.9916 | 0.9972 | 0.9874 | 0.9975 | 0.9918 | 1 |

## long_options

_Are long option descriptions truncated?_ (decision-1: 112 requests, 0 errors, qwen3.5-9b: 112 requests, 0 errors, r9-qwen9b: 112 requests, 0 errors, r9b-kl: 112 requests, 0 errors, r9b-nokl: 112 requests, 0 errors, r15-typed: 112 requests, 0 errors, r17-routing: 112 requests, 0 errors, r19-candidate: 112 requests, 0 errors, r20-scale: 112 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | r9-qwen9b | r9b-kl | r9b-nokl | r15-typed | r17-routing | r19-candidate | r20-scale |
|---|---|---|---|---|---|---|---|---|---|
| per_variant / end_010 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / end_010 / accuracy | 0.875 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_variant / end_010 / mean_p_gold | 0.9232 | 0.9458 | 0.9505 | 0.9436 | 0.9523 | 0.9526 | 0.9511 | 0.9323 | 0.9217 |
| per_variant / end_010 / input_tokens | 536.9 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 |
| per_variant / end_040 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / end_040 / accuracy | 0.875 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_variant / end_040 / mean_p_gold | 0.9231 | 0.9587 | 0.9429 | 0.9448 | 0.9483 | 0.9468 | 0.9517 | 0.9304 | 0.9087 |
| per_variant / end_040 / input_tokens | 1857 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 |
| per_variant / end_150 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / end_150 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 1 |
| per_variant / end_150 / mean_p_gold | 0.9121 | 0.9457 | 0.9416 | 0.944 | 0.951 | 0.9514 | 0.9521 | 0.931 | 0.9026 |
| per_variant / end_150 / input_tokens | 6697 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 |
| per_variant / start_000 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / start_000 / accuracy | 1 | 1 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 1 |
| per_variant / start_000 / mean_p_gold | 0.9375 | 0.9666 | 0.957 | 0.9484 | 0.9595 | 0.9557 | 0.9569 | 0.9399 | 0.9539 |
| per_variant / start_000 / input_tokens | 96.88 | 152.1 | 152.1 | 152.1 | 152.1 | 152.1 | 152.1 | 152.1 | 152.1 |
| per_variant / start_010 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / start_010 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_variant / start_010 / mean_p_gold | 0.9175 | 0.9558 | 0.9456 | 0.9431 | 0.9487 | 0.9485 | 0.9493 | 0.9289 | 0.942 |
| per_variant / start_010 / input_tokens | 536.9 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 | 592.1 |
| per_variant / start_040 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / start_040 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_variant / start_040 / mean_p_gold | 0.9194 | 0.9383 | 0.9413 | 0.9447 | 0.9469 | 0.9475 | 0.9469 | 0.9211 | 0.9409 |
| per_variant / start_040 / input_tokens | 1857 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 | 1912 |
| per_variant / start_150 / n | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| per_variant / start_150 / accuracy | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 1 | 0.9375 | 0.9375 | 1 |
| per_variant / start_150 / mean_p_gold | 0.9226 | 0.9508 | 0.9396 | 0.9414 | 0.9481 | 0.9519 | 0.9456 | 0.9246 | 0.9404 |
| per_variant / start_150 / input_tokens | 6697 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 | 6752 |
| tokens_per_filler_repeat | 44 | 44 | 44 | 44 | 44 | 44 | 44 | 44 | 44 |

## paraphrase

_Does rewording the question change the decision?_ (decision-1: 500 requests, 0 errors, qwen3.5-9b: 500 requests, 0 errors, qwen3.5-9b-sorted: 160 requests, 0 errors, r15-typed: 160 requests, 0 errors, r17-routing: 160 requests, 0 errors, r19-candidate: 160 requests, 0 errors, r20-scale: 160 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-9b-sorted | r15-typed | r17-routing | r19-candidate | r20-scale |
|---|---|---|---|---|---|---|---|
| ace / items | 50 | 50 | 16 | 16 | 16 | 16 | 16 |
| ace / per_variant / original / accuracy | 0.84 | 0.88 | 0.8125 | 0.6875 | 0.75 | 0.75 | 0.625 |
| ace / per_variant / minimal / accuracy | 0.82 | 0.86 | 0.8125 | 0.75 | 0.8125 | 0.6875 | 0.625 |
| ace / per_variant / minimal / flip_rate | 0.08 | 0.02 | 0 | 0.0625 | 0.0625 | 0.0625 | 0 |
| ace / per_variant / minimal / mean_tv | 0.0357 | 0.0511 | 0.0463 | 0.0332 | 0.0323 | 0.062 | 0.0236 |
| ace / per_variant / para1 / accuracy | 0.82 | 0.84 | 0.75 | 0.6875 | 0.8125 | 0.6875 | 0.625 |
| ace / per_variant / para1 / flip_rate | 0.06 | 0.04 | 0.0625 | 0 | 0.0625 | 0.0625 | 0 |
| ace / per_variant / para1 / mean_tv | 0.0329 | 0.0446 | 0.0653 | 0.0479 | 0.0399 | 0.0408 | 0.0279 |
| ace / per_variant / para2 / accuracy | 0.84 | 0.86 | 0.75 | 0.6875 | 0.8125 | 0.75 | 0.625 |
| ace / per_variant / para2 / flip_rate | 0.06 | 0.02 | 0.0625 | 0 | 0.0625 | 0 | 0 |
| ace / per_variant / para2 / mean_tv | 0.0344 | 0.0357 | 0.0563 | 0.0231 | 0.0309 | 0.0338 | 0.0321 |
| ace / per_variant / para3 / accuracy | 0.9 | 0.86 | 0.75 | 0.75 | 0.9375 | 0.75 | 0.6875 |
| ace / per_variant / para3 / flip_rate | 0.08 | 0.02 | 0.0625 | 0.0625 | 0.1875 | 0 | 0.0625 |
| ace / per_variant / para3 / mean_tv | 0.0843 | 0.0357 | 0.0478 | 0.0507 | 0.0525 | 0.0362 | 0.0353 |
| ace / paraphrase_flip_rate | 0.0667 | 0.0267 | 0.0625 | 0.0208 | 0.1042 | 0.0208 | 0.0208 |
| ace / items_consistent_paraphrases | 0.88 | 0.96 | 0.9375 | 0.9375 | 0.8125 | 0.9375 | 0.9375 |
| banking77 / items | 50 | 50 | 16 | 16 | 16 | 16 | 16 |
| banking77 / per_variant / original / accuracy | 0.96 | 0.96 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / per_variant / minimal / accuracy | 0.96 | 0.92 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / per_variant / minimal / flip_rate | 0 | 0.04 | 0.0625 | 0 | 0 | 0 | 0 |
| banking77 / per_variant / minimal / mean_tv | 0.0222 | 0.0554 | 0.0524 | 0.0311 | 0.024 | 0.0299 | 0.0057 |
| banking77 / per_variant / para1 / accuracy | 0.96 | 0.94 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / per_variant / para1 / flip_rate | 0 | 0.02 | 0 | 0 | 0 | 0 | 0 |
| banking77 / per_variant / para1 / mean_tv | 0.0113 | 0.0228 | 0.0132 | 0.0088 | 0.0115 | 0.0132 | 0.007 |
| banking77 / per_variant / para2 / accuracy | 0.96 | 0.94 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / per_variant / para2 / flip_rate | 0 | 0.02 | 0 | 0 | 0 | 0 | 0 |
| banking77 / per_variant / para2 / mean_tv | 0.0066 | 0.0192 | 0.0083 | 0.0136 | 0.013 | 0.0063 | 0.0031 |
| banking77 / per_variant / para3 / accuracy | 0.94 | 0.96 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / per_variant / para3 / flip_rate | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / per_variant / para3 / mean_tv | 0.0088 | 0.0285 | 0.0079 | 0.0081 | 0.0062 | 0.0059 | 0.0033 |
| banking77 / paraphrase_flip_rate | 0.0067 | 0.0133 | 0 | 0 | 0 | 0 | 0 |
| banking77 / items_consistent_paraphrases | 0.98 | 0.98 | 1 | 1 | 1 | 1 | 1 |
