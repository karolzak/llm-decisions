# Decision-model probes

Generated 2026-10-21T05:29:58+00:00. What each probe tests and what each readout design predicts: [docs/replication.md](../../replication.md).

| target | description |
|---|---|
| decision-1 | Microsoft-Decision-1 on Microsoft Foundry (post-trained Qwen3.5-9B). |
| qwen3.5-9b | Frozen Qwen3.5-9B (Decision-1's base model), option-letter readout, no training. |
| qwen3.5-2b | Frozen Qwen3.5-2B (base of the Part 3 training arms), option-letter readout. |
| r3-ce-shuffle | Qwen3.5-2B + LoRA arm `ce_shuffle` (cross-entropy, options shuffled every example). |
| r3-ce-fixed | Qwen3.5-2B + LoRA arm `ce_fixed` (cross-entropy, options in canonical label order). |
| r3-brier-shuffle | Qwen3.5-2B + LoRA arm `brier_shuffle` (Brier score, options shuffled every example). |

## Forensics

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| answers | 852 | 854 | 712 | 712 | 712 | 712 |
| logit_grid_share / 0.125 | 0.7031 | 0.0094 | 0.0042 | 0.0056 | 0.0056 | 0.0042 |
| logit_grid_share / 0.25 | 0.1444 | 0.0035 | 0.0056 | 0.0056 | 0.0028 | 0.0112 |
| logit_grid_share / 0.0625 | 0.0587 | – | 0.007 | 0.0028 | 0.0042 | 0.0084 |
| logit_grid_share / 0.5 | 0.0516 | 0.0035 | 0.0014 | 0.0028 | 0.007 | 0.0056 |
| logit_grid_share / 1 | 0.0399 | 0.0059 | 0.007 | 0.014 | 0.0112 | 0.0042 |
| logit_grid_share / none | 0.0023 | 0.9707 | 0.9649 | 0.9593 | 0.9635 | 0.9579 |
| logit_grid_main_step | 0.125 | – | – | – | – | – |
| bf16_logit_magnitude_if_main_step | [16, 32) | – | – | – | – | – |
| confidence_is_jev_formula | 1 | 1 | 1 | 1 | 1 | 1 |
| confidence_is_max_prob | 0 | 0 | 0 | 0 | 0 | 0 |
| max_abs_sum_error | 1.11e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| choice_is_argmax | 0.9977 | 1 | 1 | 1 | 1 | 1 |
| output_tokens / 1 | 820 | – | – | – | – | – |
| output_tokens / 2 | 32 | – | – | – | – | – |
| latency_p50_ms | 471 | 77.4 | 50.1 | 65.3 | 67.2 | 69.4 |
| latency_p95_ms | 810.3 | 337.4 | 163 | 198.5 | 199.8 | 197.8 |
| logit_grid_share / 0.03125 | – | 0.0035 | 0.0042 | 0.0042 | – | 0.0028 |
| logit_grid_share / 0.015625 | – | 0.0035 | 0.0056 | 0.0056 | 0.0056 | 0.0056 |
| output_tokens / 0 | – | 854 | 712 | 712 | 712 | 712 |

## order

_Does reordering the options change the decision? Is there a position bias?_ (decision-1: 176 requests, 0 errors, qwen3.5-9b: 176 requests, 0 errors, qwen3.5-2b: 176 requests, 0 errors, r3-ce-shuffle: 176 requests, 0 errors, r3-ce-fixed: 176 requests, 0 errors, r3-brier-shuffle: 176 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / accuracy | 0.8125 | 0.7917 | 0.5104 | 0.6458 | 0.6771 | 0.625 |
| ace / flip_rate_vs_original | 0 | 0.075 | 0.3125 | 0.1125 | 0.0625 | 0.075 |
| ace / items_consistent_all_orders | 1 | 0.875 | 0.0625 | 0.75 | 0.875 | 0.875 |
| ace / mean_tv_vs_original | 0.014 | 0.0874 | 0.2629 | 0.0806 | 0.0697 | 0.0538 |
| ace / mean_chosen_position | 0.5 | 0.5052 | 0.651 | 0.5104 | 0.5208 | 0.4896 |
| ace / mean_gold_position | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| ace / first_vs_last_log_lift | -0.0048 | -0.0247 | -1.183 | -0.1178 | 0.0282 | -0.1927 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / accuracy | 1 | 0.9625 | 0.9625 | 0.9625 | 0.925 | 0.9375 |
| banking77 / flip_rate_vs_original | 0 | 0.0312 | 0.0312 | 0.0312 | 0.0625 | 0 |
| banking77 / items_consistent_all_orders | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 1 |
| banking77 / mean_tv_vs_original | 0.0035 | 0.0418 | 0.1361 | 0.0303 | 0.0392 | 0.0245 |
| banking77 / mean_chosen_position | 0.4675 | 0.47 | 0.465 | 0.46 | 0.475 | 0.465 |
| banking77 / mean_gold_position | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 | 0.4675 |
| banking77 / first_vs_last_log_lift | 0.0216 | -0.5527 | 2.863 | 0.5928 | 1.172 | 1.183 |

## letter_names

_Do option names that look like answer letters confuse the readout?_ (decision-1: 128 requests, 0 errors, qwen3.5-9b: 128 requests, 0 errors, qwen3.5-2b: 128 requests, 0 errors, r3-ce-shuffle: 128 requests, 0 errors, r3-ce-fixed: 128 requests, 0 errors, r3-brier-shuffle: 128 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / accuracy_letters_aligned | 0.8125 | 0.75 | 0.5 | 0.6875 | 0.625 | 0.6875 |
| ace / accuracy_letters_shifted | 0.75 | 0.75 | 0.5 | 0.6875 | 0.625 | 0.625 |
| ace / accuracy_neutral | 0.8125 | 0.75 | 0.4375 | 0.6875 | 0.625 | 0.6875 |
| ace / accuracy_original | 0.8125 | 0.8125 | 0.4375 | 0.6875 | 0.625 | 0.6875 |
| ace / agree_with_neutral_letters_aligned | 1 | 1 | 0.9375 | 1 | 1 | 1 |
| ace / agree_with_neutral_letters_shifted | 0.9375 | 1 | 0.9375 | 0.9375 | 1 | 0.9375 |
| ace / agree_with_neutral_original | 1 | 0.9375 | 1 | 1 | 1 | 1 |
| ace / shifted_disagreements | 1 | 0 | 1 | 1 | 0 | 1 |
| ace / collision_pull | 0 | – | 0 | 0 | – | 0 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / accuracy_letters_aligned | 1 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / accuracy_letters_shifted | 1 | 1 | 0.9375 | 0.9375 | 0.6875 | 0.9375 |
| banking77 / accuracy_neutral | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / accuracy_original | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| banking77 / agree_with_neutral_letters_aligned | 0.9375 | 1 | 1 | 1 | 1 | 1 |
| banking77 / agree_with_neutral_letters_shifted | 0.9375 | 1 | 1 | 1 | 0.625 | 1 |
| banking77 / agree_with_neutral_original | 0.875 | 1 | 1 | 1 | 1 | 1 |
| banking77 / shifted_disagreements | 1 | 0 | 0 | 0 | 6 | 0 |
| banking77 / collision_pull | 0 | – | – | – | 0 | – |

## duplicate

_How is probability split between two identical options?_ (decision-1: 96 requests, 0 errors, qwen3.5-9b: 96 requests, 0 errors, qwen3.5-2b: 96 requests, 0 errors, r3-ce-shuffle: 96 requests, 0 errors, r3-ce-fixed: 96 requests, 0 errors, r3-brier-shuffle: 96 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| ace / items | 16 | 16 | 16 | 16 | 16 | 16 |
| ace / earlier_copy_share_dup_after | 0.8326 | 0.9674 | 0.6887 | 0.8149 | 0.7931 | 0.7468 |
| ace / earlier_copy_share_dup_before | 0.1594 | 0.5943 | 0.5646 | 0.7448 | 0.7112 | 0.661 |
| ace / original_name_share_dup_after | 0.8326 | 0.9674 | 0.6887 | 0.8149 | 0.7931 | 0.7468 |
| ace / original_name_share_dup_before | 0.8406 | 0.4057 | 0.4354 | 0.2552 | 0.2888 | 0.339 |
| ace / median_abs_log_ratio | 1.875 | 2.56 | 0.9537 | 1.587 | 1.194 | 0.9856 |
| ace / mass_conservation | 1.088 | 0.8553 | 1.654 | 1.202 | 1.215 | 1.227 |
| banking77 / items | 16 | 16 | 16 | 16 | 16 | 16 |
| banking77 / earlier_copy_share_dup_after | 0.8535 | 0.8803 | 0.8714 | 0.9168 | 0.9352 | 0.8448 |
| banking77 / earlier_copy_share_dup_before | 0.1458 | 0.2969 | 0.563 | 0.5037 | 0.4875 | 0.4385 |
| banking77 / original_name_share_dup_after | 0.8535 | 0.8803 | 0.8714 | 0.9168 | 0.9352 | 0.8448 |
| banking77 / original_name_share_dup_before | 0.8542 | 0.7031 | 0.437 | 0.4963 | 0.5125 | 0.5615 |
| banking77 / median_abs_log_ratio | 1.875 | 1.572 | 1.912 | 1.208 | 1.476 | 1.284 |
| banking77 / mass_conservation | 1.021 | 0.9978 | 1.055 | 1.009 | 0.9747 | 0.9808 |

## cardinality

_How do accuracy, tokens and latency scale with the number of options?_ (decision-1: 132 requests, 2 errors, qwen3.5-9b: 132 requests, 0 errors, qwen3.5-2b: 132 requests, 0 errors, r3-ce-shuffle: 132 requests, 0 errors, r3-ce-fixed: 132 requests, 0 errors, r3-brier-shuffle: 132 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| per_k / 2 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 2 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 2 / accuracy | 1 | 1 | 1 | 1 | 1 | 1 |
| per_k / 2 / mean_p_gold | 0.9931 | 0.9855 | 0.9656 | 0.9981 | 0.9971 | 0.9927 |
| per_k / 2 / input_tokens | 64.75 | 102.8 | 102.8 | 102.8 | 102.8 | 102.8 |
| per_k / 2 / latency_p50_ms | 453.7 | 64 | 45.1 | 60.8 | 61.2 | 60.5 |
| per_k / 4 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 4 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 4 / accuracy | 0.9375 | 1 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 4 / mean_p_gold | 0.9381 | 0.9731 | 0.9295 | 0.9416 | 0.9395 | 0.9438 |
| per_k / 4 / input_tokens | 83.12 | 115.4 | 115.4 | 115.4 | 115.4 | 115.4 |
| per_k / 4 / latency_p50_ms | 471.7 | 65.1 | 46.9 | 61.5 | 61.4 | 60.9 |
| per_k / 8 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 8 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 8 / accuracy | 0.9375 | 0.9375 | 0.8125 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 8 / mean_p_gold | 0.9225 | 0.9288 | 0.722 | 0.9365 | 0.9065 | 0.9166 |
| per_k / 8 / input_tokens | 125.5 | 143.8 | 143.8 | 143.8 | 143.8 | 143.8 |
| per_k / 8 / latency_p50_ms | 459.9 | 68 | 48.7 | 64 | 65.4 | 63.8 |
| per_k / 16 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 16 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 16 / accuracy | 1 | 0.9375 | 0.875 | 0.9375 | 0.9375 | 0.875 |
| per_k / 16 / mean_p_gold | 0.928 | 0.9133 | 0.7767 | 0.9001 | 0.9032 | 0.8781 |
| per_k / 16 / input_tokens | 206.4 | 199.1 | 199.1 | 199.1 | 199.1 | 199.1 |
| per_k / 16 / latency_p50_ms | 478.5 | 72.8 | 51.8 | 66.8 | 66.8 | 66.7 |
| per_k / 26 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 26 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 26 / accuracy | 0.9375 | 0.9375 | 0.625 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 26 / mean_p_gold | 0.929 | 0.9213 | 0.5578 | 0.908 | 0.8953 | 0.8896 |
| per_k / 26 / input_tokens | 306.9 | 267.8 | 267.8 | 267.8 | 267.8 | 267.8 |
| per_k / 26 / latency_p50_ms | 471.9 | 77.6 | 55.8 | 71.4 | 71.5 | 71.1 |
| per_k / 27 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 27 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 27 / accuracy | 0.9375 | 0.9375 | 0.875 | 0.9375 | 0.9375 | 0.9375 |
| per_k / 27 / mean_p_gold | 0.9104 | 0.9176 | 0.7677 | 0.9036 | 0.8933 | 0.879 |
| per_k / 27 / input_tokens | 317.5 | 466.1 | 466.4 | 466.2 | 466.4 | 466.5 |
| per_k / 27 / latency_p50_ms | 452.2 | 203.6 | 142.8 | 194 | 195 | 189.9 |
| per_k / 40 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 40 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 40 / accuracy | 0.9375 | 0.9375 | 0.8125 | 0.875 | 0.875 | 0.9375 |
| per_k / 40 / mean_p_gold | 0.9201 | 0.9181 | 0.6992 | 0.8388 | 0.8458 | 0.8343 |
| per_k / 40 / input_tokens | 447.4 | 555.1 | 556.2 | 555.2 | 555.8 | 555.8 |
| per_k / 40 / latency_p50_ms | 480.6 | 212.7 | 151.9 | 199 | 200.2 | 196.9 |
| per_k / 77 / n | 16 | 16 | 16 | 16 | 16 | 16 |
| per_k / 77 / error_rate | 0 | 0 | 0 | 0 | 0 | 0 |
| per_k / 77 / accuracy | 0.8125 | 0.9375 | 0.8125 | 0.875 | 0.875 | 0.9375 |
| per_k / 77 / mean_p_gold | 0.8485 | 0.8693 | 0.6773 | 0.8214 | 0.8226 | 0.7992 |
| per_k / 77 / input_tokens | 816.9 | 903.9 | 904.6 | 903.4 | 903.9 | 903.7 |
| per_k / 77 / latency_p50_ms | 470.8 | 296.7 | 211.6 | 272.8 | 278.9 | 274.2 |
| tokens_per_option | 10.04 | 11.11 | 11.13 | 11.11 | 11.12 | 11.11 |
| tokens_at_k0 | 45.1 | 66.4 | 66.3 | 66.5 | 66.4 | 66.5 |
| tokens_per_option_16_to_26 | 10.05 | 6.87 | 6.87 | 6.87 | 6.87 | 6.87 |
| tokens_per_option_26_to_27 | 10.62 | 198.2 | 198.6 | 198.4 | 198.6 | 198.7 |
| latency_ms_per_option | 0.14 | 3.38 | 2.41 | 3.1 | 3.17 | 3.11 |
| limits / k256 | error: HTTPError: HTTP 400: {"error":{"code":"unsupported_request_argument","message":"Model does not support request argument  | accepted | accepted | accepted | accepted | accepted |
| limits / k255 | accepted | accepted | accepted | accepted | accepted | accepted |

## question

_Does the model read the question, or only the state and options?_ (decision-1: 160 requests, 0 errors, qwen3.5-9b: 160 requests, 0 errors, qwen3.5-2b: 160 requests, 0 errors, r3-ce-shuffle: 160 requests, 0 errors, r3-ce-fixed: 160 requests, 0 errors, r3-brier-shuffle: 160 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| banking77 / accuracy_listed_first | 0.5 | 0.875 | 0.5625 | 0.5312 | 0.5312 | 0.5312 |
| banking77 / accuracy_listed_last | 0.5 | 1 | 1 | 0.5312 | 0.5 | 0.4688 |
| banking77 / accuracy_match | 1 | 0.9688 | 1 | 1 | 1 | 1 |
| banking77 / accuracy_not_match | 1 | 0.9688 | 0.5 | 0.7812 | 0.625 | 0.8438 |
| banking77 / same_as_match_listed_first | 0.5 | 0.6562 | 0.9375 | 0.9688 | 0.9688 | 0.9688 |
| banking77 / same_as_match_listed_last | 0.5 | 0.4688 | 0.5 | 0.9688 | 0.9375 | 0.9688 |
| banking77 / same_as_match_not_match | 0 | 0 | 0.5 | 0.2188 | 0.375 | 0.1562 |
| ace / accuracy_listed_first | 1 | 0.8125 | 0 | 0.1875 | 0.25 | 0.1875 |
| ace / accuracy_listed_last | 0.75 | 1 | 0.5625 | 0.3125 | 0.1875 | 0.5 |

## empty_state

_What does the model answer when the state carries no information?_ (decision-1: 20 requests, 0 errors, qwen3.5-9b: 20 requests, 0 errors, qwen3.5-2b: 20 requests, 0 errors, r3-ce-shuffle: 20 requests, 0 errors, r3-ce-fixed: 20 requests, 0 errors, r3-brier-shuffle: 20 requests, 0 errors)

| metric | decision-1 | qwen3.5-9b | qwen3.5-2b | r3-ce-shuffle | r3-ce-fixed | r3-brier-shuffle |
|---|---|---|---|---|---|---|
| banking77 / empty / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / empty / norm_entropy | 0.9533 | 0.7487 | 0.6734 | 0.8245 | 0.8507 | 0.8753 |
| banking77 / empty / p_first_position | 0.1612 | 0.2942 | 0.3986 | 0.2166 | 0.286 | 0.2166 |
| banking77 / empty / p_last_position | 0.161 | 0.341 | 0.0271 | 0.1096 | 0.094 | 0.097 |
| banking77 / empty / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / empty / top_value | pending_top_up | pending_top_up | transfer_timing | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge |
| banking77 / empty / top_value_p | 0.2006 | 0.4108 | 0.2172 | 0.2956 | 0.2505 | 0.244 |
| banking77 / na / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / na / norm_entropy | 0.9083 | 0.7762 | 0.6821 | 0.8454 | 0.7424 | 0.907 |
| banking77 / na / p_first_position | 0.1392 | 0.2648 | 0.4332 | 0.2902 | 0.4858 | 0.2118 |
| banking77 / na / p_last_position | 0.1488 | 0.3307 | 0.0426 | 0.0709 | 0.0379 | 0.0945 |
| banking77 / na / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / na / top_value | beneficiary_not_allowed | pending_top_up | transfer_timing | top_up_by_bank_transfer_charge | pending_top_up | transfer_fee_charged |
| banking77 / na / top_value_p | 0.2926 | 0.3999 | 0.2434 | 0.2176 | 0.2997 | 0.2261 |
| banking77 / no_info / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / no_info / norm_entropy | 0.8958 | 0.7396 | 0.6716 | 0.8805 | 0.8441 | 0.9104 |
| banking77 / no_info / p_first_position | 0.1575 | 0.2966 | 0.423 | 0.2357 | 0.3303 | 0.2186 |
| banking77 / no_info / p_last_position | 0.1653 | 0.3452 | 0.0317 | 0.0776 | 0.0536 | 0.0879 |
| banking77 / no_info / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / no_info / top_value | beneficiary_not_allowed | pending_top_up | pending_top_up | top_up_by_bank_transfer_charge | pending_top_up | transfer_fee_charged |
| banking77 / no_info / top_value_p | 0.287 | 0.4574 | 0.23 | 0.2366 | 0.2415 | 0.2214 |
| banking77 / lorem / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| banking77 / lorem / norm_entropy | 0.9081 | 0.7637 | 0.5907 | 0.8703 | 0.8669 | 0.916 |
| banking77 / lorem / p_first_position | 0.1533 | 0.2694 | 0.549 | 0.2029 | 0.2901 | 0.1927 |
| banking77 / lorem / p_last_position | 0.1533 | 0.3475 | 0.0178 | 0.0563 | 0.0399 | 0.057 |
| banking77 / lorem / uniform | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |
| banking77 / lorem / top_value | beneficiary_not_allowed | pending_top_up | pending_top_up | top_up_by_bank_transfer_charge | top_up_by_bank_transfer_charge | transfer_fee_charged |
| banking77 / lorem / top_value_p | 0.27 | 0.4174 | 0.2845 | 0.256 | 0.2078 | 0.2366 |
| ace / empty / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / empty / norm_entropy | 0.8623 | 0.2084 | 0.8739 | 0.6301 | 0.5326 | 0.8358 |
| ace / empty / p_first_position | 0.3163 | 0.3556 | 0.2474 | 0.3342 | 0.3106 | 0.3499 |
| ace / empty / p_last_position | 0.3432 | 0.3365 | 0.4154 | 0.3466 | 0.3255 | 0.3409 |
| ace / empty / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / empty / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / empty / top_value_p | 0.5998 | 0.9477 | 0.5444 | 0.7688 | 0.8166 | 0.6251 |
| ace / na / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / na / norm_entropy | 0.0428 | 0.2118 | 0.9112 | 0.3496 | 0.2739 | 0.5678 |
| ace / na / p_first_position | 0.3332 | 0.3437 | 0.3338 | 0.3211 | 0.324 | 0.3315 |
| ace / na / p_last_position | 0.3334 | 0.3408 | 0.3629 | 0.3444 | 0.3291 | 0.345 |
| ace / na / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / na / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / na / top_value_p | 0.9928 | 0.9483 | 0.5326 | 0.9012 | 0.9289 | 0.8061 |
| ace / no_info / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / no_info / norm_entropy | 0.0156 | 0.1514 | 0.5943 | 0.2884 | 0.2411 | 0.5252 |
| ace / no_info / p_first_position | 0.3334 | 0.3423 | 0.3412 | 0.3238 | 0.3238 | 0.3349 |
| ace / no_info / p_last_position | 0.333 | 0.334 | 0.3548 | 0.3418 | 0.3259 | 0.3413 |
| ace / no_info / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / no_info / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / no_info / top_value_p | 0.9978 | 0.9665 | 0.7868 | 0.9235 | 0.9386 | 0.8271 |
| ace / lorem / errors | 0 | 0 | 0 | 0 | 0 | 0 |
| ace / lorem / norm_entropy | 0.0331 | 0.0795 | 0.6469 | 0.2794 | 0.2786 | 0.3496 |
| ace / lorem / p_first_position | 0.3337 | 0.338 | 0.3354 | 0.3196 | 0.3207 | 0.325 |
| ace / lorem / p_last_position | 0.333 | 0.3337 | 0.3514 | 0.3393 | 0.3281 | 0.3421 |
| ace / lorem / uniform | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| ace / lorem / top_value | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable | Not-Applicable |
| ace / lorem / top_value_p | 0.9946 | 0.9848 | 0.7575 | 0.9253 | 0.9255 | 0.9023 |

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
