# Decision-model probes

Generated 2026-10-22T10:16:20+00:00. What each probe tests and what each readout design predicts: [docs/replication.md](../../replication.md).

| target | description |
|---|---|
| decision-1 | Microsoft-Decision-1 on Microsoft Foundry (post-trained Qwen3.5-9B). |

## Forensics

| metric | decision-1 |
|---|---|
| answers | 1624 |
| logit_grid_share / 0.125 | 0.7746 |
| logit_grid_share / 0.25 | 0.1367 |
| logit_grid_share / 0.5 | 0.0326 |
| logit_grid_share / 0.0625 | 0.0308 |
| logit_grid_share / 1 | 0.024 |
| logit_grid_share / none | 0.0012 |
| logit_grid_main_step | 0.125 |
| bf16_logit_magnitude_if_main_step | [16, 32) |
| confidence_is_jev_formula | 1 |
| confidence_is_max_prob | 0 |
| max_abs_sum_error | 2.22e-16 |
| choice_is_argmax | 0.9982 |
| output_tokens / 1 | 1592 |
| output_tokens / 2 | 32 |
| latency_p50_ms | 485 |
| latency_p95_ms | 886.3 |

## order

_Does reordering the options change the decision? Is there a position bias?_ (decision-1: 176 requests, 0 errors)

| metric | decision-1 |
|---|---|
| ace / items | 16 |
| ace / accuracy | 0.8125 |
| ace / flip_rate_vs_original | 0 |
| ace / items_consistent_all_orders | 1 |
| ace / mean_tv_vs_original | 0.014 |
| ace / mean_chosen_position | 0.5 |
| ace / mean_gold_position | 0.5 |
| ace / first_vs_last_log_lift | -0.0048 |
| banking77 / items | 16 |
| banking77 / accuracy | 1 |
| banking77 / flip_rate_vs_original | 0 |
| banking77 / items_consistent_all_orders | 1 |
| banking77 / mean_tv_vs_original | 0.0035 |
| banking77 / mean_chosen_position | 0.4675 |
| banking77 / mean_gold_position | 0.4675 |
| banking77 / first_vs_last_log_lift | 0.0216 |

## letter_names

_Do option names that look like answer letters confuse the readout?_ (decision-1: 128 requests, 0 errors)

| metric | decision-1 |
|---|---|
| ace / items | 16 |
| ace / accuracy_letters_aligned | 0.8125 |
| ace / accuracy_letters_shifted | 0.75 |
| ace / accuracy_neutral | 0.8125 |
| ace / accuracy_original | 0.8125 |
| ace / agree_with_neutral_letters_aligned | 1 |
| ace / agree_with_neutral_letters_shifted | 0.9375 |
| ace / agree_with_neutral_original | 1 |
| ace / shifted_disagreements | 1 |
| ace / collision_pull | 0 |
| banking77 / items | 16 |
| banking77 / accuracy_letters_aligned | 1 |
| banking77 / accuracy_letters_shifted | 1 |
| banking77 / accuracy_neutral | 0.9375 |
| banking77 / accuracy_original | 0.9375 |
| banking77 / agree_with_neutral_letters_aligned | 0.9375 |
| banking77 / agree_with_neutral_letters_shifted | 0.9375 |
| banking77 / agree_with_neutral_original | 0.875 |
| banking77 / shifted_disagreements | 1 |
| banking77 / collision_pull | 0 |

## duplicate

_How is probability split between two identical options?_ (decision-1: 96 requests, 0 errors)

| metric | decision-1 |
|---|---|
| ace / items | 16 |
| ace / earlier_copy_share_dup_after | 0.8326 |
| ace / earlier_copy_share_dup_before | 0.1594 |
| ace / original_name_share_dup_after | 0.8326 |
| ace / original_name_share_dup_before | 0.8406 |
| ace / median_abs_log_ratio | 1.875 |
| ace / mass_conservation | 1.088 |
| banking77 / items | 16 |
| banking77 / earlier_copy_share_dup_after | 0.8535 |
| banking77 / earlier_copy_share_dup_before | 0.1458 |
| banking77 / original_name_share_dup_after | 0.8535 |
| banking77 / original_name_share_dup_before | 0.8542 |
| banking77 / median_abs_log_ratio | 1.875 |
| banking77 / mass_conservation | 1.021 |

## cardinality

_How do accuracy, tokens and latency scale with the number of options?_ (decision-1: 132 requests, 2 errors)

| metric | decision-1 |
|---|---|
| per_k / 2 / n | 16 |
| per_k / 2 / error_rate | 0 |
| per_k / 2 / accuracy | 1 |
| per_k / 2 / mean_p_gold | 0.9931 |
| per_k / 2 / input_tokens | 64.75 |
| per_k / 2 / latency_p50_ms | 453.7 |
| per_k / 4 / n | 16 |
| per_k / 4 / error_rate | 0 |
| per_k / 4 / accuracy | 0.9375 |
| per_k / 4 / mean_p_gold | 0.9381 |
| per_k / 4 / input_tokens | 83.12 |
| per_k / 4 / latency_p50_ms | 471.7 |
| per_k / 8 / n | 16 |
| per_k / 8 / error_rate | 0 |
| per_k / 8 / accuracy | 0.9375 |
| per_k / 8 / mean_p_gold | 0.9225 |
| per_k / 8 / input_tokens | 125.5 |
| per_k / 8 / latency_p50_ms | 459.9 |
| per_k / 16 / n | 16 |
| per_k / 16 / error_rate | 0 |
| per_k / 16 / accuracy | 1 |
| per_k / 16 / mean_p_gold | 0.928 |
| per_k / 16 / input_tokens | 206.4 |
| per_k / 16 / latency_p50_ms | 478.5 |
| per_k / 26 / n | 16 |
| per_k / 26 / error_rate | 0 |
| per_k / 26 / accuracy | 0.9375 |
| per_k / 26 / mean_p_gold | 0.929 |
| per_k / 26 / input_tokens | 306.9 |
| per_k / 26 / latency_p50_ms | 471.9 |
| per_k / 27 / n | 16 |
| per_k / 27 / error_rate | 0 |
| per_k / 27 / accuracy | 0.9375 |
| per_k / 27 / mean_p_gold | 0.9104 |
| per_k / 27 / input_tokens | 317.5 |
| per_k / 27 / latency_p50_ms | 452.2 |
| per_k / 40 / n | 16 |
| per_k / 40 / error_rate | 0 |
| per_k / 40 / accuracy | 0.9375 |
| per_k / 40 / mean_p_gold | 0.9201 |
| per_k / 40 / input_tokens | 447.4 |
| per_k / 40 / latency_p50_ms | 480.6 |
| per_k / 77 / n | 16 |
| per_k / 77 / error_rate | 0 |
| per_k / 77 / accuracy | 0.8125 |
| per_k / 77 / mean_p_gold | 0.8485 |
| per_k / 77 / input_tokens | 816.9 |
| per_k / 77 / latency_p50_ms | 470.8 |
| tokens_per_option | 10.04 |
| tokens_at_k0 | 45.1 |
| tokens_per_option_16_to_26 | 10.05 |
| tokens_per_option_26_to_27 | 10.62 |
| latency_ms_per_option | 0.14 |
| limits / k256 | error: HTTPError: HTTP 400: {"error":{"code":"unsupported_request_argument","message":"Model does not support request argument  |
| limits / k255 | accepted |

## question

_Does the model read the question, or only the state and options?_ (decision-1: 160 requests, 0 errors)

| metric | decision-1 |
|---|---|
| banking77 / accuracy_listed_first | 0.5 |
| banking77 / accuracy_listed_last | 0.5 |
| banking77 / accuracy_match | 1 |
| banking77 / accuracy_not_match | 1 |
| banking77 / same_as_match_listed_first | 0.5 |
| banking77 / same_as_match_listed_last | 0.5 |
| banking77 / same_as_match_not_match | 0 |
| ace / accuracy_listed_first | 1 |
| ace / accuracy_listed_last | 0.75 |

## empty_state

_What does the model answer when the state carries no information?_ (decision-1: 20 requests, 0 errors)

| metric | decision-1 |
|---|---|
| banking77 / empty / errors | 0 |
| banking77 / empty / norm_entropy | 0.9533 |
| banking77 / empty / p_first_position | 0.1612 |
| banking77 / empty / p_last_position | 0.161 |
| banking77 / empty / uniform | 0.125 |
| banking77 / empty / top_value | pending_top_up |
| banking77 / empty / top_value_p | 0.2006 |
| banking77 / na / errors | 0 |
| banking77 / na / norm_entropy | 0.9083 |
| banking77 / na / p_first_position | 0.1392 |
| banking77 / na / p_last_position | 0.1488 |
| banking77 / na / uniform | 0.125 |
| banking77 / na / top_value | beneficiary_not_allowed |
| banking77 / na / top_value_p | 0.2926 |
| banking77 / no_info / errors | 0 |
| banking77 / no_info / norm_entropy | 0.8958 |
| banking77 / no_info / p_first_position | 0.1575 |
| banking77 / no_info / p_last_position | 0.1653 |
| banking77 / no_info / uniform | 0.125 |
| banking77 / no_info / top_value | beneficiary_not_allowed |
| banking77 / no_info / top_value_p | 0.287 |
| banking77 / lorem / errors | 0 |
| banking77 / lorem / norm_entropy | 0.9081 |
| banking77 / lorem / p_first_position | 0.1533 |
| banking77 / lorem / p_last_position | 0.1533 |
| banking77 / lorem / uniform | 0.125 |
| banking77 / lorem / top_value | beneficiary_not_allowed |
| banking77 / lorem / top_value_p | 0.27 |
| ace / empty / errors | 0 |
| ace / empty / norm_entropy | 0.8623 |
| ace / empty / p_first_position | 0.3163 |
| ace / empty / p_last_position | 0.3432 |
| ace / empty / uniform | 0.3333 |
| ace / empty / top_value | Not-Applicable |
| ace / empty / top_value_p | 0.5998 |
| ace / na / errors | 0 |
| ace / na / norm_entropy | 0.0428 |
| ace / na / p_first_position | 0.3332 |
| ace / na / p_last_position | 0.3334 |
| ace / na / uniform | 0.3333 |
| ace / na / top_value | Not-Applicable |
| ace / na / top_value_p | 0.9928 |
| ace / no_info / errors | 0 |
| ace / no_info / norm_entropy | 0.0156 |
| ace / no_info / p_first_position | 0.3334 |
| ace / no_info / p_last_position | 0.333 |
| ace / no_info / uniform | 0.3333 |
| ace / no_info / top_value | Not-Applicable |
| ace / no_info / top_value_p | 0.9978 |
| ace / lorem / errors | 0 |
| ace / lorem / norm_entropy | 0.0331 |
| ace / lorem / p_first_position | 0.3337 |
| ace / lorem / p_last_position | 0.333 |
| ace / lorem / uniform | 0.3333 |
| ace / lorem / top_value | Not-Applicable |
| ace / lorem / top_value_p | 0.9946 |

## multi_question

_Is the state encoded once and shared across questions in one request?_ (decision-1: 64 requests, 0 errors)

| metric | decision-1 |
|---|---|
| items | 16 |
| shared_tokens_mean | 718.1 |
| shared_fraction_of_q1_request | 0.8227 |
| marginal_tokens_same_question_again | 116 |
| max_abs_dp_q1_alone_vs_combined | 0.0476 |
| max_abs_dp_q2_alone_vs_combined | 0.04913 |
| max_abs_dp_duplicate_question | 0 |
| latency_ratio_combined_vs_q1 | 1.112 |

## determinism

_Are repeated identical requests answered identically?_ (decision-1: 60 requests, 0 errors)

| metric | decision-1 |
|---|---|
| banking77 / items | 6 |
| banking77 / identical_all_repeats | 0 |
| banking77 / max_abs_dp | 0.04344 |
| banking77 / choice_flips | 0 |
| ace / items | 6 |
| ace / identical_all_repeats | 0 |
| ace / max_abs_dp | 0.02553 |
| ace / choice_flips | 0 |

## tokens

_Which tokenizer is billed, and how large is the hidden prompt?_ (decision-1: 18 requests, 0 errors)

| metric | decision-1 |
|---|---|
| billed_input_tokens / hello | 40 |
| billed_input_tokens / japanese | 45 |
| billed_input_tokens / apple_x02 | 41 |
| billed_input_tokens / digits | 49 |
| billed_input_tokens / sql | 50 |
| billed_input_tokens / spaces | 42 |
| billed_input_tokens / apple_x32 | 71 |
| billed_input_tokens / hello_world | 43 |
| billed_input_tokens / code | 51 |
| billed_input_tokens / apple_x16 | 55 |
| billed_input_tokens / x | 40 |
| billed_input_tokens / russian | 46 |
| billed_input_tokens / apple_x04 | 43 |
| billed_input_tokens / pi | 55 |
| billed_input_tokens / apple_x01 | 40 |
| billed_input_tokens / url | 48 |
| billed_input_tokens / emoji | 46 |
| billed_input_tokens / apple_x08 | 47 |
| tokens_per_apple | 1 |
| tokenizer_fits / Qwen/Qwen3.5-9B / hidden_prompt_tokens | 39 |
| tokenizer_fits / Qwen/Qwen3.5-9B / exact_fit_fraction | 0.667 |
| tokenizer_fits / openai/gpt-oss-20b / hidden_prompt_tokens | 38 |
| tokenizer_fits / openai/gpt-oss-20b / exact_fit_fraction | 0.333 |
| best_tokenizer | Qwen/Qwen3.5-9B |

## duplicate_names

_What makes a model prefer one of two identical options?_ (decision-1: 160 requests, 0 errors)

| metric | decision-1 |
|---|---|
| ace / items | 16 |
| ace / first_share_neutral | 0.4505 |
| ace / original_share_suffix_copy | 0.8499 |
| ace / original_share_unrelated | 0.7784 |
| ace / pair_mass_neutral | 0.7656 |
| ace / pair_mass_suffix_copy | 0.7901 |
| ace / pair_mass_swap_after | 0.7926 |
| ace / pair_mass_swap_before | 0.7965 |
| ace / pair_mass_unrelated | 0.7774 |
| ace / plain_name_share_swap_after | 0.8413 |
| ace / plain_name_share_swap_before | 0.8412 |
| banking77 / items | 16 |
| banking77 / first_share_neutral | 0.4249 |
| banking77 / original_share_suffix_copy | 0.911 |
| banking77 / original_share_unrelated | 0.994 |
| banking77 / pair_mass_neutral | 0.838 |
| banking77 / pair_mass_suffix_copy | 0.9481 |
| banking77 / pair_mass_swap_after | 0.9465 |
| banking77 / pair_mass_swap_before | 0.9475 |
| banking77 / pair_mass_unrelated | 0.9433 |
| banking77 / plain_name_share_swap_after | 0.8525 |
| banking77 / plain_name_share_swap_before | 0.8567 |

## long_options

_Are long option descriptions truncated?_ (decision-1: 112 requests, 0 errors)

| metric | decision-1 |
|---|---|
| per_variant / end_010 / n | 16 |
| per_variant / end_010 / accuracy | 0.875 |
| per_variant / end_010 / mean_p_gold | 0.9232 |
| per_variant / end_010 / input_tokens | 536.9 |
| per_variant / end_040 / n | 16 |
| per_variant / end_040 / accuracy | 0.875 |
| per_variant / end_040 / mean_p_gold | 0.9231 |
| per_variant / end_040 / input_tokens | 1857 |
| per_variant / end_150 / n | 16 |
| per_variant / end_150 / accuracy | 0.9375 |
| per_variant / end_150 / mean_p_gold | 0.9121 |
| per_variant / end_150 / input_tokens | 6697 |
| per_variant / start_000 / n | 16 |
| per_variant / start_000 / accuracy | 1 |
| per_variant / start_000 / mean_p_gold | 0.9375 |
| per_variant / start_000 / input_tokens | 96.88 |
| per_variant / start_010 / n | 16 |
| per_variant / start_010 / accuracy | 0.9375 |
| per_variant / start_010 / mean_p_gold | 0.9175 |
| per_variant / start_010 / input_tokens | 536.9 |
| per_variant / start_040 / n | 16 |
| per_variant / start_040 / accuracy | 0.9375 |
| per_variant / start_040 / mean_p_gold | 0.9194 |
| per_variant / start_040 / input_tokens | 1857 |
| per_variant / start_150 / n | 16 |
| per_variant / start_150 / accuracy | 0.9375 |
| per_variant / start_150 / mean_p_gold | 0.9226 |
| per_variant / start_150 / input_tokens | 6697 |
| tokens_per_filler_repeat | 44 |

## paraphrase

_Does rewording the question change the decision?_ (decision-1: 500 requests, 0 errors)

| metric | decision-1 |
|---|---|
| ace / items | 50 |
| ace / per_variant / original / accuracy | 0.84 |
| ace / per_variant / minimal / accuracy | 0.82 |
| ace / per_variant / minimal / flip_rate | 0.08 |
| ace / per_variant / minimal / mean_tv | 0.0357 |
| ace / per_variant / para1 / accuracy | 0.82 |
| ace / per_variant / para1 / flip_rate | 0.06 |
| ace / per_variant / para1 / mean_tv | 0.0329 |
| ace / per_variant / para2 / accuracy | 0.84 |
| ace / per_variant / para2 / flip_rate | 0.06 |
| ace / per_variant / para2 / mean_tv | 0.0344 |
| ace / per_variant / para3 / accuracy | 0.9 |
| ace / per_variant / para3 / flip_rate | 0.08 |
| ace / per_variant / para3 / mean_tv | 0.0843 |
| ace / paraphrase_flip_rate | 0.0667 |
| ace / items_consistent_paraphrases | 0.88 |
| banking77 / items | 50 |
| banking77 / per_variant / original / accuracy | 0.96 |
| banking77 / per_variant / minimal / accuracy | 0.96 |
| banking77 / per_variant / minimal / flip_rate | 0 |
| banking77 / per_variant / minimal / mean_tv | 0.0222 |
| banking77 / per_variant / para1 / accuracy | 0.96 |
| banking77 / per_variant / para1 / flip_rate | 0 |
| banking77 / per_variant / para1 / mean_tv | 0.0113 |
| banking77 / per_variant / para2 / accuracy | 0.96 |
| banking77 / per_variant / para2 / flip_rate | 0 |
| banking77 / per_variant / para2 / mean_tv | 0.0066 |
| banking77 / per_variant / para3 / accuracy | 0.94 |
| banking77 / per_variant / para3 / flip_rate | 0.02 |
| banking77 / per_variant / para3 / mean_tv | 0.0088 |
| banking77 / paraphrase_flip_rate | 0.0067 |
| banking77 / items_consistent_paraphrases | 0.98 |
