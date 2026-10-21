# Replication study report (`test`)

Generated from local runs (Decision-1, gpt-oss) and Azure ML job summaries (tag `replication`). Design: [docs/replication.md](../../replication.md).

## Jobs

| job | status | exit |
|---|---|---|
| ace__r2_qwen35_9b_letter__short | Completed | 0 |
| ace__r2_qwen35_9b_letter__verbose | Completed | 0 |
| banking77__r2_qwen35_9b_letter__short | Completed | 0 |
| replication__probes-qwen3.5-9b | Completed | 0 |
| replication__r2-prompt-variants | Completed | 0 |
| replication__r3-brier_shuffle | Completed | 0 |
| replication__r3-ce_fixed | Completed | 0 |
| replication__r3-ce_shuffle | Completed | 0 |
| replication__r3-frozen | Completed | 0 |

## Accuracy

| method | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.6191 | 0.7409 | 0.8698 |
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.7409 | 0.75 |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.7673 | – |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.6782 | – |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.3882 | 0.5071 |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.5436 | 0.6445 |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.4909 | 0.6412 |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.5109 | 0.6503 |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.5891 | 0.7906 |

## ACE short: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.6191 | 0.5269 | 1.154 | 0.2218 | 2.751 | 0.8362 | 0.0391 | 0.667 | 777.4 | 809.3 |
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.5857 | 0.8977 | 0.1875 | 1.613 | 0.7844 | 0.0812 | 0.7034 | 129.3 | 845.1 |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.7768 | 0.5811 | 0.0556 | 0.6363 | 0.591 | 0.0552 | 0.8432 | 122.7 | 818.1 |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.5442 | 1.24 | 0.2588 | 2.512 | 0.8874 | 0.0811 | 0.6261 | 130.5 | 845.3 |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.208 | 2.31 | 0.5567 | 7.702 | 1.053 | 0.1874 | 0.4489 | 77.08 | 845.1 |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.4278 | 1.032 | 0.1638 | 1.803 | 0.9544 | 0.055 | 0.583 | 85.78 | 845.1 |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.418 | 1.111 | 0.1929 | 2.204 | 0.9729 | 0.0555 | 0.5682 | 87.46 | 845.1 |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.4318 | 1.004 | 0.1416 | 1.576 | 0.9564 | 0.0564 | 0.5864 | 86.54 | 845.1 |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.373 | 2.505 | 0.4523 | 6.51 | 0.9636 | 0.1399 | 0.5 | 106.3 | 870.4 |

## ACE verbose: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.7409 | 0.7298 | 0.7265 | 0.1059 | 1.647 | 0.6443 | 0.0653 | 0.775 | 636 | 859.3 |
| Qwen3.5-9B frozen, letter readout | 0.7409 | 0.727 | 0.6827 | 0.1058 | 1.333 | 0.6374 | 0.0464 | 0.7989 | 136.9 | 910.1 |
| Qwen3.5-9B frozen, markdown prompt | 0.7673 | 0.7614 | 0.6039 | 0.0614 | 0.8215 | 0.6278 | 0.0859 | 0.8136 | 132.3 | 883.1 |
| Qwen3.5-9B frozen, JSON prompt | 0.6782 | 0.663 | 0.9141 | 0.1639 | 1.777 | 0.7675 | 0.0629 | 0.7227 | 137.1 | 907.3 |
| Qwen3.5-2B frozen, letter readout | 0.3882 | 0.234 | 1.943 | 0.477 | 6.218 | 1.062 | 0.1893 | 0.4443 | 80.13 | 910.1 |
| 2B + LoRA, CE, shuffled options | 0.5436 | 0.4848 | 0.947 | 0.1721 | 1.411 | 0.9055 | 0.1088 | 0.5659 | 90.26 | 910.1 |
| 2B + LoRA, CE, fixed option order | 0.4909 | 0.4054 | 1.074 | 0.2602 | 1.866 | 0.952 | 0.1212 | 0.5125 | 90.02 | 910.1 |
| 2B + LoRA, Brier, shuffled options | 0.5109 | 0.4252 | 0.9857 | 0.1829 | 1.445 | 0.9447 | 0.112 | 0.5523 | 89.17 | 910.1 |
| gpt-oss-20b letter readout (C6, context) | 0.5891 | 0.5304 | 1.875 | 0.3114 | 4.278 | 0.8875 | 0.0732 | 0.6375 | 114.5 | 933.4 |

## BANKING77: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.8698 | 0.8667 | 0.4919 | 0.0155 | 1.018 | 0.492 | 0.0142 | 0.9436 | 743.6 | 817.3 |
| Qwen3.5-9B frozen, letter readout | 0.75 | 0.7443 | 1.234 | 0.0965 | 1.406 | 1.11 | 0.0209 | 0.8433 | 289.9 | 906.3 |
| Qwen3.5-2B frozen, letter readout | 0.5071 | 0.5026 | 2.698 | 0.1355 | 1.502 | 2.43 | 0.0765 | 0.5726 | 230.7 | 907 |
| 2B + LoRA, CE, shuffled options | 0.6445 | 0.6304 | 1.58 | 0.1276 | 1.396 | 1.442 | 0.0288 | 0.7167 | 285.2 | 906.2 |
| 2B + LoRA, CE, fixed option order | 0.6412 | 0.6272 | 1.675 | 0.1395 | 1.462 | 1.492 | 0.021 | 0.7163 | 283.7 | 906.2 |
| 2B + LoRA, Brier, shuffled options | 0.6503 | 0.6413 | 1.436 | 0.0624 | 1.173 | 1.41 | 0.0227 | 0.7297 | 297.2 | 906.3 |
| gpt-oss-20b letter readout (C6, context) | 0.7906 | 0.7843 | 1.556 | 0.1576 | 1.931 | 0.99 | 0.0376 | 0.8754 | 115.8 | 749.5 |

## ACE short: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.8291 | 0.5016 | 0.6571 | 0.2945 | 0.9105 | 0.6904 | 0.8367 | 0.8691 | 0.0429 | {'Compliant': 0.334, 'Non-Compliant': 0.609, 'Not-Applicable': 0.057} |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.6282 | 0.3816 | 0.3988 | 0.1255 | 0.8696 | 0.195 | 0.2441 | 0.3076 | 0.1426 | {'Compliant': 0.28, 'Non-Compliant': 0.439, 'Not-Applicable': 0.281} |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.7973 | 0.5318 | 0.567 | 0.31 | 0.8768 | 0.6558 | 0.8013 | 0.8544 | 0.0524 | {'Compliant': 0.224, 'Non-Compliant': 0.714, 'Not-Applicable': 0.063} |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.6527 | 0.6313 | 0.0582 | 0.3691 | 0.7759 | 0.5615 | 0.7107 | 0.8146 | 0.1285 | {'Compliant': 0.019, 'Non-Compliant': 0.979, 'Not-Applicable': 0.002} |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.7891 | 0.554 | 0.5271 | 0.3364 | 0.8676 | 0.6689 | 0.8309 | 0.8468 | 0.0617 | {'Compliant': 0.264, 'Non-Compliant': 0.73, 'Not-Applicable': 0.006} |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.7891 | 0.5615 | 0.519 | 0.3409 | 0.88 | 0.6664 | 0.8353 | 0.8558 | 0.0573 | {'Compliant': 0.239, 'Non-Compliant': 0.755, 'Not-Applicable': 0.006} |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.8064 | 0.5535 | 0.5664 | 0.3409 | 0.8747 | 0.6985 | 0.8407 | 0.8634 | 0.0666 | {'Compliant': 0.269, 'Non-Compliant': 0.726, 'Not-Applicable': 0.005} |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.7327 | 0.5878 | 0.3517 | 0.3482 | 0.846 | 0.605 | 0.7577 | 0.8209 | 0.1003 | {'Compliant': 0.098, 'Non-Compliant': 0.868, 'Not-Applicable': 0.034} |

## ACE verbose: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.7409 | 0.8118 | 0.4122 | 0.6799 | 0.1727 | 0.9105 | 0.5501 | 0.6995 | 0.7537 | 0.0517 | {'Compliant': 0.326, 'Non-Compliant': 0.532, 'Not-Applicable': 0.142} |
| Qwen3.5-9B frozen, markdown prompt | 0.7673 | 0.6918 | 0.3534 | 0.5234 | 0.1036 | 0.7719 | 0.2341 | 0.3099 | 0.3347 | 0.1259 | {'Compliant': 0.214, 'Non-Compliant': 0.422, 'Not-Applicable': 0.365} |
| Qwen3.5-9B frozen, JSON prompt | 0.6782 | 0.7873 | 0.4437 | 0.6176 | 0.1927 | 0.9104 | 0.5342 | 0.7184 | 0.7969 | 0.0543 | {'Compliant': 0.225, 'Non-Compliant': 0.643, 'Not-Applicable': 0.132} |
| Qwen3.5-2B frozen, letter readout | 0.3882 | 0.5718 | 0.5421 | 0.0649 | 0.2409 | 0.8377 | 0.3859 | 0.5846 | 0.6768 | 0.1565 | {'Compliant': 0.0, 'Non-Compliant': 0.96, 'Not-Applicable': 0.04} |
| 2B + LoRA, CE, shuffled options | 0.5436 | 0.7136 | 0.4864 | 0.4425 | 0.2264 | 0.8956 | 0.4955 | 0.6962 | 0.7731 | 0.0826 | {'Compliant': 0.174, 'Non-Compliant': 0.765, 'Not-Applicable': 0.061} |
| 2B + LoRA, CE, fixed option order | 0.4909 | 0.6727 | 0.5111 | 0.3306 | 0.2355 | 0.8726 | 0.4728 | 0.7046 | 0.788 | 0.0829 | {'Compliant': 0.119, 'Non-Compliant': 0.845, 'Not-Applicable': 0.036} |
| 2B + LoRA, Brier, shuffled options | 0.5109 | 0.6945 | 0.5014 | 0.3874 | 0.2355 | 0.8803 | 0.4965 | 0.7133 | 0.7891 | 0.0861 | {'Compliant': 0.163, 'Non-Compliant': 0.806, 'Not-Applicable': 0.031} |
| gpt-oss-20b letter readout (C6, context) | 0.5891 | 0.6764 | 0.4459 | 0.4159 | 0.1791 | 0.934 | 0.3369 | 0.5198 | 0.6877 | 0.1277 | {'Compliant': 0.045, 'Non-Compliant': 0.706, 'Not-Applicable': 0.249} |

## BANKING77: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.75 | 0.8127 | 0.0135 | 0.8101 | 0.114 | 0.6923 | 0.5587 | 0.7436 | 0.7776 | – | – |
| Qwen3.5-2B frozen, letter readout | 0.5071 | 0.5412 | 0.0136 | 0.5349 | 0.1175 | 0.3978 | 0.3172 | 0.46 | 0.5754 | – | – |
| 2B + LoRA, CE, shuffled options | 0.6445 | 0.7071 | 0.0136 | 0.7031 | 0.1182 | 0.6319 | 0.4463 | 0.62 | 0.6762 | – | – |
| 2B + LoRA, CE, fixed option order | 0.6412 | 0.7003 | 0.0136 | 0.6962 | 0.1169 | 0.6194 | 0.4348 | 0.6151 | 0.676 | – | – |
| 2B + LoRA, Brier, shuffled options | 0.6503 | 0.7185 | 0.0136 | 0.7146 | 0.1198 | 0.6558 | 0.4629 | 0.631 | 0.695 | – | – |
| gpt-oss-20b letter readout (C6, context) | 0.7906 | 0.8494 | 0.0134 | 0.8473 | 0.1097 | 0.7219 | 0.6023 | 0.7276 | 0.7528 | 0.0871 | – |

## Same model, different runs (e.g. prompt templates)

Runs of one job compared example by example: how much the decisions move when only the prompt changes.

| dataset | pair | n | acc A | acc B | agreement | kappa | mean JS |
|---|---|---|---|---|---|---|---|
| ace | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_json__short | 1100 | 0.6473 | 0.5964 | 0.8545 | 0.7014 | 0.0269 |
| ace | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_markdown__short | 1100 | 0.6473 | 0.7745 | 0.6736 | 0.4762 | 0.1016 |
| ace | r2_qwen35_9b_letter__verbose vs r2_qwen35_9b_letter_json__verbose | 1100 | 0.7409 | 0.6782 | 0.8491 | 0.7333 | 0.0269 |
| ace | r2_qwen35_9b_letter__verbose vs r2_qwen35_9b_letter_markdown__verbose | 1100 | 0.7409 | 0.7673 | 0.6909 | 0.5276 | 0.1142 |
| ace | r2_qwen35_9b_letter_json__short vs r2_qwen35_9b_letter_markdown__short | 1100 | 0.5964 | 0.7745 | 0.6509 | 0.4243 | 0.1336 |
| ace | r2_qwen35_9b_letter_json__verbose vs r2_qwen35_9b_letter_markdown__verbose | 1100 | 0.6782 | 0.7673 | 0.6827 | 0.4985 | 0.1211 |

## Training arms (held-out corpus rows, options shuffled)

| arm | loss | shuffle | steps | train_s | acc | nll | ece | mean_conf | pred_first | gold_first |
|---|---|---|---|---|---|---|---|---|---|---|
| brier_shuffle | brier | True | 1500 | 5507 | 0.885 | 0.2969 | 0.0356 | 0.9165 | 0.3075 | 0.3025 |
| ce_fixed | ce | False | 1500 | 5492 | 0.88 | 0.2958 | 0.0494 | 0.9248 | 0.3125 | 0.3025 |
| ce_shuffle | ce | True | 1500 | 5466 | 0.89 | 0.2861 | 0.0405 | 0.9263 | 0.3175 | 0.3025 |

`brier_shuffle` held-out accuracy by step: 250: 0.8675, 500: 0.885, 750: 0.885, 1000: 0.885, 1250: 0.88, 1500: 0.885
`ce_fixed` held-out accuracy by step: 250: 0.86, 500: 0.89, 750: 0.89, 1000: 0.8825, 1250: 0.8825, 1500: 0.88
`ce_shuffle` held-out accuracy by step: 250: 0.855, 500: 0.9075, 750: 0.8925, 1000: 0.8875, 1250: 0.8875, 1500: 0.89

## Probes

Side by side for decision-1, qwen3.5-9b, qwen3.5-2b, r3-ce-shuffle, r3-ce-fixed, r3-brier-shuffle: [probe-report.md](probe-report.md).

