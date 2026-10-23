# Replication study report (`test`)

Generated from local runs (Decision-1, gpt-oss) and Azure ML job summaries (tag `replication`). Design: [docs/replication.md](../../replication.md).

## Jobs

| job | status | exit |
|---|---|---|
| ace__r2_qwen35_9b_letter__short | Completed | 0 |
| ace__r2_qwen35_9b_letter__verbose | Completed | 0 |
| banking77__r2_qwen35_9b_letter__short | Completed | 0 |
| replication__probes-qwen3.5-9b-r7r8 | Completed | 0 |
| replication__probes-qwen3.5-9b | Completed | 0 |
| replication__probes-qwen9b-sorted | Completed | 0 |
| replication__r14-qwen9b | Completed | 0 |
| replication__r15-typed | Completed | 0 |
| replication__r16-qwen9b | Completed | 0 |
| replication__r16b-qwen9b | Completed | 0 |
| replication__r17-routing | Completed | 0 |
| replication__r19-candidate | Completed | 0 |
| replication__r19-frozen | Completed | 0 |
| replication__r2-labels-ace-numbers | Completed | 0 |
| replication__r2-labels-banking-codes | Completed | 0 |
| replication__r2-labels-banking-numbers | Completed | 0 |
| replication__r2-prompt-variants | Completed | 0 |
| replication__r20-scale | Completed | 0 |
| replication__r3-brier_shuffle | Completed | 0 |
| replication__r3-ce_fixed | Completed | 0 |
| replication__r3-ce_shuffle | Completed | 0 |
| replication__r3-frozen | Completed | 0 |
| replication__r5-banking-prompts | Completed | 0 |
| replication__r6-paraphrase-qwen9b | Completed | 0 |
| replication__r9-qwen9b | Completed | 0 |
| replication__r9b-kl | Completed | 0 |
| replication__r9b-nokl | Completed | 0 |

## Accuracy

| method | ACE short | ACE verbose | BANKING77 |
|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.6191 | 0.7409 | 0.8698 |
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.7409 | 0.75 |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.7673 | 0.6968 |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.6782 | 0.7438 |
| Qwen3.5-9B frozen, letter codes (one pass) | – | – | 0.7055 |
| Qwen3.5-9B frozen, numbers (one pass) | 0.6455 | 0.7209 | 0.6812 |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.3882 | 0.5071 |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.5436 | 0.6445 |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.4909 | 0.6412 |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.5109 | 0.6503 |
| R9: 9B + LoRA decision training | 0.5727 | 0.6064 | 0.7487 |
| R9b: R9 + none option, KL 1.0 | 0.7 | 0.75 | 0.7331 |
| R9b: R9 + none option, no KL | 0.7382 | 0.7673 | 0.7416 |
| R15: R9b + score/noul training | 0.6518 | 0.7127 | 0.7497 |
| R17: R9b-kl + Bitext banking/support routing | 0.7009 | 0.78 | 0.7526 |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.6682 | 0.7227 | 0.7364 |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.5855 | 0.5609 | 0.7438 |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.5891 | 0.7906 |

## ACE short: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.6191 | 0.5269 | 1.154 | 0.2218 | 2.751 | 0.8362 | 0.0391 | 0.667 | 777.4 | 809.3 |
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.5857 | 0.8977 | 0.1875 | 1.613 | 0.7844 | 0.0812 | 0.7034 | 129.3 | 845.1 |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.7768 | 0.5811 | 0.0556 | 0.6363 | 0.591 | 0.0552 | 0.8432 | 122.7 | 818.1 |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.5442 | 1.24 | 0.2588 | 2.512 | 0.8874 | 0.0811 | 0.6261 | 130.5 | 845.3 |
| Qwen3.5-9B frozen, numbers (one pass) | 0.6455 | 0.5968 | 0.9376 | 0.1887 | 1.615 | 0.8185 | 0.0863 | 0.6716 | 298.7 | 845.1 |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.208 | 2.31 | 0.5567 | 7.702 | 1.053 | 0.1874 | 0.4489 | 77.08 | 845.1 |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.4278 | 1.032 | 0.1638 | 1.803 | 0.9544 | 0.055 | 0.583 | 85.78 | 845.1 |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.418 | 1.111 | 0.1929 | 2.204 | 0.9729 | 0.0555 | 0.5682 | 87.46 | 845.1 |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.4318 | 1.004 | 0.1416 | 1.576 | 0.9564 | 0.0564 | 0.5864 | 86.54 | 845.1 |
| R9: 9B + LoRA decision training | 0.5727 | 0.4636 | 1.023 | 0.2142 | 1.881 | 0.8839 | 0.0982 | 0.6318 | 155.5 | 845.1 |
| R9b: R9 + none option, KL 1.0 | 0.7 | 0.6898 | 0.6965 | 0.0969 | 1.113 | 0.6847 | 0.0709 | 0.7568 | 156.6 | 845.1 |
| R9b: R9 + none option, no KL | 0.7382 | 0.7324 | 0.6195 | 0.0706 | 1.067 | 0.615 | 0.0613 | 0.8023 | 158.8 | 845.1 |
| R15: R9b + score/noul training | 0.6518 | 0.6254 | 0.7874 | 0.1293 | 1.268 | 0.7565 | 0.0964 | 0.6966 | 155.8 | 845.1 |
| R17: R9b-kl + Bitext banking/support routing | 0.7009 | 0.6692 | 0.695 | 0.0648 | 1.12 | 0.6878 | 0.0483 | 0.7648 | 158 | 845.1 |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.6682 | 0.6386 | 0.7658 | 0.1209 | 1.285 | 0.7347 | 0.0624 | 0.7284 | 155 | 845.1 |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.5855 | 0.5304 | 0.9926 | 0.1943 | 1.698 | 0.8904 | 0.0792 | 0.6227 | 168.2 | 845.1 |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.373 | 2.505 | 0.4523 | 6.51 | 0.9636 | 0.1399 | 0.5 | 106.3 | 870.4 |

## ACE verbose: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.7409 | 0.7298 | 0.7265 | 0.1059 | 1.647 | 0.6443 | 0.0653 | 0.775 | 636 | 859.3 |
| Qwen3.5-9B frozen, letter readout | 0.7409 | 0.727 | 0.6827 | 0.1058 | 1.333 | 0.6374 | 0.0464 | 0.7989 | 136.9 | 910.1 |
| Qwen3.5-9B frozen, markdown prompt | 0.7673 | 0.7614 | 0.6039 | 0.0614 | 0.8215 | 0.6278 | 0.0859 | 0.8136 | 132.3 | 883.1 |
| Qwen3.5-9B frozen, JSON prompt | 0.6782 | 0.663 | 0.9141 | 0.1639 | 1.777 | 0.7675 | 0.0629 | 0.7227 | 137.1 | 907.3 |
| Qwen3.5-9B frozen, numbers (one pass) | 0.7209 | 0.7093 | 0.7136 | 0.1017 | 1.301 | 0.6728 | 0.0503 | 0.7761 | 324.1 | 910.1 |
| Qwen3.5-2B frozen, letter readout | 0.3882 | 0.234 | 1.943 | 0.477 | 6.218 | 1.062 | 0.1893 | 0.4443 | 80.13 | 910.1 |
| 2B + LoRA, CE, shuffled options | 0.5436 | 0.4848 | 0.947 | 0.1721 | 1.411 | 0.9055 | 0.1088 | 0.5659 | 90.26 | 910.1 |
| 2B + LoRA, CE, fixed option order | 0.4909 | 0.4054 | 1.074 | 0.2602 | 1.866 | 0.952 | 0.1212 | 0.5125 | 90.02 | 910.1 |
| 2B + LoRA, Brier, shuffled options | 0.5109 | 0.4252 | 0.9857 | 0.1829 | 1.445 | 0.9447 | 0.112 | 0.5523 | 89.17 | 910.1 |
| R9: 9B + LoRA decision training | 0.6064 | 0.5579 | 0.8704 | 0.1812 | 1.472 | 0.8019 | 0.1024 | 0.6591 | 164 | 910.1 |
| R9b: R9 + none option, KL 1.0 | 0.75 | 0.7504 | 0.591 | 0.071 | 0.9859 | 0.5922 | 0.0749 | 0.8102 | 166.2 | 910.1 |
| R9b: R9 + none option, no KL | 0.7673 | 0.769 | 0.5514 | 0.0625 | 1.024 | 0.5497 | 0.0576 | 0.8284 | 167.1 | 910.1 |
| R15: R9b + score/noul training | 0.7127 | 0.7096 | 0.6712 | 0.0981 | 1.058 | 0.6651 | 0.0878 | 0.7489 | 165.1 | 910.1 |
| R17: R9b-kl + Bitext banking/support routing | 0.78 | 0.7797 | 0.5412 | 0.0243 | 0.8643 | 0.5438 | 0.042 | 0.8352 | 166.8 | 910.1 |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.7227 | 0.7164 | 0.6607 | 0.0961 | 1.147 | 0.6469 | 0.06 | 0.7727 | 163.8 | 910.1 |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.5609 | 0.5137 | 0.9784 | 0.2258 | 1.587 | 0.8863 | 0.1287 | 0.6068 | 178.4 | 910.1 |
| gpt-oss-20b letter readout (C6, context) | 0.5891 | 0.5304 | 1.875 | 0.3114 | 4.278 | 0.8875 | 0.0732 | 0.6375 | 114.5 | 933.4 |

## BANKING77: accuracy, calibration, cost

`T`, `nll_cal`, `ece_cal`: temperature fit on the calibration split. `p50_ms`: Decision-1 over the internet, local models GPU time.

| method | accuracy | macro_f1 | nll | ece | T | nll_cal | ece_cal | acc@80 | p50_ms | in_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision-1 (post-trained Qwen3.5-9B) | 0.8698 | 0.8667 | 0.4919 | 0.0155 | 1.018 | 0.492 | 0.0142 | 0.9436 | 743.6 | 817.3 |
| Qwen3.5-9B frozen, letter readout | 0.75 | 0.7443 | 1.234 | 0.0965 | 1.406 | 1.11 | 0.0209 | 0.8433 | 285.4 | 906.3 |
| Qwen3.5-9B frozen, markdown prompt | 0.6968 | 0.6859 | 1.46 | 0.1004 | 1.388 | 1.338 | 0.0321 | 0.7955 | 272.8 | 798.2 |
| Qwen3.5-9B frozen, JSON prompt | 0.7438 | 0.7392 | 1.405 | 0.1267 | 1.68 | 1.139 | 0.0368 | 0.8251 | 305.3 | 1182 |
| Qwen3.5-9B frozen, letter codes (one pass) | 0.7055 | 0.6965 | 1.633 | 0.1753 | 1.656 | 1.313 | 0.0341 | 0.7971 | 129.2 | 618.6 |
| Qwen3.5-9B frozen, numbers (one pass) | 0.6812 | 0.6707 | 1.573 | 0.1259 | 1.34 | 1.45 | 0.0239 | 0.765 | 7336 | 685.6 |
| Qwen3.5-2B frozen, letter readout | 0.5071 | 0.5026 | 2.698 | 0.1355 | 1.502 | 2.43 | 0.0765 | 0.5726 | 230.7 | 907 |
| 2B + LoRA, CE, shuffled options | 0.6445 | 0.6304 | 1.58 | 0.1276 | 1.396 | 1.442 | 0.0288 | 0.7167 | 285.2 | 906.2 |
| 2B + LoRA, CE, fixed option order | 0.6412 | 0.6272 | 1.675 | 0.1395 | 1.462 | 1.492 | 0.021 | 0.7163 | 283.7 | 906.2 |
| 2B + LoRA, Brier, shuffled options | 0.6503 | 0.6413 | 1.436 | 0.0624 | 1.173 | 1.41 | 0.0227 | 0.7297 | 297.2 | 906.3 |
| R9: 9B + LoRA decision training | 0.7487 | 0.7421 | 1.121 | 0.0836 | 1.269 | 1.061 | 0.017 | 0.8328 | 153.1 | 618.6 |
| R9b: R9 + none option, KL 1.0 | 0.7331 | 0.7255 | 1.147 | 0.069 | 1.256 | 1.094 | 0.0225 | 0.8198 | 155.1 | 618.6 |
| R9b: R9 + none option, no KL | 0.7416 | 0.7346 | 1.302 | 0.1264 | 1.496 | 1.125 | 0.0222 | 0.8182 | 156 | 618.6 |
| R15: R9b + score/noul training | 0.7497 | 0.7405 | 1.118 | 0.0679 | 1.297 | 1.054 | 0.0183 | 0.8393 | 154.2 | 618.6 |
| R17: R9b-kl + Bitext banking/support routing | 0.7526 | 0.7469 | 1.125 | 0.0733 | 1.325 | 1.05 | 0.0218 | 0.8304 | 153.6 | 618.6 |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.7364 | 0.7274 | 1.147 | 0.0707 | 1.244 | 1.097 | 0.0236 | 0.8247 | 152.9 | 618.6 |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.7438 | 0.7359 | 1.075 | 0.0755 | 1.15 | 1.054 | 0.0243 | 0.8267 | 165.8 | 618.6 |
| gpt-oss-20b letter readout (C6, context) | 0.7906 | 0.7843 | 1.556 | 0.1576 | 1.931 | 0.99 | 0.0376 | 0.8754 | 115.8 | 749.5 |

## ACE short: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.6473 | 0.8291 | 0.5016 | 0.6571 | 0.2945 | 0.9105 | 0.6904 | 0.8367 | 0.8691 | 0.0429 | {'Compliant': 0.334, 'Non-Compliant': 0.609, 'Not-Applicable': 0.057} |
| Qwen3.5-9B frozen, markdown prompt | 0.7745 | 0.6282 | 0.3816 | 0.3988 | 0.1255 | 0.8696 | 0.195 | 0.2441 | 0.3076 | 0.1426 | {'Compliant': 0.28, 'Non-Compliant': 0.439, 'Not-Applicable': 0.281} |
| Qwen3.5-9B frozen, JSON prompt | 0.5964 | 0.7973 | 0.5318 | 0.567 | 0.31 | 0.8768 | 0.6558 | 0.8013 | 0.8544 | 0.0524 | {'Compliant': 0.224, 'Non-Compliant': 0.714, 'Not-Applicable': 0.063} |
| Qwen3.5-9B frozen, numbers (one pass) | 0.6455 | 0.8291 | 0.5094 | 0.6516 | 0.2955 | 0.9108 | 0.6905 | 0.8298 | 0.857 | 0.0475 | {'Compliant': 0.283, 'Non-Compliant': 0.647, 'Not-Applicable': 0.07} |
| Qwen3.5-2B frozen, letter readout | 0.3782 | 0.6527 | 0.6313 | 0.0582 | 0.3691 | 0.7759 | 0.5615 | 0.7107 | 0.8146 | 0.1285 | {'Compliant': 0.019, 'Non-Compliant': 0.979, 'Not-Applicable': 0.002} |
| 2B + LoRA, CE, shuffled options | 0.5418 | 0.7891 | 0.554 | 0.5271 | 0.3364 | 0.8676 | 0.6689 | 0.8309 | 0.8468 | 0.0617 | {'Compliant': 0.264, 'Non-Compliant': 0.73, 'Not-Applicable': 0.006} |
| 2B + LoRA, CE, fixed option order | 0.5291 | 0.7891 | 0.5615 | 0.519 | 0.3409 | 0.88 | 0.6664 | 0.8353 | 0.8558 | 0.0573 | {'Compliant': 0.239, 'Non-Compliant': 0.755, 'Not-Applicable': 0.006} |
| 2B + LoRA, Brier, shuffled options | 0.5482 | 0.8064 | 0.5535 | 0.5664 | 0.3409 | 0.8747 | 0.6985 | 0.8407 | 0.8634 | 0.0666 | {'Compliant': 0.269, 'Non-Compliant': 0.726, 'Not-Applicable': 0.005} |
| R9: 9B + LoRA decision training | 0.5727 | 0.8518 | 0.5427 | 0.676 | 0.3455 | 0.9105 | 0.7606 | 0.9099 | 0.9288 | 0.029 | {'Compliant': 0.29, 'Non-Compliant': 0.698, 'Not-Applicable': 0.012} |
| R9b: R9 + none option, KL 1.0 | 0.7 | 0.7791 | 0.4769 | 0.5777 | 0.2382 | 0.9313 | 0.5568 | 0.7108 | 0.7494 | 0.0628 | {'Compliant': 0.236, 'Non-Compliant': 0.617, 'Not-Applicable': 0.146} |
| R9b: R9 + none option, no KL | 0.7382 | 0.7709 | 0.455 | 0.5796 | 0.2118 | 0.9528 | 0.525 | 0.6774 | 0.7295 | 0.0684 | {'Compliant': 0.27, 'Non-Compliant': 0.565, 'Not-Applicable': 0.165} |
| R15: R9b + score/noul training | 0.6518 | 0.8109 | 0.503 | 0.6196 | 0.2809 | 0.9223 | 0.641 | 0.8113 | 0.8569 | 0.0452 | {'Compliant': 0.233, 'Non-Compliant': 0.662, 'Not-Applicable': 0.105} |
| R17: R9b-kl + Bitext banking/support routing | 0.7009 | 0.8182 | 0.4742 | 0.6542 | 0.2564 | 0.9433 | 0.6406 | 0.8021 | 0.8269 | 0.0493 | {'Compliant': 0.335, 'Non-Compliant': 0.564, 'Not-Applicable': 0.102} |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.6682 | 0.8145 | 0.4938 | 0.6336 | 0.2736 | 0.9269 | 0.6439 | 0.7996 | 0.8279 | 0.0479 | {'Compliant': 0.274, 'Non-Compliant': 0.626, 'Not-Applicable': 0.1} |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.5855 | 0.7855 | 0.5338 | 0.5398 | 0.31 | 0.8739 | 0.6358 | 0.795 | 0.8315 | 0.0545 | {'Compliant': 0.226, 'Non-Compliant': 0.715, 'Not-Applicable': 0.058} |
| gpt-oss-20b letter readout (C6, context) | 0.4709 | 0.7327 | 0.5878 | 0.3517 | 0.3482 | 0.846 | 0.605 | 0.7577 | 0.8209 | 0.1003 | {'Compliant': 0.098, 'Non-Compliant': 0.868, 'Not-Applicable': 0.034} |

## ACE verbose: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.7409 | 0.8118 | 0.4122 | 0.6799 | 0.1727 | 0.9105 | 0.5501 | 0.6995 | 0.7537 | 0.0517 | {'Compliant': 0.326, 'Non-Compliant': 0.532, 'Not-Applicable': 0.142} |
| Qwen3.5-9B frozen, markdown prompt | 0.7673 | 0.6918 | 0.3534 | 0.5234 | 0.1036 | 0.7719 | 0.2341 | 0.3099 | 0.3347 | 0.1259 | {'Compliant': 0.214, 'Non-Compliant': 0.422, 'Not-Applicable': 0.365} |
| Qwen3.5-9B frozen, JSON prompt | 0.6782 | 0.7873 | 0.4437 | 0.6176 | 0.1927 | 0.9104 | 0.5342 | 0.7184 | 0.7969 | 0.0543 | {'Compliant': 0.225, 'Non-Compliant': 0.643, 'Not-Applicable': 0.132} |
| Qwen3.5-9B frozen, numbers (one pass) | 0.7209 | 0.8045 | 0.4206 | 0.6626 | 0.1782 | 0.9235 | 0.5387 | 0.7172 | 0.775 | 0.0495 | {'Compliant': 0.29, 'Non-Compliant': 0.565, 'Not-Applicable': 0.145} |
| Qwen3.5-2B frozen, letter readout | 0.3882 | 0.5718 | 0.5421 | 0.0649 | 0.2409 | 0.8377 | 0.3859 | 0.5846 | 0.6768 | 0.1565 | {'Compliant': 0.0, 'Non-Compliant': 0.96, 'Not-Applicable': 0.04} |
| 2B + LoRA, CE, shuffled options | 0.5436 | 0.7136 | 0.4864 | 0.4425 | 0.2264 | 0.8956 | 0.4955 | 0.6962 | 0.7731 | 0.0826 | {'Compliant': 0.174, 'Non-Compliant': 0.765, 'Not-Applicable': 0.061} |
| 2B + LoRA, CE, fixed option order | 0.4909 | 0.6727 | 0.5111 | 0.3306 | 0.2355 | 0.8726 | 0.4728 | 0.7046 | 0.788 | 0.0829 | {'Compliant': 0.119, 'Non-Compliant': 0.845, 'Not-Applicable': 0.036} |
| 2B + LoRA, Brier, shuffled options | 0.5109 | 0.6945 | 0.5014 | 0.3874 | 0.2355 | 0.8803 | 0.4965 | 0.7133 | 0.7891 | 0.0861 | {'Compliant': 0.163, 'Non-Compliant': 0.806, 'Not-Applicable': 0.031} |
| R9: 9B + LoRA decision training | 0.6064 | 0.7845 | 0.466 | 0.5966 | 0.2273 | 0.924 | 0.5853 | 0.7974 | 0.8682 | 0.0443 | {'Compliant': 0.236, 'Non-Compliant': 0.695, 'Not-Applicable': 0.069} |
| R9b: R9 + none option, KL 1.0 | 0.75 | 0.8 | 0.4162 | 0.6574 | 0.1573 | 0.9653 | 0.4876 | 0.6648 | 0.6848 | 0.0561 | {'Compliant': 0.219, 'Non-Compliant': 0.576, 'Not-Applicable': 0.205} |
| R9b: R9 + none option, no KL | 0.7673 | 0.8036 | 0.4092 | 0.6676 | 0.1518 | 0.9461 | 0.4943 | 0.6779 | 0.7103 | 0.0532 | {'Compliant': 0.235, 'Non-Compliant': 0.554, 'Not-Applicable': 0.211} |
| R15: R9b + score/noul training | 0.7127 | 0.8009 | 0.4306 | 0.6503 | 0.18 | 0.9293 | 0.5325 | 0.739 | 0.7974 | 0.046 | {'Compliant': 0.189, 'Non-Compliant': 0.622, 'Not-Applicable': 0.189} |
| R17: R9b-kl + Bitext banking/support routing | 0.78 | 0.8164 | 0.4069 | 0.6904 | 0.1527 | 0.9345 | 0.5274 | 0.6942 | 0.7118 | 0.052 | {'Compliant': 0.265, 'Non-Compliant': 0.538, 'Not-Applicable': 0.196} |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.7227 | 0.8191 | 0.4254 | 0.6852 | 0.1836 | 0.9356 | 0.57 | 0.7324 | 0.7622 | 0.0477 | {'Compliant': 0.247, 'Non-Compliant': 0.59, 'Not-Applicable': 0.163} |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.5609 | 0.7373 | 0.483 | 0.4919 | 0.2291 | 0.9008 | 0.5304 | 0.7333 | 0.8116 | 0.0659 | {'Compliant': 0.17, 'Non-Compliant': 0.758, 'Not-Applicable': 0.072} |
| gpt-oss-20b letter readout (C6, context) | 0.5891 | 0.6764 | 0.4459 | 0.4159 | 0.1791 | 0.934 | 0.3369 | 0.5198 | 0.6877 | 0.1277 | {'Compliant': 0.045, 'Non-Compliant': 0.706, 'Not-Applicable': 0.249} |

## BANKING77: agreement with Decision-1, example by example

`kappa`: agreement beyond chance; `same_answer_when_both_wrong`: shared mistakes; `p_gold_*`: correlation of the probability given to the right answer.

| method | acc_run | top1_agreement | chance_agreement | kappa | both_wrong | same_answer_when_both_wrong | error_correlation_phi | p_gold_pearson | p_gold_spearman | mean_js_divergence | prediction_mix_run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B frozen, letter readout | 0.75 | 0.8127 | 0.0135 | 0.8101 | 0.114 | 0.6923 | 0.5587 | 0.7436 | 0.7776 | – | – |
| Qwen3.5-9B frozen, markdown prompt | 0.6968 | 0.7607 | 0.0135 | 0.7574 | 0.1185 | 0.6384 | 0.5109 | 0.6922 | 0.7633 | – | – |
| Qwen3.5-9B frozen, JSON prompt | 0.7438 | 0.8114 | 0.0136 | 0.8088 | 0.1149 | 0.7203 | 0.5554 | 0.735 | 0.7786 | – | – |
| Qwen3.5-9B frozen, letter codes (one pass) | 0.7055 | 0.7779 | 0.0137 | 0.7748 | 0.1185 | 0.7096 | 0.5226 | 0.7041 | 0.7792 | – | – |
| Qwen3.5-9B frozen, numbers (one pass) | 0.6812 | 0.7532 | 0.0138 | 0.7498 | 0.1185 | 0.7068 | 0.491 | 0.6839 | 0.7474 | – | – |
| Qwen3.5-2B frozen, letter readout | 0.5071 | 0.5412 | 0.0136 | 0.5349 | 0.1175 | 0.3978 | 0.3172 | 0.46 | 0.5754 | – | – |
| 2B + LoRA, CE, shuffled options | 0.6445 | 0.7071 | 0.0136 | 0.7031 | 0.1182 | 0.6319 | 0.4463 | 0.62 | 0.6762 | – | – |
| 2B + LoRA, CE, fixed option order | 0.6412 | 0.7003 | 0.0136 | 0.6962 | 0.1169 | 0.6194 | 0.4348 | 0.6151 | 0.676 | – | – |
| 2B + LoRA, Brier, shuffled options | 0.6503 | 0.7185 | 0.0136 | 0.7146 | 0.1198 | 0.6558 | 0.4629 | 0.631 | 0.695 | – | – |
| R9: 9B + LoRA decision training | 0.7487 | 0.8253 | 0.0136 | 0.8229 | 0.1169 | 0.7694 | 0.5766 | 0.7735 | 0.7956 | – | – |
| R9b: R9 + none option, KL 1.0 | 0.7331 | 0.8094 | 0.0136 | 0.8068 | 0.1182 | 0.7473 | 0.5605 | 0.7604 | 0.788 | – | – |
| R9b: R9 + none option, no KL | 0.7416 | 0.8101 | 0.0135 | 0.8075 | 0.114 | 0.7436 | 0.5452 | 0.7284 | 0.7439 | – | – |
| R15: R9b + score/noul training | 0.7497 | 0.8247 | 0.0136 | 0.8223 | 0.1149 | 0.7853 | 0.5649 | 0.7654 | 0.8076 | – | – |
| R17: R9b-kl + Bitext banking/support routing | 0.7526 | 0.8273 | 0.0136 | 0.8249 | 0.1182 | 0.7335 | 0.5921 | 0.7806 | 0.8195 | – | – |
| R19: candidate replica (R9b-kl recipe, sorted options) | 0.7364 | 0.8117 | 0.0136 | 0.8091 | 0.1179 | 0.7438 | 0.5634 | 0.7644 | 0.8003 | – | – |
| R20: R17 recipe at 5x steps, LoRA r64 | 0.7438 | 0.8058 | 0.0135 | 0.8032 | 0.1143 | 0.6818 | 0.551 | 0.7526 | 0.8089 | – | – |
| gpt-oss-20b letter readout (C6, context) | 0.7906 | 0.8494 | 0.0134 | 0.8473 | 0.1097 | 0.7219 | 0.6023 | 0.7276 | 0.7528 | 0.0871 | – |

## Same model, different runs (e.g. prompt templates)

Runs of one job compared example by example: how much the decisions move when only the prompt changes.

| dataset | pair | n | acc A | acc B | agreement | kappa | mean JS |
|---|---|---|---|---|---|---|---|
| ace | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_json__short | 1100 | 0.6473 | 0.5964 | 0.8545 | 0.7014 | 0.0269 |
| ace | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_markdown__short | 1100 | 0.6473 | 0.7745 | 0.6736 | 0.4762 | 0.1016 |
| ace | r2_qwen35_9b_letter__short vs r2_qwen35_9b_numbers__short | 1100 | 0.6473 | 0.6455 | 0.9327 | 0.8674 | 0.0073 |
| ace | r2_qwen35_9b_letter__verbose vs r2_qwen35_9b_letter_json__verbose | 1100 | 0.7409 | 0.6782 | 0.8491 | 0.7333 | 0.0269 |
| ace | r2_qwen35_9b_letter__verbose vs r2_qwen35_9b_letter_markdown__verbose | 1100 | 0.7409 | 0.7673 | 0.6909 | 0.5276 | 0.1142 |
| ace | r2_qwen35_9b_letter__verbose vs r2_qwen35_9b_numbers__verbose | 1100 | 0.7409 | 0.7209 | 0.93 | 0.8802 | 0.0092 |
| ace | r2_qwen35_9b_letter_json__short vs r2_qwen35_9b_letter_markdown__short | 1100 | 0.5964 | 0.7745 | 0.6509 | 0.4243 | 0.1336 |
| ace | r2_qwen35_9b_letter_json__verbose vs r2_qwen35_9b_letter_markdown__verbose | 1100 | 0.6782 | 0.7673 | 0.6827 | 0.4985 | 0.1211 |
| banking77 | r2_qwen35_9b_codes__short vs r2_qwen35_9b_letter__short | 3080 | 0.7055 | 0.75 | 0.8149 | 0.8122 | 0.098 |
| banking77 | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_json__short | 3080 | 0.75 | 0.7438 | 0.8711 | 0.8692 | 0.055 |
| banking77 | r2_qwen35_9b_letter__short vs r2_qwen35_9b_letter_markdown__short | 3080 | 0.75 | 0.6968 | 0.7581 | 0.7546 | 0.1202 |
| banking77 | r2_qwen35_9b_letter__short vs r2_qwen35_9b_numbers__short | 3080 | 0.75 | 0.6812 | 0.7935 | 0.7904 | 0.1172 |
| banking77 | r2_qwen35_9b_letter_json__short vs r2_qwen35_9b_letter_markdown__short | 3080 | 0.7438 | 0.6968 | 0.7643 | 0.7608 | 0.1292 |

## Training arms (held-out corpus rows, options shuffled)

| arm | loss | shuffle | steps | train_s | acc | nll | ece | mean_conf | pred_first | gold_first |
|---|---|---|---|---|---|---|---|---|---|---|
| brier_shuffle | brier | True | 1500 | 5507 | 0.885 | 0.2969 | 0.0356 | 0.9165 | 0.3075 | 0.3025 |
| ce_fixed | ce | False | 1500 | 5492 | 0.88 | 0.2958 | 0.0494 | 0.9248 | 0.3125 | 0.3025 |
| ce_shuffle | ce | True | 1500 | 5466 | 0.89 | 0.2861 | 0.0405 | 0.9263 | 0.3175 | 0.3025 |
| r15_typed | ce_brier | True | 600 | 4435 | 0.9 | 0.3132 | 0.0402 | 0.8625 | 0.2524 | 0.267 |
| r17_routing | ce_brier | True | 600 | 4508 | 0.9033 | 0.2922 | 0.0222 | 0.9178 | 0.2133 | 0.2167 |
| r19_candidate | ce_brier | True | 600 | 4285 | 0.8867 | 0.323 | 0.0498 | 0.9106 | 0.2133 | 0.23 |
| r20_scale | ce_brier | True | 3000 | 2.118e+04 | 0.93 | 0.2346 | 0.026 | 0.9493 | 0.2167 | 0.2167 |
| r9_qwen9b | ce_brier | True | 600 | 4699 | 0.9067 | 0.28 | 0.0324 | 0.9151 | 0.2233 | 0.23 |
| r9b_kl | ce_brier | True | 600 | 4438 | 0.8767 | 0.3178 | 0.0418 | 0.9109 | 0.2133 | 0.23 |
| r9b_nokl | ce_brier | True | 600 | 3639 | 0.9067 | 0.2588 | 0.0403 | 0.9315 | 0.22 | 0.23 |

`brier_shuffle` held-out accuracy by step: 250: 0.8675, 500: 0.885, 750: 0.885, 1000: 0.885, 1250: 0.88, 1500: 0.885
`ce_fixed` held-out accuracy by step: 250: 0.86, 500: 0.89, 750: 0.89, 1000: 0.8825, 1250: 0.8825, 1500: 0.88
`ce_shuffle` held-out accuracy by step: 250: 0.855, 500: 0.9075, 750: 0.8925, 1000: 0.8875, 1250: 0.8875, 1500: 0.89
`r15_typed` held-out accuracy by step: 100: 0.88, 200: 0.8667, 300: 0.8933, 400: 0.9033, 500: 0.9033, 600: 0.9
`r17_routing` held-out accuracy by step: 100: 0.8967, 200: 0.9133, 300: 0.91, 400: 0.9067, 500: 0.9, 600: 0.9033
`r19_candidate` held-out accuracy by step: 100: 0.8767, 200: 0.8933, 300: 0.87, 400: 0.8767, 500: 0.8867, 600: 0.8867
`r20_scale` held-out accuracy by step: 500: 0.9033, 1000: 0.9067, 1500: 0.93, 2000: 0.91, 2500: 0.9333, 3000: 0.93
`r9_qwen9b` held-out accuracy by step: 100: 0.8733, 200: 0.8767, 300: 0.8867, 400: 0.8833, 500: 0.9067, 600: 0.9067
`r9b_kl` held-out accuracy by step: 100: 0.88, 200: 0.8867, 300: 0.87, 400: 0.8733, 500: 0.8767, 600: 0.8767
`r9b_nokl` held-out accuracy by step: 100: 0.8833, 200: 0.8967, 300: 0.88, 400: 0.8933, 500: 0.9033, 600: 0.9067

## Probes

Side by side for decision-1, qwen3.5-9b, qwen3.5-9b-sorted, r9-qwen9b, r9b-kl, r9b-nokl, r15-typed, r17-routing, r19-candidate, r20-scale, qwen3.5-2b, r3-ce-shuffle, r3-ce-fixed, r3-brier-shuffle: [probe-report.md](probe-report.md).

