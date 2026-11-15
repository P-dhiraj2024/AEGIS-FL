# AEGIS-FL: Conformally Calibrated Multi-Test Update Filtering for Byzantine-Robust FL in UAV-Assisted Edge Networks

Target venue: IEEE Transactions on Network and Service Management (TNSM)

## Compile the paper
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Or upload this folder to Overleaf. Set `\showplanfalse` in `main.tex` before submission to hide the planning appendix.
Tables (`tables/*.tex`, `tables/numbers.tex`) and figures (`figures/*.pdf`) are generated, never hand-edited.

## Run the code
```
cd B_AEGIS_TNSM/code
python scripts/prepare_data.py                 # Edge-IIoTset + SensIT npz, N=500 Dirichlet partitions
python -m src.run --smoke                      # 2 rounds, 20 clients (rule 2)
python -m src.run --config configs/default.yaml --seed 0 --set data.dataset=cifar10 attack.type=min_max
python scripts/make_jobs.py lr && python scripts/jobqueue.py results/jobs_lr.txt --workers 6
python scripts/pick_lr.py                      # -> configs/lr.yaml
bash scripts/finish_all.sh                     # all phases, retries, MLflow sync, tables, figures
```
Results: `code/results/runs/<dataset>/<method>/<attack>_g<ratio>[_a<alpha>][_eps<eps>]/<seed>/`
(`config.yaml`, `history.json`, `summary.json`, `meta.json` with git commit and wall time).
Runs write plain files because 7 parallel workers share one SQLite MLflow store badly;
`scripts/sync_mlflow.py` imports all finished runs into `code/results/mlruns/mlflow.db` afterwards.

## Setup actually run
| Item | Value |
|---|---|
| Hierarchy | 500 ground clients -> 10 UAV edge servers (K-means positions, 100 m, urban A2G channel, 10 MHz each) -> cloud |
| Devices | 60% IoT sensor nodes (0.5-1 GHz, 0.1 W, 5 local steps, availability 0.90, static); 40% smartphones (1.5-2.5 GHz, 0.2 W, 10 local steps, availability 0.85, random-walk 0.5-2 m/s, re-associate every round) |
| Round | 50 clients scheduled, deadline = 95th percentile latency (stragglers dropped), batch 32, plain local SGD, server lr 1 |
| Datasets (6) | CICIoT2023 (8 cl.), Edge-IIoTset (15 cl.), SensIT (3 cl.), Fashion-MNIST, FEMNIST (natural writers), CIFAR-10; Dirichlet 0.5 for all but FEMNIST |
| Splits | client train (90%), validation (pooled 10% hold-out minus root set), official test set; root set 500 clean samples (FLTrust + AEGIS references) |
| Models | MLP 128-64 (tabular), CNN 16-32-fc128 (28x28), CNN 32-64-64-fc128 (CIFAR-10); bf16 autocast for CNN local training |
| Rounds | 300 for every run |
| LR | FedAvg no-attack grid, smallest LR within 1 pp of the best validation accuracy (`results/lr_selection.txt`): 0.3 tabular, 0.1 images |
| Attacks | none, label flip, sign flip (x4), Gaussian (sigma 1), LIE, Min-Max, Min-Sum, DBA, Neurotoxin, adaptive; gamma = 0.3 (ratio sweep 0.1-0.5) |
| Defences | FedAvg, Krum, trimmed mean, median, FLTrust, FLAME, FLDetector, FreqFed, AEGIS-FL (same rounds, steps, LR, clients) |

## Deviations / decisions to review
1. **Datasets.** UAVIDS-2025 and UAV-NIDD are not on disk. Place their CSVs in `data/raw/uavids2025/` and
   `data/raw/uav_nidd/` and run `python data/scripts/preprocess_ids.py`. They were replaced by Edge-IIoTset and
   SensIT (both already in `datasets_raw/`) so that six datasets could be run now.
2. **Projection seed (Eq. sketch).** A learned detector cannot work with a projection that is re-drawn every
   round: for a fresh Gaussian R the law of R v/||v|| is N(0, I/k) for every v. The UAVs therefore share one
   secret seed, unknown to clients (`sketch_refresh_every` = never in all runs). Eq. (sketch) is unchanged.
3. **Centring.** The VAE scores s_i - c, where c is the sketch of the cloud's update on half of the root set.
   Without centring the benign sketch distribution drifts with the global model.
4. **Nonconformity normalisation (Eq. conformal).** Raw scores drift upward between detector retrains, which
   broke calibration (47% benign rejection in a pilot). Scores are divided by the median score of the round's
   reference updates (16 non-IID root shards); Eq. (conformal) is applied to the normalised scores. Calibration
   buffer = scores of trusted clients (reputation >= 0.55, kept whether accepted or not) + reference scores;
   adaptive alpha uses err_t on the same set. The guarantee of Theorem 1 holds only under exchangeability;
   the measured benign rejection is reported per alpha (`tab_conformal`).
5. **Psi surrogate (Eq. gan).** Psi = layer-mean of 1 - cos(c + delta, c) (first-order loss change, JL
   preserves inner products); the detector term uses min(S, 2 tau) to keep the game bounded.
6. **Secure aggregation + DP.** Baselines run at the cloud on plaintext updates (they need individual updates).
   Main grid: AEGIS-FL with pairwise-mask secure sum, no DP noise (epsilon = inf) for a like-for-like accuracy
   comparison; DP is evaluated separately (epsilon in {1, 4, 8, 16}, `tab_privacy`). The default config's
   epsilon = 8 was not used for the main grid.
7. **Seeds.** Main grid (6 datasets x 9 defences x 10 conditions): seeds 0-4 for every cell (rule 3 met).
   Calibration and strong-adaptive studies: seeds 0-2; malicious-ratio, ablation and privacy studies: seeds 0-1
   (deviation from rule 3, stated in the paper's protocol paragraph).
8. **Baselines not run.** OptiGradTrust, MT-FedHealRL and RoSEFed: no code or algorithm description in the repo.
   FLDetector and FreqFed are re-implementations (docstrings cite the papers); FLDetector is adapted to partial
   participation (prediction from the client's own previous round).
9. **Krum / trimmed mean** are given the true number of attackers in the round (favourable to them).
10. **Clipping.** Accepted AEGIS updates are clipped to 1.5x the median reference-update norm before the secure
    sum (needed for DP; also bounds scaled attacks).
11. **Norm coordinates added to the detector input (changes Eq. sketch's feature vector).** With Eq. (sketch)
    alone the detector was nearly blind in pilots (CICIoT2023, seed 0, 300 rounds): sign-flip TPR 0.11, label-flip
    TPR = FPR; robustness came only from clipping. Appending the L per-layer log-norm ratios
    (`defense.norm_features: true`) gave sign-flip TPR 0.89, Gaussian 1.0, label flip 0.35. A collusion feature
    (max cosine to another sketch) and per-coordinate scaling did not help LIE / Min-Max and are off.
    The spec-exact detector is run as ablation `aegis-no_norm`. **Please confirm this change for the paper.**
12. **Reported accuracy / ASR** = mean of the evaluations at rounds 260, 270, ..., 300 (single-round values
    oscillate by up to 20 pp under label flipping on FEMNIST); `test_acc_at_best_val` (validation-selected round)
    is stored in every summary as well.
13. **Non-finite updates** are discarded by the server for every method before aggregation.
14. **Superseded runs** (detector without norm features, adaptive attacker without feature mirroring, runs that
    diverged before item 13) are kept in `code/results/runs_superseded/v1/` (list in MOVED.txt).
15. **Final detector (v3, used for all reported AEGIS-FL runs; you asked to "fix and improve").** Four
    nonconformity scores -- VAE (Eq. score), per-layer log-norm excess (vs references with the same local-step
    budget), misalignment with g_c, collusion (max sketch cosine to another update) -- each with a conformal
    buffer of *clean reference* scores; reject if min p-value <= alpha_t / 4 (Bonferroni). This replaces item 4's
    calibration from trusted clients (poisonable: accepted attackers inflated the threshold). ACI feedback uses
    trusted clients + references. Eq. (conformal) in main.tex is rewritten accordingly (per-score thresholds at
    alpha_t / J); Theorem 1 now states the union bound. Pilot evidence (seed 0): LIE / Min-Max / Min-Sum TPR
    0.88-0.99 (was ~0.02), F-MNIST sign flip 0.83 (was 0.17-0.32), benign rejection 5.3-6.8 % at alpha = 5 %.
    Ablations remove each piece (`no_norm`, `no_normtest`, `no_align`, `no_collusion`, `no_vae`, `no_gan`,
    `no_conformal`, `no_reputation`). Earlier detector versions: `code/results/runs_superseded/v1`, `v3`.
16. **Adaptive attacker (final)** mirrors all four tests with a surrogate (own projection seed), starts from
    mu + N(0, diag(sigma^2)) bases and takes the largest perturbation that keeps every surrogate score inside the
    benign 90th percentile (`attack.adaptive.target_quantile: 0.9`).
17. **Title changed** (user decision, option b) from "Autoencoder-Guided, GAN-Hardened and Conformally Calibrated ..." to
    "Conformally Calibrated Multi-Test Update Filtering ...", because the ablation shows the VAE performs on par with a
    plain distance score and the GAN red team has no measurable effect; both are kept as components and reported.
18. **Final deliverables:** `main.pdf` (13 pages, IEEEtran journal, Times clone under XeTeX) and `supplementary.pdf`
    (per-dataset tables). Compile on Overleaf with pdfLaTeX for real Times and re-check the page count.
# AEGIS-FL
