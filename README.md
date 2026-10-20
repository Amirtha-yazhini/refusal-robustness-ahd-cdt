# Spread it out vs keep it in one place: testing two ways to protect a model's refusal behaviour

Does spreading a model's refusal behaviour across many attention heads (Attention Head Dropout, AHD)
or concentrating it in a small circuit (Circuit-Restricted Weight Arithmetic, C-ΔΘ) make it harder
to remove? Both defences are applied to Llama-3.2-1B-Instruct, alongside a plain safety fine-tune
as a control, and attacked with directional ablation, refusal-head masking and harmful fine-tuning.

## Main findings

Fraction of 100 held-out AdvBench completions judged unsafe by Llama Guard 3 1B (lower is safer):

| Model | No attack | Dir. ablation | Top 32 heads masked | FT 5 steps | FT 10 steps |
|---|---|---|---|---|---|
| Original | 0.10 | 0.55 | 0.39 | 0.96 | 0.96 |
| Plain fine-tune | 0.01 | 0.05 | 0.01 | 0.85 | 0.97 |
| AHD | 0.01 | 0.62 | 0.00 | 0.88 | 0.94 |
| C-ΔΘ | 0.01 | 0.39 | 0.11 | 0.00 | 0.75 |

- **Head masking** mainly hurts the original model. The plain fine-tune resists it as well as AHD,
  so this robustness seems to come from safety fine-tuning, not from head dropout.
- **AHD is the most vulnerable to directional ablation** (62% unsafe, against 5% for its control),
  and it has the largest capability cost (benign loss 2.01 → 3.22, against 2.47 for the plain fine-tune).
- **C-ΔΘ delays harmful fine-tuning the most.** It is 0% unsafe after 5 steps, and in the 3-seed
  sweep it takes 7–10 steps to reach 50% unsafe, against 3–5 steps for the others. Every model is
  89–97% unsafe within 15 steps, and the delay is confounded (see Known issues).
- **Mechanism:** AHD spreads refusal over the most heads (effective count 172, against 93–143), but its
  heads are the least aligned with the refusal direction (0.053, against 0.094–0.140). This rules out the
  "backup copies" explanation. One untested possibility is that AHD routes refusal through the MLPs.

## Repository layout

```
notebooks/
  1_localised_vs_distributed_safety.ipynb  builds the four models (original, plain fine-tune, AHD, C-ΔΘ)
                                            and runs the three attacks; Llama Guard + phrase judges
  2_defence_step_sweep.ipynb               retrains the models, saves them, and runs the harmful
                                            fine-tuning attack at 3/5/7/10/15/20 steps x 3 seeds
  3_refusal_mechanism_checks.ipynb         loads the saved models from notebook 2 and runs the
                                            head-alignment, multi-direction/random-direction and
                                            activation-addition checks
results/
  run1/        notebook 1: results_phrase_judge.csv, results_llama_guard.csv
  sweep/       notebook 2: sweep_results.csv, sweep_summary.csv, sweep_break_step.csv, sweep_chart.png
  mechanism/   notebook 3: check1_heads.csv, check2_erase.csv, check3_add_r.csv, mechanism_checks.png
```

All rates are fractions of 100 held-out AdvBench prompts. "unsafe" columns are Llama Guard 3 1B labels
(lower is safer); "refusal" columns are phrase matching (higher is safer).

## How to reproduce

Everything runs on a free Kaggle GPU (T4, 16 GB).

1. Request access to [meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
   on Hugging Face, create a read token, and add it to Kaggle as a secret named `HF_TOKEN`
   (Add-ons → Secrets). The main model, `unsloth/Llama-3.2-1B-Instruct`, is ungated.
2. **Notebook 1:** Settings → Accelerator → GPU T4, then Save & Run All (about 1.5–2 h).
   The first cell stops with a message if PyTorch can't see a CUDA GPU.
3. **Notebook 2:** same settings, Save & Run All (about 2–2.5 h). It saves the trained models as
   `version_*.pt` (about 2 GB each) in its output.
4. **Notebook 3:** Add Input → Your Work → Notebooks → the finished notebook 2, then Save & Run All
   (about 20–30 min). No retraining.

Data is downloaded automatically: AdvBench from the llm-attacks GitHub repo and Alpaca from
Hugging Face (`tatsu-lab/alpaca`). Splits use seed 0 and are identical across notebooks.

Note: the notebooks write the models' answers to harmful prompts to `all_answers.json`,
`sweep_answers.json` and `check2_answers.json`. These are excluded from this repository on purpose.

## Known issues

- **Skipped steps in the fine-tuning attack.** Training uses fp16 autocast with a gradient scaler
  (initial scale 2^16). Steps whose scaled gradients overflow are skipped silently, so early
  "steps" in notebook 2 do not always change the weights, and C-ΔΘ loses the most steps.
  A fix is to count only applied updates or train in fp32/bf16.
- **C-ΔΘ was trained against the attack's answer format.** Its compliance model (θ⁻) is trained on
  AdvBench "Sure, here is …" targets, the same format the fine-tuning attack uses (on disjoint prompts).
- **One training run per defence.** Seeds in notebook 2 vary only the attack. Two trainings of
  C-ΔΘ with the same α gave noticeably different models.
- **Simplified C-ΔΘ.** Attribution patching instead of EAP-IG, α ∈ {0.25, 0.5, 0.75, 1.0},
  one pooled harm category.

## References

- Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- Huang et al. (2025), Attention Head Dropout. [Add link.]
- Kasliwal, Seth & Sankarapu (2026), [C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal](https://arxiv.org/abs/2602.04521)
- Zou et al. (2023), [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) (AdvBench)
- Meta, [Llama Guard 3 1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)

## Blog post

The full writeup is in [WRITEUP.md](WRITEUP.md). It was originally posted at [Spread it out vs keep it in one place: testing two ways to protect a model's refusal behaviour](https://forum.effectivealtruism.org/posts/MTHRT8qvjaGXDgFgm/spread-it-out-vs-keep-it-in-one-place-testing-two-ways-to-1) (EA Forum)
