# Stop-Content Decoupled On-Policy Distillation

Standard token-level OPD applies one KL direction to the complete vocabulary.
This couples two distinct decisions: whether to emit EOS, and which non-EOS
token to emit when continuing. Stop-content OPD factorizes those decisions.

For EOS probability `h` and the non-EOS conditional distribution `p_bar`,
reverse KL obeys the chain rule

```text
KL(p_student || p_teacher)
  = KL(Bern(h_student) || Bern(h_teacher))
    + (1 - h_student) KL(p_bar_student || p_bar_teacher).
```

The implementation keeps reverse KL for conditional content and makes the
Bernoulli stop KL independently configurable as `forward` or `reverse`.
Teacher and student EOS probabilities use exact full-vocabulary normalization;
the content term retains the repository's teacher-top-k approximation.

Relative to the flat reverse-KL baseline, the method arm changes two coupled
parts of the objective: it exposes the stop/content factorization and uses
forward KL for the exposed stop term. A follow-up ablation should set
`stop_kl_type: reverse` to isolate the effect of factorization from KL direction.

Enable the method with:

```yaml
loss_fn:
    kl_type: reverse
    stop_content:
        enabled: true
        eos_token_id: 151643
        stop_kl_type: forward
        stop_kl_weight: 1.0
        probability_eps: 1.0e-7
```

`teacher_logsumexp` and `teacher_eos_logits` are produced by the DTensor
teacher workers. The feature therefore currently requires the DTensor teacher
backend. W&B receives the conditional content KL, stop KL, and mean teacher and
student EOS probabilities under the standard training token mask.

Use `run_opd_reverse_vs_stop_content.sh` for the matched reverse-baseline and
factorized-method experiment.
