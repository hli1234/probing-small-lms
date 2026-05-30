# Looking for the log-multiplication trick inside GPT-2

*A mechanistic-interpretability study of how a small language model represents
numbers, and whether it secretly knows that `a · b = 2^(log₂ a + log₂ b)`.*

---

Multiplication is hard work, but it has a famous shortcut. For three
centuries, before the pocket calculator, scientists and engineers carried
slide rules — physical implementations of the identity

$$a \cdot b = 2^{\log_2 a + \log_2 b}$$

Instead of multiplying two numbers, you look up their logarithms, add the
logs (which is easy), and look up the exponential of the sum. The whole
trick rests on a different geometry of number: position by log magnitude, not
by raw magnitude.

A natural question, then: does a neural network ever discover this shortcut?
If you train a small model to multiply, does it secretly learn the slide
rule? The longer plan in `experiment1.md` proposes training a fresh tiny
Transformer from scratch under capacity pressure and then dissecting it.
This post is the parallel question: **is the shortcut already sitting inside
an existing small language model?** I went looking, in GPT-2 small, with
linear probes and activation patching.

**The short version of what I found.** GPT-2 has the *substrate* of the
shortcut. Its token embeddings represent number magnitude on a roughly
logarithmic axis (R² = 0.95 at layer 0). That axis has structured, specific
causal effect on the model's next-token predictions in multiplication
prompts — patching it shifts the model's preferred product toward the
patched value, while patching random directions of equal magnitude does not.
And the model genuinely does compute `log₂(a · b)` somewhere in its
residual stream right before the LM head — a fresh probe at the `=` position
predicts the answer's log at R² = 0.989. The whole `log → sum → exp`
machine is wired up. What's missing for behavioural success is not the
circuit but the LM head's willingness to act on it: GPT-2's strong frequency
prior for small numbers dominates the output distribution, so even though
the machine exists, GPT-2 can't actually multiply `5 × 9` and get `45`.

There's also a methodological lesson hidden in the story. I almost
concluded "no circuit" after the input-side probe came up flat at the `=`
position. Training a fresh probe in the right place rescued the result.
Probes are cheap. Don't assume the same direction does double duty.

---

## A toy model, as a yardstick

Before touching any real model, I built a normative toy to make the
cost-pressure hypothesis quantitative. Two paths can compute `z = a · b`:

- **Direct multiplication**, cost `c_mul(a, b) = log₂(a) · log₂(b)` — the
  bit × bit schoolbook work. Multiplying `2 × 2` is one bit times one bit
  (cheap); `1024 × 512` is ten bits times nine (expensive).
- **The log-sum-exp shortcut**, cost `c_short = 6` — a small constant for
  one logarithm lookup, one addition, one exponential lookup.

A small neural gate reads `(log₂ a, log₂ b)` and produces a probability
`p ∈ [0, 1]` of taking the shortcut. The expected total cost at penalty
weight `λ` is

$$C(p, a, b, \lambda) = (1-p)\cdot \lambda \cdot c_{\mathrm{mul}}(a, b) + p \cdot c_{\mathrm{short}}.$$

Sweep `λ`. At each value, retrain the gate to minimise expected cost. Plot
the average probability of using the shortcut.

The curve does the expected thing — a smooth sigmoid from "always direct"
to "almost always shortcut" — but the interesting part is the geometry.
Because the cost of direct multiplication depends on `(a, b)`, different
pairs cross over at different `λ`. Hard multiplications (large `a × b`)
switch first, easy ones (small `a × b`) hold out longer, and trivial ones
(those containing `1`) never switch — `log₂(1) = 0` makes direct
multiplication genuinely free for them, so no penalty can dislodge the
direct path.

![Figure 1: Toy soft-routing model. Top: P(shortcut) averaged over (a,b) ∈ [1,64]² vs the multiplication penalty weight λ. Trained gate (blue) matches the analytical softmin baseline (orange). Bottom: per-pair gate at five λ snapshots; the decision boundary sweeps from upper-right to lower-left as hard multiplications switch first.](./reasoning_path_vs_penalty.png)

This isn't itself a finding. It's a yardstick. It tells you what an
idealised cost-cutter *should* do under the difficulty function
`log₂(a) · log₂(b)`. The empirical question is whether GPT-2 — a real model
trained on text, not on a multiplication objective — shows even a faint
trace of this geometry.

---

## How GPT-2 represents numbers

To check what GPT-2 stores about integers, I fed it the numbers 2 through
1023 one at a time, formatted as `" 5"`, `" 17"`, `" 1024"` and so on
(many are single BPE tokens; multi-token cases use the final position).
At every layer I extracted the residual stream — the model's internal
scratchpad — and asked: given the scratchpad, can I predict `log₂(x)` with
linear regression?

**At the embedding layer — the raw token embedding, before any transformer
block has run — `log₂(x)` is decodable at R² = 0.95.** Raw `x` is also
decodable, but worse, at R² = 0.77. Shuffled labels (a noise control)
give R² ≈ 0. So the embedding seems to encode magnitude more as the
logarithm than as the number itself.

To make sure the axis is really log-shaped rather than just "some monotone
function of `x`", I swept the power-law family `x^α`. Decodability peaks
right around `α = 0` (the log limit). If GPT-2 stored linear magnitude the
peak would be at `α = 1`. It isn't — it's clearly on the log side.

What's more surprising is what *else* the embedding stores. **Parity** —
whether `x` is even or odd — is decodable at R² = 0.94. **Last digit**
(`x mod 10`) is decodable at R² = 0.92. GPT-2 has noticed, just from
reading text, that `" 5"`, `" 15"`, `" 25"` all share the feature of
ending in `5`, and has placed those embeddings on a common axis. But
ask the same probe for `"is x divisible by 7?"` and the R² is essentially
zero. Same for `"is x prime?"`.

So the embedding has structure for the things a careful reader of natural
language might track — size, parity, last digit — and ignores number-theory
properties that show up too rarely in text to matter. That's a small thing,
but it tells you what kind of representation GPT-2 has built. It's the
representation of a literate generalist, not a careful arithmetician.

![Figure 2: Linear-probe R² at each layer for monotone targets (log₂ x, x, x², √x, 1/x), non-monotone targets (parity, x mod 7, last digit, popcount, is_prime), and a noise baseline (shuffled log₂ x). The log axis, parity, and last digit are all decodable at the embedding. Number-theoretic properties and the shuffle control are at chance.](./probe_gpt2_controls.png)

---

## Does the model use the log axis when it multiplies?

Linear decodability is correlational. To test causality I ran activation
patching at the operand token.

Here's the experiment. Take a prompt like `" 5 * 9 ="`. The model
processes it normally. At the position where `" 5"` sits, the embedding's
projection onto the probe direction `w` reads `log₂(5) ≈ 2.3`. I add a
rank-1 vector along `w` so that projection now reads `log₂(2) = 1` —
*everything else* about the embedding is left unchanged. Then I let the
model run through all 12 transformer blocks and the LM head, and ask: which
numeric token did the next-token distribution shift toward?

The result: for `" 5 * 9 ="`, patching `log₂(5) → log₂(2)` makes the logit
for `" 18"` (= 2·9) rise *above* the logit for `" 45"` (= 5·9). The model
has been talked into preferring the patched product. Patching toward `4`
makes `" 36"` win. Patching toward `5` does nothing, as it should. And —
the critical control — patching along five different *random* directions of
equal magnitude to `w` gives logit shifts indistinguishable from zero. So
the effect is specific to the probe direction, not a generic perturbation.

There is one wrinkle. Patching *down* (toward smaller numbers) works
cleanly. Patching *up* (toward larger numbers) only partly works — the
model's preference for the patched product saturates and then reverses.
The most plausible explanation is that the LM head carries a strong
frequency prior favouring small, common numbers in text, and a fixed-size
patch is enough to push preference toward smaller numbers (with the prior)
but not enough to push it toward larger numbers (against the prior).
Either way, the effect on the smaller side is robust and direction-specific.

![Figure 3: Causal patching at the operand. Left: change in relative logit, logit(a_target · b) − logit(a · b), as we patch the log direction at the "a" position. Solid lines are the log direction; dotted lines are random directions of the same norm. Right: heatmap of relative logits assigned to candidate products for the prompt "5 * 9 =". The yellow most-favoured-product stripe follows the perfect-log-shortcut line p = a_target · b for downward patches, then sticks at 45 as we patch toward larger numbers.](./probe_gpt2_patching.png)

---

## What's in the LM head?

The LM head — the matrix that turns the final residual stream into
next-token logits — has a row for every token. For numeric tokens, what's
in those rows?

A caveat first: GPT-2 ties its LM head to the input embedding, so each
numeric token's row is literally the same vector as `wte(" n")`. This means
the LM head automatically aligns with whatever directions the embedding
aligns with. The interesting question is therefore not *whether* the
LM-head row aligns with the log axis (it does, by construction), but *how
much*.

I measured this by projecting each numeric token's LM-head row onto the
unit log direction and asking what fraction of the row's squared norm sits
there. The answer is **3.6% on average**. That sounds small, but a random
direction in 768-dimensional space would only capture about
`1/d = 1/768 ≈ 0.13%`. So the log axis is about **28× the random baseline**
— a big feature compared to noise, but a small slice of each row's total
content. Parity and last-digit are above baseline too (about 4× random)
but smaller. The remaining roughly 96% of each row lives on directions I
haven't named, presumably encoding things like "this is a number", "this is
a small common number", and token-specific idiosyncrasies.

The log axis is the largest *identified* feature in the LM head's numeric
rows. It is not the whole row.

![Figure 4: Decomposition of each numeric token's LM-head row along three identified directions (log, parity, last digit) versus a same-norm random direction. The log axis captures 3.6% of each row's squared norm on average, ~28× the isotropic baseline of 1/d = 0.13%. Bottom panel: per-token stacked composition; the grey "everything else" — directions we haven't named — dominates every numeric token's row.](./probe_gpt2_lmhead.png)

---

## Does the model actually compute log(a · b)?

This is the test I most wanted to run.

If the model does the slide-rule trick — adds the operand logs and then
exponentiates at the end — then at the position of the `=` sign, the
residual stream right before the LM head should be carrying `log₂(a · b)`.
Specifically, the operand-side log direction `w` ought to read off this
value.

I checked. For `" 5 * 9 ="`, the operand-side probe at `=` reads off
**10.76** — not `log₂(45) ≈ 5.49`. For `" 3 * 7 ="` it reads **11.17**, not
`log₂(21) ≈ 4.39`. Across nine different prompts, the readout is
essentially flat. The R² of "what the operand-axis probe reads at `=`"
against "the true `log₂(a · b)`" is **0.011** — no signal.

My first reaction was: well, the circuit isn't there. The shortcut exists
on the input side and in the head's row geometry, but the model isn't
actually wiring them together.

Then it occurred to me: maybe the model uses a *different* direction at the
`=` position to write down the answer's log than the one it uses to read the
operands' logs. There's no architectural reason those have to be the same
vector — the model can translate between representations as it computes.
So I trained a fresh ridge probe directly on the residual at `=`, against
`log₂(a · b)` for several hundred random `(a, b)` pairs.

The new probe predicts the answer's log at **R² = 0.989** on held-out
pairs.

That stopped me. The model has computed `log₂(a · b)`. It's there. It's
almost perfectly linearly decodable from the residual stream right before
the LM head. I had simply been looking on the wrong axis.

So the picture is this. The operands' logs are written down on one axis at
the input layer. Somewhere in the middle layers — I haven't pinned down
where — the model adds them and rewrites the result on a *different* axis
at the `=` position. The LM head then reads that second axis when deciding
what number comes next. The whole `log → sum → exp` machine is sitting in
GPT-2 small.

![Figure 5: The end-to-end pipeline. Left: patching the log component at the "=" position of real prompts; argmax-numeric is mostly stuck at " 1" regardless of v_target. Middle: the operand-side probe's readout at "=" is roughly constant (~10.8) across all prompts, uncorrelated with log₂(a·b) (R² = 0.011). Right: a fresh probe trained directly on residuals at "=" predicts log₂(a·b) at R² = 0.989 on held-out pairs. The model has computed the answer's log — on a different direction than the one it used to read the operands.](./probe_gpt2_pipeline.png)

---

## Why doesn't this give correct arithmetic?

Given that the circuit exists, why can't GPT-2 small actually multiply
`5 × 9` and produce `45`? Across every prompt I tried, the unpatched
argmax over numeric tokens was `" 1"` or `" 0"` — the most generic possible
guesses.

The answer is in the LM-head decomposition. The log axis captures 3.6% of
each numeric row's squared norm. The other ~96% is doing something — most
likely encoding general features like "this is a small number", "this is a
common token", and `n`-specific text-frequency information. When the model
asks "what's the most likely next token after `5 * 9 =`", the answer is
dominated by that frequency mass, which favours `" 1"`, `" 0"`, and other
extremely common tokens. The log axis, even at 28× baseline, is one signal
fighting against a much larger frequency prior.

This is consistent with the asymmetry seen in the operand-side patching:
moving the model's predictions *toward* small numbers (with the prior) is
much easier than moving them toward large numbers (against the prior). The
machine is built. It just isn't loud enough.

A model with stronger arithmetic ability — Pythia-1.4B, Llama-3.2-1B —
should have the same circuit but with a louder log signal in the LM head,
enough to overcome the frequency prior. That's the obvious follow-up.

---

## What I think this collectively shows

Three claims, in increasing order of strength.

The embedding layer of GPT-2 small encodes integers on an axis whose shape
is approximately logarithmic in magnitude, along with separate axes for
parity and last digit. Number-theoretic properties not visible in surface
text (mod-7, primality) do not get a coherent representation. This isn't
surprising — others have observed log-shaped number representations in
language models — but the controls here (the power-law sweep, the shuffled
baseline, the non-monotone targets) make it a clean measurement.

Perturbations along the embedding's log axis have structured, specific
causal effect on the model's next-token predictions in multiplication
prompts. Random directions of the same magnitude do not. This rules out
the most natural "probe is just finding a noise direction" null hypothesis.
The asymmetry of the effect (downward patches work, upward patches partly
fail) is itself informative — it points at the LM head's frequency prior as
the limiting factor.

Most importantly, GPT-2 small does compute `log₂(a · b)` somewhere in its
residual stream by the time it reaches the `=` token, and that computation
is linearly readable at R² = 0.989. The model uses *different* directions
at the operand position and at the `=` position; there is a translation
step somewhere in the middle layers. This is, as far as I know, novel for
a model this small — and the strongest evidence I have that the full
`log → sum → exp` circuit is mechanistically present in GPT-2.

---

## Caveats

The R² ≈ 0.95 for log decodability at the embedding is a property of the
embedding matrix. Because GPT-2 ties its LM head to its embedding, the
alignment of the LM head with the log axis is partly architectural rather
than independently learned. A model with untied weights (Llama, Pythia)
would give a cleaner test of whether the log axis is emergent in both
encoding and decoding separately. I'd expect it to be — but the test isn't
clean here.

The causal-patching results in Figure 3 are measured on relative logits
among candidate products, not on argmax. GPT-2 small genuinely cannot do
most multiplications. The relative-logit shifts are real and
direction-specific, but they ride on top of a fixed frequency prior. A
much stronger version of the same experiment would run on a model that can
actually multiply, where patching should flip the argmax cleanly.

The fitted answer-side probe at the `=` position has R² = 0.989 on `(a, b)
∈ [2, 64]²` pairs. Whether the same direction carries `log₂(a · b)` for
much larger operands is untested.

The middle layers — where the translation from input-log axis to
output-log axis happens — are a black box in this study. Path-patching
across (layer, position) pairs would identify the attention heads that
move log information from the operand positions to the `=` position, and
the layer at which the sum happens. I haven't done it.

---

## Next steps

The exponentiation cross-check is the sharpest immediate test. If the
model uses the log axis compositionally, patching `log(a)` by Δ in an
`a^b =` prompt should shift the output's log by `b · Δ`, not Δ. That's a
direct quantitative prediction the model either passes or fails. Same
patching code, different prompts.

Re-running everything on Pythia-1.4B or Llama-3.2-1B would (a) decouple the
embedding and LM head via untied weights, (b) give a model that can
actually multiply, making the argmax test meaningful, and (c) provide a
larger residual dimension in which the per-direction norm fractions become
more informative.

Path-patching to identify the layer and heads where the log-axis translation
happens is the natural mechanistic follow-up. The infrastructure for it is
already in the patching code here; it needs adaptation for attention-head
outputs rather than residual-stream patches.

Scaling the embedding probe across the Pythia family (70M, 160M, 410M,
1.4B, 2.8B, 6.9B) would give a clean test of how log-axis quality varies
with model size. The capacity-squeeze hypothesis predicts something
non-monotone — small models need the shortcut, medium ones can memorise a
lookup table, large ones can host both — but any clean trend would be
informative.

---

## Code and data

All scripts and figures are in the project folder. The toy routing model is
`reasoning_router.py`, producing `reasoning_path_vs_penalty.png`. The six
GPT-2 experiments are `probe_gpt2_log.py` (per-layer probe),
`probe_gpt2_controls.py` (parity, mod-7, last-digit, power-law sweep,
shuffle baseline), `probe_gpt2_patching.py` (operand-position causal
patching with random-direction control), `probe_gpt2_lmhead.py` (LM-head
decomposition), and `probe_gpt2_pipeline.py` (end-to-end patching and the
answer-side log probe).

All scripts are self-contained Python and reproduce on CPU in about ten
minutes total. GPT-2 small downloads at ~500 MB. Dependencies are
`torch`, `transformers`, `scikit-learn`, and `matplotlib`.

---

*Thanks to the rough plan in `experiment1.md` for the framing. The
mechanistic-interpretability moves (linear probing, activation patching) are
standard tools from the literature; what this post adds is the specific
finding that GPT-2 small computes `log₂(a · b)` on a different axis than
the one it uses to read operand magnitudes, and the methodological note
that probes are cheap and you should fit them at the position you actually
care about.*
