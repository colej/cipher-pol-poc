## Experiment 1: Is dissipation identifiable?

Inputs. Snapshots $\mathbf{u}(t_k)$ with $k \ge k_\text{min}$ from held-out trajectories. At $t=0$ the initial energy 
sits at modes $i,j \le 10$, mostly below the cutoff, so $B$ is small. Choose $k_\text{min}$ from the data by computing 
$\|B\|/\|A\|$ per snapshot before any model runs.

Fit. Pooled over grid points and inputs, with $X = [A\;\;B]$:

$$g = \alpha A + \gamma B + r, \qquad \mathbf{c} = \begin{bmatrix}\alpha\\ \gamma\end{bmatrix} = (X^\top X)^{-1}X^\top g.$$

Reference coefficients are (1, 1) as $t \to 0$. Repeat with the solver increment $(\mathcal S_t(\mathbf{u}) - \mathbf{u})/t$ 
from the data at the same $t$. The solver fit is a reference, not a ceiling: a model that takes one Euler step could align 
better with $A$ and $B$ while being less accurate.

#### What to Report

1) Model error relative to the increment: $\|f_\theta - \mathcal S_t\| / \|\mathcal S_t(\mathbf{u}) - \mathbf{u}\|$.
2) $\hat\alpha, \hat\gamma$ with trajectory-bootstrap CIs, for model and solver.
3) $\Delta R^2$ from adding $B$ to an $A$-only fit.
4) $\|B\|$ against the model-error term $\|\epsilon_\theta/t\|$ and against the $O(t)$ correction (estimated from the difference between fits at $t = 0.05$ and $t = 0.10$).
5) Identity error $\|f_\theta(\mathbf{u},0) - \mathbf{u}\|$. Zero-lead-time pairs are in training.
6) Correlation of $A$ and $B$, and the condition number of $X^\top X$.

#### Positive result
$\hat\gamma$'s CI excludes 0 for both model and solver, and $\|B\| > \|\epsilon_\theta/t\|$ on the chosen inputs. 


## Experiment 2. Channel-group ablation screen

Narrow hypothesis: at one activation site, the model already separates the two contributions into different fixed groups of units. 

#### 1.1 Site

One site, fixed before results; record it in CLAUDE.md. scOT is U-shaped: SwinV2 stages joined by patch merging/expansion, with ConvNeXt blocks on the skip connections. This constrains the choice:

Bottleneck (matches a SESSIONS.md probe layer). This is not a cut: information reaches the output through the skips without passing it. Small-scale content, where $B$ lives, is the most likely to bypass it, so effects there are bounded.
Mid-depth encoder block at high resolution. This is a cut for everything downstream, including its skip branch.

Default: the encoder option. Layer names are unverified.

Basis. Residual-stream channels are not generally a privileged basis; their only per-channel structure comes from the time-conditioned LayerNorm gains. MLP hidden units after GELU are privileged. Default: the MLP hidden units of the chosen block.

#### 1.2 Groups (decision point)
A. Fixed partition into $G = 16$ disjoint groups, by index or seeded random, fixed before results. Simple, but a selective group only appears if term-relevant units happen to cluster in one group. If they are spread out, every group looks the same, so a negative result is weak.
B. Single-unit discovery, then grouping. Score each unit on discovery trajectories, group units by their $(\tilde a_i, \tilde b_i)$, and confirm the groups on untouched trajectories. More forward passes, but no arbitrary grouping.

Default: A. If A is null, run B before moving to Experiment 2.

#### 1.3 Intervention

For units in group $G$, attenuate toward the discovery-set mean:

$$h_G' = \mu_G + (1-\lambda)(h_G - \mu_G), \qquad \lambda = 0.2 \text{ for discovery}.$$

$\mu_G$ is pooled over tokens and inputs (alternative: per token position; decide before running). This is an intervention; it does not guarantee activations stay on-distribution.

#### 1.4 Readout

Change relative to the model's own baseline at the fixed lead time $t = 0.05$:

$$\delta g_G = \frac{f_{\theta,G}(\mathbf{u},t) - f_\theta(\mathbf{u},t)}{t} = a_G A + b_G B + \eta_G.$$

Differencing against the baseline cancels $\mathbf{u}$ and the baseline fit. Report the normalised coefficients and the unexplained fraction:

$$\tilde a_G = a_G/\hat\alpha, \qquad \tilde b_G = b_G/\hat\gamma, \qquad \rho_G = \|\eta_G\| / \|\delta g_G\|.$$

Pre-registered thresholds (values [TBD before running]): suppression $s$, preservation tolerance $\tau$, residual cap $\rho_\text{max}$.

Observed change	Interpretation
$\tilde a_G \le -s$, $\lvert\tilde b_G\rvert \le \tau$, $\rho_G \le \rho_\text{max}$	Group selectively supports the advection-aligned contribution
$\lvert\tilde a_G\rvert \le \tau$, $\tilde b_G \le -s$, $\rho_G \le \rho_\text{max}$	Group selectively supports the dissipation-aligned contribution
Both coefficients outside $\tau$	Group affects both contributions
$\rho_G > \rho_\text{max}$	Intervention changes the output in ways these terms do not explain

"Preserved" means within $\tau$, not merely statistically insignificant. A positive $b_G$ (added smoothing) counts as not preserved.

1.5 Null

The 16 groups are not a null. Draw random unit sets of the same size (e.g. 100) and apply the same intervention and readout. A candidate must meet the criteria above and lie outside the null cloud.

1.6 Discovery figure

Scatter of $(\tilde a_G, \tilde b_G)$ for each group, coloured by $\rho_G$, over the random-set null cloud. Look for reliable points near the negative horizontal axis and others near the negative vertical axis.

1.7 Confirmation

On untouched trajectories, test the best candidate for each term at three attenuation strengths, e.g. $\lambda \in \{0.1, 0.2, 0.4\}$ (placeholders). Require:

suppression of the intended coefficient that increases with $\lambda$;
the other coefficient within $\tau$;
$\rho_G \le \rho_\text{max}$;
reproducibility across trajectories (bootstrap CI).

Zero-lead-time check. $\delta g$ is undefined at $t=0$, so compare raw output changes: $\|f_{\theta,G}(\mathbf{u},0) - f_\theta(\mathbf{u},0)\|$ against $t\,\|\delta g_G\|$. If they are comparable, the effect is damage to the model's reconstruction of the input, not a change in tendency.

Spectral check. $B$ is a filtered high-pass of $\mathbf{u}$. An intervention that generically damages small-scale content can project onto $B$. Compare the radially averaged spectrum of $\delta g_G$ with that of $B$, and flag candidates whose effect is a broad high-$k$ change.

1.8 What it establishes
Positive: different unit groups selectively influence the advection-aligned and dissipation-aligned parts of the prediction. This is causal evidence, but it does not show that the groups compute the operators.
Negative: this grouping at this site did not reveal separation. A rotated or distributed representation could be missed, which motivates Experiment 2.

Cost: one baseline forward pass, plus one pass per group or random set, per input.

## Experiment 3 (Optional). Subspace search

Run only if Experiment 1 (A and B) is null, or to test whether a rotated basis does better. Changes from v1:

Locate with gradients of the fitted coefficients rather than raw output alignment. With $X$ fixed by the input,
$$\nabla_{h_\ell} c_j = \left(\frac{\partial g_\theta}{\partial h_\ell}\right)^{\!\top} X (X^\top X)^{-1} \mathbf{e}_j.$$
Stack over tokens and discovery inputs, then take the top-$r$ singular vectors to get a channel subspace in $\mathbb R^{d_\ell}$. Applied at every token, its projection changes up to $n_\ell r$ dimensions, so report $r/d_\ell$ and $n_\ell$ per site.
Intervene with $h_\lambda = h - \lambda P_S(h - \mu)$ and sweep $\lambda$.
Read out with the same $\delta g$ regression: the full $2\times2$ effect matrix plus the residual, rather than a single specificity score.
Nulls: random subspaces matched on removed activation energy; directions selected for generic output sensitivity (e.g. the gradient of $\|g\|^2$); the same selection procedure run on shuffled input–term pairings.
Selection: choose sites and $r$ on validation trajectories and confirm on untouched ones, or use a null that includes the whole search (max statistic).
Site ordering must follow the U-shaped topology. Separation found only near the output is weak evidence about upstream computation.
