# NECIL-HSI Architecture Specification

## 1. Problem the Architecture Must Solve

The architecture targets four coupled failures in non-exemplar class-incremental hyperspectral image classification:

1. **Feature-space overlap**
   - Spectrally similar classes occupy competing regions of the learned representation.
   - This produces ambiguous decisions even before incremental learning and becomes more severe when new classes are introduced.

2. **Representation drift**
   - Updating the HSI backbone for new classes changes the representation of historical classes.
   - Persistent old decision information can become incompatible with the updated feature space.

3. **Boundary information loss**
   - Old HSI samples are unavailable after their learning phase.
   - Without persistent discriminative structure, the model loses explicit information about how historical classes were separated.

4. **Classifier imbalance**
   - A conventional trainable cumulative classifier is updated mainly using current-phase classes.
   - This can bias decisions toward new classes and make old/new scores incomparable.

These failures are related but are not identical:

\[
\text{feature-space overlap}
\rightarrow
\text{ambiguous class boundaries}
\]

\[
\text{representation drift}
\rightarrow
\text{historical boundaries become incompatible}
\]

\[
\text{boundary information loss}
\rightarrow
\text{old discrimination cannot be reconstructed from real old HSI}
\]

\[
\text{classifier imbalance}
\rightarrow
\text{old/new decision scores become biased}
\]

The architecture assigns a specific component to each failure.

---

# 2. Architecture Components

## 2.1 HSI Backbone

### Role

Learn one canonical HSI representation

\[
z=f_{\theta}(X)
\]

from:

- the ordered full-band center spectrum;
- center-relative local spatial context.

The full spectrum is the spectral anchor.

Spatial context provides a residual refinement of the same representation:

\[
z=z_s+\Delta z_{\text{context}}.
\]

There is no separate spectral geometry space and no separate spatial geometry space.

### Problem addressed

- feature-space overlap;
- discriminative HSI representation learning.

The backbone must organize spectrally similar classes so that their discriminative interfaces can be learned.

---

## 2.2 Pairwise Boundary Geometry Bank

For every unordered pair of represented classes \(a,b\), store one shared affine boundary:

\[
h_{ab}(z)=n_{ab}^{T}z+q_{ab},
\qquad a<b.
\]

The same boundary is used with opposite orientation by the two classes.

For class \(c\) relative to rival \(j\):

\[
s_{cj}(z)=
\begin{cases}
h_{cj}(z), & c<j,\\
-h_{jc}(z), & j<c.
\end{cases}
\]

The decision cell of class \(c\) is

\[
\mathcal C_c
=
\left\{
z:
s_{cj}(z)\ge0,\ \forall j\neq c
\right\}.
\]

The class energy is

\[
E_c(z)
=
-\min_{j\neq c}s_{cj}(z).
\]

Therefore:

\[
E_c(z)\le0
\iff
z\in\mathcal C_c.
\]

### Role

Persist the **discriminative interfaces between classes**, rather than storing only where individual classes are centered.

### Problems addressed

- feature-space overlap at the decision level;
- boundary information preservation;
- old/new boundary extension during incremental learning.

Strict interiors of two class cells cannot overlap because the same pair boundary is used with opposite signs.

---

## 2.3 Equal-Rule Geometry Classifier

The classifier has no class-specific trainable weights.

For every represented class:

\[
\operatorname{logit}_c(z)=-E_c(z).
\]

Prediction is:

\[
\hat y=\arg\min_c E_c(z).
\]

### Role

Apply exactly the same decision rule to old and new classes.

### Problem addressed

- classifier imbalance.

There is no separately trained old head, new head, calibration bias, temperature, or class-specific classifier parameter.

---

## 2.4 Spectral Replay

Spectral replay is used only after the base phase.

### Role

Provide historical HSI evidence when real old HSI is unavailable.

Replay has two architectural responsibilities:

1. keep the evolving backbone compatible with persistent old-old boundaries;
2. provide the missing old-class side needed to learn old-new boundaries.

Replay is therefore not generic sample generation.

It is historical spectral evidence used to preserve and extend discriminative geometry.

### Problems addressed

- representation drift;
- old/new feature-space interference;
- boundary preservation;
- learning old-new interfaces without real old exemplars.

---

# 3. Base Phase

## 3.1 Input

Base real HSI samples:

\[
D_0=
\{(X_i,y_i)\}.
\]

Only base-phase real data is available.

---

## 3.2 Representation Learning

Each HSI input is encoded:

\[
z_i=f_{\theta_0}(X_i).
\]

The backbone learns spectral-primary HSI features in one canonical representation space.

---

## 3.3 Base Boundary Construction

For every base class pair \(a,b\), create one trainable shared boundary:

\[
h_{ab}(z).
\]

The complete base geometry is:

\[
\mathcal G_0
=
\left\{
h_{ab}:
a,b\in\mathcal C_0,\ a<b
\right\}.
\]

The boundaries and the backbone are optimized together.

---

## 3.4 Base Objective

Use only the decision-relevant class-uniform objective:

\[
L_{\text{base}}
=
\lambda_{\text{cls}}L_{\text{cls}}
+
\lambda_{\text{fit}}L_{\text{fit}}.
\]

### Classification

\[
L_{\text{cls}}
=
CE(-E,y).
\]

This trains class discrimination using the deployed geometry score.

### Decision-cell fit

\[
L_{\text{fit}}
=
\operatorname{ReLU}(E_y).
\]

This penalizes a real sample when it violates any boundary defining its own class cell.

No box-overlap loss is needed.

No prototype loss is used.

No arbitrary boundary margin is introduced.

---

## 3.5 Base Finalization

The finalized state contains:

\[
\boxed{
\theta_0,\quad
\mathcal G_0
}
\]

where:

- \(\theta_0\) is the finalized HSI backbone;
- \(\mathcal G_0\) is the learned base pairwise boundary bank.

The exact learned boundaries used during optimization are persisted.

There is no train-time geometry that is discarded and replaced with another geometry.

---

# 4. Incremental Phase \(t\)

Let:

\[
\mathcal C_{\text{old}}
\]

be all previously learned classes and

\[
\mathcal C_{\text{new}}
\]

the current new classes.

At the beginning of phase \(t\):

- real new HSI is available;
- real old HSI is unavailable;
- old-old pairwise geometry is persistent.

---

## 4.1 Persistent Old Geometry

Retain all historical old-old boundaries:

\[
\mathcal G_{\text{old}}
=
\{
h_{ab}:
a,b\in\mathcal C_{\text{old}}
\}.
\]

These boundaries preserve historical class discrimination.

They are not reconstructed from current new data.

---

## 4.2 Encode Real New HSI

Current real HSI is encoded:

\[
z_n=f_{\theta_t}(X_n).
\]

Score the new features against persistent old geometry.

This reveals which old classes or old boundaries are most threatened by the new representation.

---

## 4.3 Spectral Replay

Generate/reconstruct old-class HSI evidence:

\[
\tilde X_o.
\]

Replay samples are passed through the current backbone:

\[
\tilde z_o=f_{\theta_t}(\tilde X_o).
\]

The replayed old representations must remain compatible with the persistent old geometry.

For old class \(c\):

\[
E_c^{\text{old}}(\tilde z_o)
\]

provides the historical compatibility signal.

This prevents the backbone from drifting freely away from the old decision structure.

---

# 5. Representation-Drift Control

Representation drift is controlled by using replayed old HSI with persistent old boundaries.

For replayed old class \(c\):

\[
L_{\text{old-fit}}
=
\operatorname{ReLU}
\left(
E_c^{\text{old}}(\tilde z_o)
\right).
\]

If the updated backbone moves old-class evidence across an old boundary, this term becomes positive.

The backbone is therefore trained to restore compatibility with historical geometry.

The default architecture does not transport old geometry.

Alignment is introduced only if replay leaves measurable residual drift.

---

# 6. Learning Incremental Boundaries

The old-old geometry already exists.

The current phase introduces only the missing boundaries.

## 6.1 Old-New Boundaries

For every:

\[
a\in\mathcal C_{\text{old}},
\qquad
u\in\mathcal C_{\text{new}},
\]

learn:

\[
h_{au}.
\]

Real new HSI supplies the new-class side.

Spectral replay supplies the old-class side.

Thus:

\[
\boxed{
\text{replayed old evidence}
+
\text{real new evidence}
\rightarrow
\text{old-new boundary}
}
\]

---

## 6.2 New-New Boundaries

For every pair:

\[
u,v\in\mathcal C_{\text{new}},
\qquad
u<v,
\]

learn:

\[
h_{uv}.
\]

Both sides use real current-phase HSI.

---

# 7. Feature-Space Overlap Control

Feature-space overlap is handled at two connected levels.

## Representation level

The backbone is optimized by the geometry-derived classification and fit losses so that confusable HSI classes become discriminable.

## Decision level

Each class pair shares one boundary.

For classes \(a,b\):

\[
h_{ab}(z)>0
\]

supports class \(a\), while

\[
h_{ab}(z)<0
\]

supports class \(b\).

Therefore a feature cannot be in the strict interiors of both class cells simultaneously.

During incremental learning:

- replayed old HSI constrains the old side;
- real new HSI constrains the new side.

This directly targets old/new feature-space interference.

---

# 8. Boundary Preservation

Boundary preservation has two parts.

## 8.1 Preserve old-old boundaries

Historical boundaries remain persistent:

\[
h_{ab}^{t}
=
h_{ab}^{t-1},
\qquad
a,b\in\mathcal C_{\text{old}}.
\]

Replay adapts the current representation to remain compatible with them.

## 8.2 Extend the geometry

Add only:

\[
\mathcal G_{\text{old-new}}
\]

and

\[
\mathcal G_{\text{new-new}}.
\]

After phase \(t\):

\[
\mathcal G_t
=
\mathcal G_{t-1}
\cup
\mathcal G_{\text{old-new}}
\cup
\mathcal G_{\text{new-new}}.
\]

Thus historical discrimination is retained while new discrimination is added.

---

# 9. Classifier-Imbalance Control

All classes use exactly the same energy:

\[
E_c(z)
=
-\min_{j\neq c}s_{cj}(z).
\]

All logits are:

\[
-E_c(z).
\]

There is no trainable cumulative classifier head.

Therefore old and new classes are not assigned separate classifier parameter groups.

Current-phase class-frequency imbalance is handled by class-uniform empirical risk.

The cumulative decision mechanism remains:

\[
\boxed{
\hat y
=
\arg\min_{c\in\mathcal C_{\text{seen}}}
E_c(z)
}
\]

for every phase.

---

# 10. Incremental Phase Objective

The incremental objective uses the same decision semantics for real-new and replay-old HSI.

For real-new samples:

- classify against all seen classes;
- fit them to their correct new decision cells;
- their old-new boundaries are learned jointly with replayed old evidence.

For replay-old samples:

- classify against all seen classes;
- preserve compatibility with persistent old-old boundaries;
- provide the old-class side for old-new boundary learning.

Conceptually:

\[
L_t
=
L_{\text{real-new}}
+
L_{\text{replay-old}}.
\]

Both terms use the same geometry energy and equal-rule classifier.

No separate old classifier and new classifier are introduced.

---

# 11. Phase Finalization

At the end of phase \(t\):

### Keep

- finalized backbone \(\theta_t\);
- all persistent old-old boundaries;
- newly learned old-new boundaries;
- newly learned new-new boundaries;
- only the compact spectral state required by the replay mechanism.

### Do not keep

- real old exemplars;
- old HSI patches;
- prototype memory;
- teacher model;
- independent old/new classifier heads.

The geometry bank becomes:

\[
\boxed{
\mathcal G_t
=
\{
h_{ab}:
a,b\in\mathcal C_{\text{seen}}^{t},
a<b
\}
}
\]

and is used directly in the next phase.

---

# 12. Complete Architecture Flow

```text
BASE PHASE
==========

real base HSI
      │
      ▼
HSI spectral-primary backbone
      │
      ▼
canonical representation z
      │
      ▼
learn all base pairwise boundaries
      │
      ▼
decision cells
      │
      ▼
equal-rule energy classifier
      │
      ▼
persist backbone + pairwise geometry


INCREMENTAL PHASE t
===================

                    real new HSI
                         │
                         ▼
                  current backbone
                         │
                         ▼
                    new features
                         │
                         ▼
                persistent old geometry
                         │
                         ▼
              identify old/new conflict
                         │
                         ▼
                   spectral replay
                         │
                         ▼
                  replayed old HSI
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       replayed old              real new
             │                       │
             ▼                       ▼
       current backbone         current backbone
             │                       │
             ▼                       ▼
          old features             new features
             │                       │
             ├──────────┬────────────┤
             │          │            │
             ▼          ▼            ▼
        preserve     learn        learn
        old-old     old-new      new-new
        boundaries  boundaries   boundaries
             │          │            │
             └──────────┴────────────┘
                        ▼
                extended geometry bank
                        │
                        ▼
                 same energy E_c(z)
                        │
                        ▼
                    argmin E_c
                        │
                        ▼
                    finalize phase
```

---

# 13. Architecture-to-Problem Mapping

| Problem | Architectural solution |
|---|---|
| Feature-space overlap | HSI backbone + shared pairwise boundaries + decision-cell fit |
| Representation drift | spectral replay evaluated through persistent old boundaries |
| Boundary information loss | persistent old-old pairwise boundary geometry |
| Boundary preservation | retain old-old boundaries; replay maintains representation compatibility |
| Old-new interference | replay-old + real-new jointly define old-new interfaces |
| Classifier imbalance | parameter-free equal-rule energy classifier |
| HSI class-frequency imbalance | class-uniform empirical risk |
| Missing old data | spectral replay; no real old exemplar storage |

---

# 14. Architecture Principle

The complete architecture follows one fixed division of responsibility:

\[
\boxed{
\text{Backbone learns HSI features}
}
\]

\[
\boxed{
\text{Pairwise geometry preserves and extends discriminative boundaries}
}
\]

\[
\boxed{
\text{Spectral replay preserves old representation compatibility and supplies old boundary evidence}
}
\]

\[
\boxed{
\text{Equal-rule energy classification prevents cumulative classifier-head imbalance}
}
\]

Feature-space overlap is reduced by learning discriminative representations and explicit shared class-pair interfaces.

Representation drift is controlled through replay against persistent historical boundaries.

Boundary information is preserved directly rather than reconstructed from point prototypes.

Incremental phases extend the existing discriminative geometry instead of rebuilding the historical class structure.
