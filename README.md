# cipher-pol
Computational Interpretability for Physics via Hidden Encoded Representations in Pretrained Operator Learning

Proof of Concept work for using LLM interpretability tools on PDE foundation models

---

## Introduction

The evolution of many physical systems, including fluid flows, are governed by a set of partial differential equations (PDEs). Generally speaking, PDEs specify a relationship between quantities/fields and their derivatives with respect to multiple variables. Depending on the given system a PDE can describe evolution, equilibrium, or constraints. Each term in the equation has a mathematical form (determined by the operator) and a physical meaning (e.g. transport, dissipation, pressure). Together these terms define how different regimes defined by balances arise and transition as those balances shift and the stability of solutions change.

At scale, numerically calculating the outcome of these PDEs to a high fidelity is computationally exhaustive. Thus, compromises are made, either to resolution, numerical accuracy, or to the fidelity of the assumed physics. Instead, researchers have explored high fidelity machine learning surrogates for solving PDEs.

Decades of work have demonstrated that using vanilla neural networks (NNs) to emulate physics processes is, at the very least, extremely challenging. As a result, there has been an explosion of methods and models dedicated to producing more accurate AI physics surrogates. In particular, we are interested in Neural Operators (NOs). 

NOs are a class of machine learning models that learn the operator that maps between two infinite dimensional function spaces together. While they are being rapidly adopted in a number of areas, we do not yet know how they compute internally. Correct predictions do not imply physical structure: a model may correctly reproduce solutions while its internal computation has no correspondence to the terms in the governing equation.

Here, we investigate whether a pretrained PDE foundation model's computation is factored or entangled. A factored internal representation would contain identifiable and causally separable components that correspond to terms, operators and regimes. An entangled internal representation would map state to state without that structure. We look to address this by using techniques developed in mechanistic interpretability research. In this work, we use Poseidon-T (Herde et al., 2024) with 21 million parameters for our experiments.

## Hypothesis and experiments

We construct a hypothesis and three experiments: Poseidon-T's activations contain components that selectively mediate the parts of its inferences which correspond to two terms of incompressible Navier–Stokes, projected advection and the dataset's spectral dissipation. 

Experiment 1: Is spectral dissipation measurable?
Experiment 2: Can a channel-group ablation screen at a single activation site identify causal links to observed features in predictions?
Experiment 3: (Optional) Search for rotated subspaces. 

---



> Fixed, low-dimensional groups of units (E2) or channel subspaces (E3) at a specified activation site in Poseidon-T selectively mediate output contributions aligned with projected advection and with the dataset's spectral dissipation.

- **Positive result:** evidence of causal separability under the tested interventions. It does not show that the component *computes* the operator; it may carry information used downstream.
- **Negative result:** no separability was detected within the representation class, sites, ranks and interventions tested. Distributed, nonlinear or input-dependent implementations are not excluded.

---

## References

- Herde, M., et al. (2024). Poseidon: Efficient foundation models for PDEs. arXiv:2405.19101.
- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. EMNLP.
- Plaut, D. C. (1995). Double dissociation without modularity: Evidence from connectionist neuropsychology. *Journal of Clinical and Experimental Neuropsychology*, 17(2), 291–321.
