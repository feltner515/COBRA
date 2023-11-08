# ADDirichletBC

!syntax description /BCs/ADDirichletBC

## Description

`ADDirichletBC` is the simplest type of `NodalBC`, and is used for
imposing so-called "essential" boundary conditions on systems of
partial differential equations (PDEs).  Such boundary conditions force
a particular set of degrees of freedom (DOFs) defined by the
`boundary` parameter to take on a single, controllable value. This
class is appropriate to use for PDEs of the form

\begin{equation}
\begin{aligned}
  -\nabla^2 u &= f && \quad \in \Omega \\
  u &= g && \quad \in \partial \Omega_D \\
  \frac{\partial u}{\partial n} &= h && \quad \in \partial \Omega_N,
\end{aligned}
\end{equation}

where $\Omega \subset \mathbb{R}^n$ is the domain, and $\partial
\Omega = \partial \Omega_D \cup \partial \Omega_N$ is its boundary. In
this case, a `ADDirichletBC` object is used to impose the condition (2)
on the subset of the boundary denoted by $\partial \Omega_D$. In this case, the
`value` corresponds to the constant $g$, and the user must define one
or more sidesets corresponding to the boundary subset $\partial \Omega_D$.

Note that `ADDirichletBC` computes its Jacobian using automatic differentiation.

## Preset boundary conditions

With the parameter `preset = true`, the value of the boundary condition is applied
before the solve begins. With `preset = false`, the boundary condition is
only enforced as the solve progresses. In most situations, presetting the boundary
condition is better.

## Example Input Syntax

!listing test/tests/bcs/ad_bc_preset_nodal/bc_preset_nodal.i block=BCs

!syntax parameters /BCs/ADDirichletBC

!syntax inputs /BCs/ADDirichletBC

!syntax children /BCs/ADDirichletBC
