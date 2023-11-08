# PorousFlowSink

!syntax description /BCs/PorousFlowSink

This sink is
\begin{equation*}
s = f(t, x) \ ,
\end{equation*}
where $f$ is a MOOSE Function of time and position on the boundary.

If $f>0$ then the boundary condition will act as a sink, while if $f<0$ the boundary condition acts as a source.  If applied to a fluid-component equation, the function $f$ has units kg.m$^{-2}$.s$^{-1}$.  If applied to the heat equation, the function $f$ has units J.m$^{-2}$.s$^{-1}$.  These units are potentially modified if the extra building blocks enumerated below are used.

In addition, the sink may be multiplied by any or all of the following
quantities through the `optional parameters` list.

- Fluid relative permeability
- Fluid mobility ($k_{ij}n_{i}n_{j}k_{r} \rho / \nu$, where $n$ is the normal vector to the boundary)
- Fluid mass fraction
- Fluid internal energy
- Thermal conductivity

See [boundary conditions](boundaries.md) for many more details and discussion.

!syntax parameters /BCs/PorousFlowSink

!syntax inputs /BCs/PorousFlowSink

!syntax children /BCs/PorousFlowSink
