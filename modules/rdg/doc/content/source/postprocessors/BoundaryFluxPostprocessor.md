# BoundaryFluxPostprocessor

!syntax description /Postprocessors/BoundaryFluxPostprocessor

This post-processor is used to query flux vectors that are computed via
objects derived from `BoundaryFluxBase`. It computes the side integral of the
specified entry of the flux at a specified boundary.

This post-processor is useful for conservation tests because the inflow and
outflow fluxes for a domain can be integrated. Note that in this case, one still
needs to integrate the post-processor over time; this is done with the
[TimeIntegratedPostprocessor](/TimeIntegratedPostprocessor.md) post-processor.

!syntax parameters /Postprocessors/BoundaryFluxPostprocessor

!syntax inputs /Postprocessors/BoundaryFluxPostprocessor

!syntax children /Postprocessors/BoundaryFluxPostprocessor
