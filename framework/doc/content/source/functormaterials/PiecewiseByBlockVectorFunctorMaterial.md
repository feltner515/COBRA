# PiecewiseByBlockVectorFunctorMaterial ADPiecewiseByBlockVectorFunctorMaterial

!syntax description /FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial

## Overview

This object is useful for providing a material property vector value that is discontinuous from
subdomain to subdomain. [!param](/FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial/prop_name) is
required to specify the name of the material vector property. The map parameter
[!param](/FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial/subdomain_to_prop_value)
is used for specifying the property vector value on a subdomain name basis; the first member of each pair should
be a subdomain name while the second member should be a vector functor.

!alert note
ADPiecewiseByBlockVectorFunctorMaterial is the version of this object with automatic differentiation.
AD vector functors must be specified as the values on each block.

!syntax parameters /FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial

!syntax inputs /FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial

!syntax children /FunctorMaterials/PiecewiseByBlockVectorFunctorMaterial
