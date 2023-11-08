# INSFVEnergyAdvection

This object adds a $\rho c_p \vec u \cdot \nabla T$ term to a finite volume
formulation of a heat transport equation. The user can control what (material) quantity is
advected through the `advected_quantity` parameter. The default value is the
name `rho_cp_temp` which corresponds to a material property name declared by
[INSFVEnthalpyMaterial.md].

!syntax parameters /FVKernels/INSFVEnergyAdvection

!syntax inputs /FVKernels/INSFVEnergyAdvection

!syntax children /FVKernels/INSFVEnergyAdvection
