mu = 1.1
rho = 1
advected_interp_method = 'average'
velocity_interp_method = 'rc'

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = 0
    xmax = 5
    ymin = 0
    ymax = 1
    nx = 40
    ny = 10
  []
[]

[Problem]
  coord_type = 'RZ'
  rz_coord_axis = 'X'
[]

[GlobalParams]
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = PINSFVRhieChowInterpolator
    u = u
    v = v
    pressure = pressure
    porosity = porosity
  []
[]

[Variables]
  [u]
    type = PINSFVSuperficialVelocityVariable
    initial_condition = 1
  []
  [v]
    type = PINSFVSuperficialVelocityVariable
    initial_condition = 1e-6
  []
  [pressure]
    type = INSFVPressureVariable
  []
[]

[AuxVariables]
  [porosity]
    family = MONOMIAL
    order = CONSTANT
    fv = true
    initial_condition = 0.5
  []
[]

[FVKernels]
  inactive = 'v_pressure_volumetric'
  [mass]
    type = PINSFVMassAdvection
    variable = pressure
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []

  [u_advection]
    type = PINSFVMomentumAdvection
    variable = u
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    porosity = porosity
    momentum_component = 'x'
  []
  [u_viscosity]
    type = PINSFVMomentumDiffusion
    variable = u
    mu = ${mu}
    porosity = porosity
    momentum_component = 'x'
  []
  [u_pressure]
    type = PINSFVMomentumPressureFlux
    variable = u
    momentum_component = 'x'
    pressure = pressure
    porosity = porosity
  []
  [u_friction]
    type = PINSFVMomentumFriction
    variable = u
    momentum_component = 'x'
    porosity = porosity
    Darcy_name = 'Darcy_coefficient'
    Forchheimer_name = 'Forchheimer_coefficient'
    rho = ${rho}
  []

  [v_advection]
    type = PINSFVMomentumAdvection
    variable = v
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    porosity = porosity
    momentum_component = 'y'
  []
  [v_viscosity]
    type = PINSFVMomentumDiffusion
    variable = v
    mu = ${mu}
    porosity = porosity
    momentum_component = 'y'
  []
  [v_pressure_volumetric]
    type = PINSFVMomentumPressure
    variable = v
    momentum_component = 'y'
    pressure = pressure
    porosity = porosity
  []
  [v_pressure_by_parts_flux]
    type = PINSFVMomentumPressureFlux
    variable = v
    momentum_component = 'y'
    pressure = pressure
    porosity = porosity
  []
  [v_pressure_by_parts_volume_term]
    type = PNSFVMomentumPressureFluxRZ
    variable = v
    pressure = pressure
    porosity = porosity
    momentum_component = 'y'
  []
  [v_friction]
    type = PINSFVMomentumFriction
    variable = v
    momentum_component = 'y'
    porosity = porosity
    Darcy_name = 'Darcy_coefficient'
    Forchheimer_name = 'Forchheimer_coefficient'
    rho = ${rho}
  []
[]

[FVBCs]
  inactive = 'free-slip-u free-slip-v'
  [inlet-u]
    type = INSFVInletVelocityBC
    boundary = 'left'
    variable = u
    function = '1'
  []
  [inlet-v]
    type = INSFVInletVelocityBC
    boundary = 'left'
    variable = v
    function = 0
  []

  [no-slip-u]
    type = INSFVNoSlipWallBC
    boundary = 'top'
    variable = u
    function = 0
  []
  [no-slip-v]
    type = INSFVNoSlipWallBC
    boundary = 'top'
    variable = v
    function = 0
  []
  [free-slip-u]
    type = INSFVNaturalFreeSlipBC
    boundary = 'top'
    variable = u
    momentum_component = 'x'
  []
  [free-slip-v]
    type = INSFVNaturalFreeSlipBC
    boundary = 'top'
    variable = v
    momentum_component = 'y'
  []
  [symmetry-u]
    type = PINSFVSymmetryVelocityBC
    boundary = 'bottom'
    variable = u
    u = u
    v = v
    mu = ${mu}
    momentum_component = 'x'
  []
  [symmetry-v]
    type = PINSFVSymmetryVelocityBC
    boundary = 'bottom'
    variable = v
    u = u
    v = v
    mu = ${mu}
    momentum_component = 'y'
  []
  [symmetry-p]
    type = INSFVSymmetryPressureBC
    boundary = 'bottom'
    variable = pressure
  []

  [outlet-p]
    type = INSFVOutletPressureBC
    boundary = 'right'
    variable = pressure
    function = 0
  []
[]

[Materials]
  [darcy]
    type = ADGenericVectorFunctorMaterial
    prop_names = 'Darcy_coefficient Forchheimer_coefficient'
    prop_values = '0.1 0.1 0.1 0.1 0.1 0.1'
  []
[]

[Executioner]
  type = Steady
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'
  line_search = 'none'
  nl_rel_tol = 1e-11
  nl_abs_tol = 1e-14
[]

# Some basic Postprocessors to visually examine the solution
[Postprocessors]
  [inlet-p]
    type = SideAverageValue
    variable = pressure
    boundary = 'left'
  []
  [outlet-u]
    type = SideIntegralVariablePostprocessor
    variable = u
    boundary = 'right'
  []
[]

[Outputs]
  exodus = true
[]
