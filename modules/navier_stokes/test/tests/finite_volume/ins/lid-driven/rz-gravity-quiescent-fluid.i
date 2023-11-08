mu = .01
rho = 1

[GlobalParams]
  velocity_interp_method = 'rc'
  advected_interp_method = 'average'
  two_term_boundary_expansion = true
  rhie_chow_user_object = 'rc'
[]

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = 1
    xmax = 2
    ymin = 0
    ymax = 1
    nx = 10
    ny = 10
  []
[]

[Problem]
  coord_type = 'RZ'
[]

[Variables]
  [u]
    type = INSFVVelocityVariable
  []
  [v]
    type = INSFVVelocityVariable
  []
  [pressure]
    type = INSFVPressureVariable
  []
  [lambda]
    family = SCALAR
    order = FIRST
  []
[]

[AuxVariables]
  [U]
    order = CONSTANT
    family = MONOMIAL
    fv = true
  []
[]

[AuxKernels]
  [mag]
    type = VectorMagnitudeAux
    variable = U
    x = u
    y = v
  []
[]

[UserObjects]
  [rc]
    type = INSFVRhieChowInterpolator
    u = u
    v = v
    pressure = pressure
  []
[]

[FVKernels]
  [mass]
    type = INSFVMassAdvection
    variable = pressure
    rho = ${rho}
  []
  [mean_zero_pressure]
    type = FVIntegralValueConstraint
    variable = pressure
    lambda = lambda
  []

  [u_advection]
    type = INSFVMomentumAdvection
    variable = u
    rho = ${rho}
    momentum_component = 'x'
  []

  [u_viscosity]
    type = INSFVMomentumDiffusion
    variable = u
    mu = 'mu'
    momentum_component = 'x'
  []

  [u_pressure]
    type = INSFVMomentumPressure
    variable = u
    momentum_component = 'x'
    pressure = pressure
  []

  [u_gravity]
    type = INSFVMomentumGravity
    variable = u
    momentum_component = 'x'
    rho = ${rho}
    gravity = '0 -1 0'
  []

  [v_advection]
    type = INSFVMomentumAdvection
    variable = v
    rho = ${rho}
    momentum_component = 'y'
  []

  [v_viscosity]
    type = INSFVMomentumDiffusion
    variable = v
    mu = 'mu'
    momentum_component = 'y'
  []

  [v_pressure]
    type = INSFVMomentumPressure
    variable = v
    momentum_component = 'y'
    pressure = pressure
  []

  [v_gravity]
    type = INSFVMomentumGravity
    variable = v
    momentum_component = 'y'
    rho = ${rho}
    gravity = '0 -1 0'
  []
[]

[FVBCs]
  [free_slip_x]
    type = INSFVNaturalFreeSlipBC
    variable = u
    boundary = 'left right top bottom'
    momentum_component = 'x'
  []

  [free_slip_y]
    type = INSFVNaturalFreeSlipBC
    variable = v
    boundary = 'left right top bottom'
    momentum_component = 'y'
  []
[]

[Materials]
  [mu]
    type = ADGenericFunctorMaterial
    prop_names = 'mu'
    prop_values = '${mu}'
  []
[]

[Executioner]
  type = Steady
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'
  nl_rel_tol = 1e-12
[]

[Outputs]
  exodus = true
[]
