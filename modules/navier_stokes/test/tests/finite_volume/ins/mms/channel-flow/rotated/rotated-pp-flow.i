mu=0.5
rho=1.1
advected_interp_method='average'
velocity_interp_method='average'
two_term_boundary_expansion=true

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = 0
    xmax = 10
    ymin = -1
    ymax = 1
    nx = 10
    ny = 2
  []
  [rotate]
    type = TransformGenerator
    input = gen
    transform = 'rotate'
    vector_value = '45 0 0'
  []
[]

[GlobalParams]
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = INSFVRhieChowInterpolator
    u = u
    v = v
    pressure = pressure
  []
[]

[AuxVariables]
  [vel_exact_x][]
  [vel_exact_y][]
  [p_exact][]
[]

[AuxKernels]
  [u_exact]
    type = FunctionAux
    variable = vel_exact_x
    function = exact_u
  []
  [v_exact]
    type = FunctionAux
    variable = vel_exact_y
    function = exact_v
  []
  [p_exact]
    type = FunctionAux
    variable = p_exact
    function = exact_p
  []
[]

[Variables]
  [u]
    type = INSFVVelocityVariable
    initial_condition = 1
    two_term_boundary_expansion = ${two_term_boundary_expansion}
  []
  [v]
    type = INSFVVelocityVariable
    initial_condition = 1
    two_term_boundary_expansion = ${two_term_boundary_expansion}
  []
  [pressure]
    type = INSFVPressureVariable
    two_term_boundary_expansion = ${two_term_boundary_expansion}
  []
[]

[FVKernels]
  [mass]
    type = INSFVMassAdvection
    variable = pressure
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []
  [mass_forcing]
    type = FVBodyForce
    variable = pressure
    function = forcing_p
  []

  [u_advection]
    type = INSFVMomentumAdvection
    variable = u
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    momentum_component = 'x'
  []
  [u_viscosity]
    type = INSFVMomentumDiffusion
    variable = u
    mu = ${mu}
    momentum_component = 'x'
  []
  [u_pressure]
    type = INSFVMomentumPressure
    variable = u
    momentum_component = 'x'
    pressure = pressure
  []
  [u_forcing]
    type = INSFVBodyForce
    variable = u
    functor = forcing_u
    momentum_component = 'x'
  []

  [v_advection]
    type = INSFVMomentumAdvection
    variable = v
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    momentum_component = 'y'
  []
  [v_viscosity]
    type = INSFVMomentumDiffusion
    variable = v
    mu = ${mu}
    momentum_component = 'y'
  []
  [v_pressure]
    type = INSFVMomentumPressure
    variable = v
    momentum_component = 'y'
    pressure = pressure
  []
  [v_forcing]
    type = INSFVBodyForce
    variable = v
    functor = forcing_v
    momentum_component = 'y'
  []
[]

[FVBCs]
  [inlet-u]
    type = INSFVInletVelocityBC
    boundary = 'left'
    variable = u
    function = 'exact_u'
  []
  [inlet-v]
    type = INSFVInletVelocityBC
    boundary = 'left'
    variable = v
    function = 'exact_v'
  []
  [walls-u]
    type = INSFVNoSlipWallBC
    variable = u
    boundary = 'top bottom'
    function = 'exact_u'
  []
  [walls-v]
    type = INSFVNoSlipWallBC
    variable = v
    boundary = 'top bottom'
    function = 'exact_v'
  []
  [outlet_p]
    type = INSFVOutletPressureBC
    boundary = 'right'
    variable = pressure
    function = 'exact_p'
  []
[]

[Functions]
  [exact_u]
    type = ParsedFunction
    expression = '0.25*sqrt(2)*(1.0 - 1/2*(-x + y)^2)/mu'
    symbol_names = 'mu'
    symbol_values = '${mu}'
  []
  [exact_rhou]
    type = ParsedFunction
    expression = '0.25*sqrt(2)*rho*(1.0 - 1/2*(-x + y)^2)/mu'
    symbol_names = 'mu rho'
    symbol_values = '${mu} ${rho}'
  []
  [forcing_u]
    type = ParsedFunction
    expression = '0'
  []
  [exact_v]
    type = ParsedFunction
    expression = '0.25*sqrt(2)*(1.0 - 1/2*(-x + y)^2)/mu'
    symbol_names = 'mu'
    symbol_values = '${mu}'
  []
  [exact_rhov]
    type = ParsedFunction
    expression = '0.25*sqrt(2)*rho*(1.0 - 1/2*(-x + y)^2)/mu'
    symbol_names = 'mu rho'
    symbol_values = '${mu} ${rho}'
  []
  [forcing_v]
    type = ParsedFunction
    expression = '0'
  []
  [exact_p]
    type = ParsedFunction
    expression = '-1/2*sqrt(2)*(x + y) + 10.0'
  []
  [forcing_p]
    type = ParsedFunction
    expression = '0'
  []
[]

[Executioner]
  type = Steady
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -ksp_gmres_restart -sub_pc_type -sub_pc_factor_shift_type'
  petsc_options_value = 'asm      100                lu           NONZERO'
  line_search = 'none'
[]

[Outputs]
  csv = true
  [dof]
    type = DOFMap
    execute_on = 'initial'
  []
[]

[Postprocessors]
  [h]
    type = AverageElementSize
    outputs = 'console csv'
    execute_on = 'timestep_end'
  []
  [./L2u]
    type = ElementL2Error
    variable = u
    function = exact_u
    outputs = 'console csv'
    execute_on = 'timestep_end'
  [../]
  [./L2v]
    type = ElementL2Error
    variable = v
    function = exact_v
    outputs = 'console csv'
    execute_on = 'timestep_end'
  [../]
  [./L2p]
    variable = pressure
    function = exact_p
    type = ElementL2Error
    outputs = 'console csv'
    execute_on = 'timestep_end'
  [../]
[]
