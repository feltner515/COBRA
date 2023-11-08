# apply a sink flux with use_mobility=true and observe the correct behavior
[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1
  ny = 1
  nz = 1
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
  zmin = 0
  zmax = 2
[]

[GlobalParams]
  PorousFlowDictator = dictator
[]

[UserObjects]
  [dictator]
    type = PorousFlowDictator
    porous_flow_vars = 'pp'
    number_fluid_phases = 1
    number_fluid_components = 1
  []
  [pc]
    type = PorousFlowCapillaryPressureVG
    m = 0.5
    alpha = 1
  []
[]

[Variables]
  [pp]
  []
[]

[ICs]
  [pp]
    type = FunctionIC
    variable = pp
    function = y+1
  []
[]

[Kernels]
  [mass0]
    type = PorousFlowMassTimeDerivative
    fluid_component = 0
    variable = pp
  []
[]

[FluidProperties]
  [simple_fluid]
    type = SimpleFluidProperties
    bulk_modulus = 1.3
    density0 = 1.1
    thermal_expansion = 0
    viscosity = 1.1
  []
[]

[Materials]
  [temperature]
    type = PorousFlowTemperature
  []
  [ppss]
    type = PorousFlow1PhaseP
    porepressure = pp
    capillary_pressure = pc
  []
  [massfrac]
    type = PorousFlowMassFraction
  []
  [simple_fluid]
    type = PorousFlowSingleComponentFluid
    fp = simple_fluid
    phase = 0
  []
  [porosity]
    type = PorousFlowPorosityConst
    porosity = 0.1
  []
  [permeability]
    type = PorousFlowPermeabilityConst
    permeability = '0.2 0 0 0 0.1 0 0 0 0.1'
  []
  [relperm]
    type = PorousFlowRelativePermeabilityCorey
    n = 2
    phase = 0
  []
[]

[AuxVariables]
  [flux_out]
  []
  [xval]
  []
  [yval]
  []
[]

[ICs]
  [xval]
    type = FunctionIC
    variable = xval
    function = x
  []
  [yval]
    type = FunctionIC
    variable = yval
    function = y
  []
[]

[Functions]
  [mass00]
    type = ParsedFunction
    expression = 'vol*por*dens0*exp(pp/bulk)'
    symbol_names = 'vol por dens0 pp bulk'
    symbol_values = '0.25 0.1 1.1 p00 1.3'
  []
  [mass01]
    type = ParsedFunction
    expression = 'vol*por*dens0*exp(pp/bulk)'
    symbol_names = 'vol por dens0 pp bulk'
    symbol_values = '0.25 0.1 1.1 p01 1.3'
  []
  [expected_mass_change00]
    type = ParsedFunction
    expression = 'fcn*perm*dens0*exp(pp/bulk)/visc*area*dt'
    symbol_names = 'fcn perm dens0 pp bulk visc area dt'
    symbol_values = '6   0.2  1.1  p00 1.3  1.1  0.5  1E-3'
  []
  [expected_mass_change01]
    type = ParsedFunction
    expression = 'fcn*perm*dens0*exp(pp/bulk)/visc*area*dt'
    symbol_names = 'fcn perm dens0 pp bulk visc area dt'
    symbol_values = '6   0.2  1.1  p01 1.3  1.1  0.5  1E-3'
  []
  [mass00_expect]
    type = ParsedFunction
    expression = 'mass_prev-mass_change'
    symbol_names = 'mass_prev mass_change'
    symbol_values = 'm00_prev  del_m00'
  []
  [mass01_expect]
    type = ParsedFunction
    expression = 'mass_prev-mass_change'
    symbol_names = 'mass_prev mass_change'
    symbol_values = 'm01_prev  del_m01'
  []
[]

[Postprocessors]
  [p00]
    type = PointValue
    point = '0 0 0'
    variable = pp
    execute_on = 'initial timestep_end'
  []
  [m00]
    type = FunctionValuePostprocessor
    function = mass00
    execute_on = 'initial timestep_end'
  []
  [m00_prev]
    type = FunctionValuePostprocessor
    function = mass00
    execute_on = 'timestep_begin'
    outputs = 'console'
  []
  [del_m00]
    type = FunctionValuePostprocessor
    function = expected_mass_change00
    execute_on = 'timestep_end'
    outputs = 'console'
  []
  [m00_expect]
    type = FunctionValuePostprocessor
    function = mass00_expect
    execute_on = 'timestep_end'
  []
  [p10]
    type = PointValue
    point = '1 0 0'
    variable = pp
    execute_on = 'initial timestep_end'
  []
  [p01]
    type = PointValue
    point = '0 1 0'
    variable = pp
    execute_on = 'initial timestep_end'
  []
  [m01]
    type = FunctionValuePostprocessor
    function = mass01
    execute_on = 'initial timestep_end'
  []
  [m01_prev]
    type = FunctionValuePostprocessor
    function = mass01
    execute_on = 'timestep_begin'
    outputs = 'console'
  []
  [del_m01]
    type = FunctionValuePostprocessor
    function = expected_mass_change01
    execute_on = 'timestep_end'
    outputs = 'console'
  []
  [m01_expect]
    type = FunctionValuePostprocessor
    function = mass01_expect
    execute_on = 'timestep_end'
  []
  [p11]
    type = PointValue
    point = '1 1 0'
    variable = pp
    execute_on = 'initial timestep_end'
  []
[]

[BCs]
  [flux]
    type = PorousFlowSink
    boundary = 'left'
    variable = pp
    use_mobility = true
    use_relperm = true
    fluid_phase = 0
    flux_function = 6
    save_in = flux_out
  []
[]

[Preconditioning]
  [andy]
    type = SMP
    full = true
    petsc_options_iname = '-ksp_type -pc_type -sub_pc_type -snes_max_it -sub_pc_factor_shift_type -pc_asm_overlap'
    petsc_options_value = 'gmres asm lu 10000 NONZERO 2'
  []
[]

[Executioner]
  type = Transient
  solve_type = Newton
  dt = 1E-3
  end_time = 0.03
  nl_rel_tol = 1E-12
  nl_abs_tol = 1E-12
[]

[Outputs]
  file_base = s02
  [console]
    type = Console
    execute_on = 'nonlinear linear'
    interval = 30
  []
  [csv]
    type = CSV
    execute_on = 'timestep_end'
    interval = 3
  []
[]
