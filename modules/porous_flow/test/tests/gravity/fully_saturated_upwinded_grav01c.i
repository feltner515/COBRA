# Checking that gravity head is established
# 1phase, 2-component, constant fluid-bulk, constant viscosity, constant permeability
# fully saturated with fully-saturated Kernel with upwinding
# For better agreement with the analytical solution (ana_pp), just increase nx
# NOTE: this test is numerically delicate because the steady-state configuration is independent of the mass fraction, so the frac variable can assume any value as long as mass-fraction is conserved

[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 100
  xmin = -1
  xmax = 0
[]

[GlobalParams]
  PorousFlowDictator = dictator
[]

[Variables]
  [pp]
    [InitialCondition]
      type = RandomIC
      min = 0
      max = 1
    []
  []
  [frac]
    [InitialCondition]
      type = RandomIC
      min = 0
      max = 1
    []
  []
[]

[Kernels]
  [flux1]
    type = PorousFlowFullySaturatedAdvectiveFlux
    variable = pp
    fluid_component = 1
    gravity = '-1 0 0'
  []
  [flux0]
    type = PorousFlowFullySaturatedAdvectiveFlux
    variable = frac
    fluid_component = 0
    gravity = '-1 0 0'
  []
[]

[Functions]
  [ana_pp]
    type = ParsedFunction
    symbol_names = 'g B p0 rho0'
    symbol_values = '1 1.2 0 1'
    expression = '-B*log(exp(-p0/B)+g*rho0*x/B)' # expected pp at base
  []
[]

[BCs]
  [z]
    type = DirichletBC
    variable = pp
    boundary = right
    value = 0
  []
[]

[UserObjects]
  [dictator]
    type = PorousFlowDictator
    porous_flow_vars = 'pp frac'
    number_fluid_phases = 1
    number_fluid_components = 2
  []
[]

[FluidProperties]
  [simple_fluid]
    type = SimpleFluidProperties
    bulk_modulus = 1.2
    density0 = 1
    viscosity = 1
    thermal_expansion = 0
  []
[]

[Materials]
  [temperature]
    type = PorousFlowTemperature
  []
  [ppss]
    type = PorousFlow1PhaseFullySaturated
    porepressure = pp
  []
  [massfrac]
    type = PorousFlowMassFraction
    mass_fraction_vars = frac
  []
  [simple_fluid]
    type = PorousFlowSingleComponentFluid
    fp = simple_fluid
    phase = 0
  []
  [permeability]
    type = PorousFlowPermeabilityConst
    permeability = '1 0 0  0 2 0  0 0 3'
  []
[]

[Postprocessors]
  [pp_base]
    type = PointValue
    variable = pp
    point = '-1 0 0'
  []
  [pp_analytical]
    type = FunctionValuePostprocessor
    function = ana_pp
    point = '-1 0 0'
  []
[]

[Preconditioning]
  [andy]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady
  solve_type = Newton
  nl_rel_tol = 1E-12
  petsc_options_iname = '-pc_factor_shift_type'
  petsc_options_value = 'NONZERO'
  nl_max_its = 100
[]

[Outputs]
  csv = true
[]
