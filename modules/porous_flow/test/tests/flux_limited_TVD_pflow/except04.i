# Exception test: fe_order specified but not fe_family
[Mesh]
  type = GeneratedMesh
  dim = 1
[]

[GlobalParams]
  gravity = '1 2 3'
  PorousFlowDictator = dictator
[]

[Variables]
  [pp]
  []
  [tracer]
  []
[]

[FluidProperties]
  [the_simple_fluid]
    type = SimpleFluidProperties
  []
[]

[PorousFlowUnsaturated]
  porepressure = pp
  mass_fraction_vars = tracer
  fp = the_simple_fluid
[]

[UserObjects]
  [advective_flux_calculator]
    type = PorousFlowAdvectiveFluxCalculatorSaturated
    fe_order = First
  []
[]

[Materials]
  [permeability]
    type = PorousFlowPermeabilityConst
    permeability = '1 0 0  0 2 0  0 0 3'
  []
[]

[Executioner]
  type = Steady
  solve_type = Newton
[]
