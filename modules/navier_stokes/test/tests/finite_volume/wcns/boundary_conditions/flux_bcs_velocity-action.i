l = 10

# Artificial fluid properties
# For a real case, use a GeneralFluidFunctorProperties and a viscosity rampdown
# or initialize very well!
k = 1
cp = 1000
mu = 1e2

# Operating conditions
inlet_temp = 300
outlet_pressure = 1e5
inlet_velocity = 0.001

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = 0
    xmax = ${l}
    ymin = 0
    ymax = 1
    nx = 10
    ny = 5
  []
[]

[FluidProperties]
  [fp]
    type = FlibeFluidProperties
  []
[]

[Modules]
  [NavierStokesFV]
    compressibility = 'weakly-compressible'
    add_energy_equation = true
    add_scalar_equation = true
    passive_scalar_names = 'scalar'

    density = 'rho'
    dynamic_viscosity = 'mu'
    thermal_conductivity = 'k'
    specific_heat = 'cp'
    passive_scalar_diffusivity = 1.1

    initial_velocity = '${inlet_velocity} 1e-15 0'
    initial_temperature = '${inlet_temp}'
    initial_pressure = '${outlet_pressure}'
    initial_scalar_variables = 0.1

    inlet_boundaries = 'left'
    momentum_inlet_types = 'flux-velocity'
    flux_inlet_pps = 'inlet_u'
    energy_inlet_types = 'flux-velocity'
    energy_inlet_function = 'inlet_T'
    passive_scalar_inlet_types = 'flux-velocity'
    passive_scalar_inlet_function = 'inlet_scalar_value'

    wall_boundaries = 'top bottom'
    momentum_wall_types = 'noslip noslip'
    energy_wall_types = 'heatflux heatflux'
    energy_wall_function = '0 0'

    outlet_boundaries = 'right'
    momentum_outlet_types = 'fixed-pressure'
    pressure_function = '${outlet_pressure}'

    external_heat_source = 'power_density'
    passive_scalar_source = 2.1

    mass_advection_interpolation = 'average'
    momentum_advection_interpolation = 'average'
    energy_advection_interpolation = 'average'
  []
[]

[Postprocessors]
  [inlet_u]
    type = Receiver
    default = ${inlet_velocity}
  []
  [inlet_T]
    type = Receiver
    default = ${inlet_temp}
  []
  [inlet_scalar_value]
    type = Receiver
    default = 0.2
  []
[]

[AuxVariables]
  [power_density]
    type = MooseVariableFVReal
    initial_condition = 1e4
  []
[]

[Materials]
  [const_functor]
    type = ADGenericFunctorMaterial
    prop_names = 'cp k mu'
    prop_values = '${cp} ${k} ${mu}'
  []
  [rho]
    type = RhoFromPTFunctorMaterial
    fp = fp
    temperature = T_fluid
    pressure = pressure
  []
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu       NONZERO'

  [TimeStepper]
    type = IterationAdaptiveDT
    dt = 1e-2
    optimal_iterations = 6
  []
  end_time = 1

  nl_abs_tol = 1e-9
  nl_max_its = 50
  line_search = 'none'

  automatic_scaling = true
[]

[Outputs]
  exodus = true
  execute_on = FINAL
[]
