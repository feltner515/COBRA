[Materials]
  [mat]
    type = ADGenericConstantMaterial
    block = 'blk:0'
    prop_names = 'density specific_heat thermal_conductivity'
    prop_values = '1000 100 30'
  []
[]

[FluidProperties]
  [fp]
    type = StiffenedGasFluidProperties
    gamma = 2.35
    cv = 1816.0
    q = -1.167e6
    p_inf = 1.0e9
    q_prime = 0
  []
[]

[Closures]
  [simple_closures]
    type = Closures1PhaseSimple
  []
[]

[Functions]
  [T_init]
    type = ParsedFunction
    expression = '1000*y+300+30*z'
  []
[]
[GlobalParams]
  scaling_factor_1phase = '1 1 1e-3'
  gravity_vector = '0 0 0'
[]
[Components]
  [fch]
    type = FlowChannel1Phase
    position = '0 0 0'
    orientation = '0 0 1'
    fp = fp
    n_elems = 6
    length = 1
    initial_T = T_init
    initial_p = 1.01e5
    initial_vel = 0
    closures = simple_closures
    A   = 0.00314159
    D_h  = 0.2
    f = 0.01
  []
  [in]
    type = InletVelocityTemperature1Phase
    input = 'fch:in'
    vel = 1
    T = 300
  []
  [out]
    type = Outlet1Phase
    input = 'fch:out'
    p = 1.01e5
  []
  [blk]
    type = HeatStructureFromFile3D
    file = mesh.e
    position = '0 0 0'
    initial_T = T_init
  []
  [ht]
    type = HeatTransferFromHeatStructure3D1Phase
    flow_channels = 'fch'
    hs = blk
    boundary = blk:rmin
    Hw = 10000
    P_hf = 0.1564344650402309
  []
[]

[Postprocessors]
  [energy_hs]
    type = ADHeatStructureEnergy3D
    block = blk:0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [energy_fch]
    type = ElementIntegralVariablePostprocessor
    block = fch
    variable = rhoEA
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [total_energy]
    type = SumPostprocessor
    values = 'energy_fch energy_hs'
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [energy_change]
    type = ChangeOverTimePostprocessor
    change_with_respect_to_initial = true
    postprocessor = total_energy
    compute_relative_change = true
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]
[Preconditioning]
  [pc]
    type = SMP
    full = true
    petsc_options_iname = '-snes_test_err'
    petsc_options_value = ' 1e-9'
  []
[]
[Executioner]
  type = Transient
  scheme = bdf2
  dt = 0.1
  num_steps = 1

  solve_type = PJFNK
  line_search = basic
  abort_on_solve_fail = true
  nl_abs_tol = 1e-8
[]

[Outputs]
  file_base = 'phy.conservation'
  csv = true
  show = 'energy_change'
  execute_on = 'final'
[]
