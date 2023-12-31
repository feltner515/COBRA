[GlobalParams]
  gravity_vector = '0 0 0'

  initial_T = 444.447
  initial_p = 7e6
  initial_vel = 0

  scaling_factor_1phase = '1 1 1e-5'

  closures = simple_closures
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

[Components]
  [pipe]
    type = FlowChannel1Phase
    fp = fp
    # geometry
    position = '0 0 0'
    orientation = '1 0 0'
    A   = 1.0000000000e-04
    f = 0.0
    length = 1
    n_elems = 100
  []

  [in]
    type = InletVelocityTemperature1Phase
    input = 'pipe:in'
    vel = -1.0
    T     = 444.447
  []

  [out]
    type = InletStagnationPressureTemperature1Phase
    input = 'pipe:out'
    p0 = 7e6
    T0 = 444.447
  []
[]

[Preconditioning]
  [SMP_PJFNK]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  scheme = 'bdf2'

  dt = 0.1
  start_time = 0.0
  end_time = 5

  solve_type = 'PJFNK'
  line_search = 'basic'
  nl_rel_tol = 0
  nl_abs_tol = 1e-6
  nl_max_its = 10

  l_tol = 1e-3
  l_max_its = 100

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  abort_on_solve_fail = true
[]

[Outputs]
  [exodus]
    type = Exodus
    file_base = phy.reversed_flow
    show = 'vel T p'
  []
  velocity_as_vector = false
[]
