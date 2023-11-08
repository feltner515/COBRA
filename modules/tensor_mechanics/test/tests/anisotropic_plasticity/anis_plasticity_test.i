[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 1
  ny = 1
  nz = 1
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
  volumetric_locking_correction = true
[]

[AuxVariables]
  [hydrostatic_stress]
    order = CONSTANT
    family = MONOMIAL
  []
  [plasticity_strain_xx]
    order = CONSTANT
    family = MONOMIAL
  []
  [plasticity_strain_xy]
    order = CONSTANT
    family = MONOMIAL
  []
  [plasticity_strain_yy]
    order = CONSTANT
    family = MONOMIAL
  []
[]
[Variables]
  [disp_x]
    scaling = 1e-10
  []
  [disp_y]
    scaling = 1e-10
  []
  [disp_z]
    scaling = 1e-10
  []
[]
[AuxKernels]
  [hydrostatic_stress]
    type = ADRankTwoScalarAux
    variable = hydrostatic_stress
    rank_two_tensor = stress
    scalar_type = Hydrostatic
  []
  [plasticity_strain_xx]
    type = ADRankTwoAux
    rank_two_tensor = trial_plasticity_plastic_strain
    variable = plasticity_strain_xx
    index_i = 0
    index_j = 0
  []
  [plasticity_strain_xy]
    type = ADRankTwoAux
    rank_two_tensor = trial_plasticity_plastic_strain
    variable = plasticity_strain_xy
    index_i = 0
    index_j = 1
  []
  [plasticity_strain_yy]
    type = ADRankTwoAux
    rank_two_tensor = trial_plasticity_plastic_strain
    variable = plasticity_strain_yy
    index_i = 1
    index_j = 1
  []
[]

[Functions]
  [pull]
    type = PiecewiseLinear
    x = '0 1e3 1e8'
    y = '0 1e2 1e2'
  []
[]

[Modules/TensorMechanics/Master]
  [all]
    strain = FINITE
    add_variables = true
    incremental = true
    generate_output = 'elastic_strain_xx elastic_strain_yy elastic_strain_xy stress_xx stress_xy stress_yy'
    use_automatic_differentiation = true
  []
[]

[Materials]

  [elasticity_tensor]
    type = ADComputeIsotropicElasticityTensor
    youngs_modulus = 206800
    poissons_ratio = 0.0
  []

  [elastic_strain]
    type = ADComputeMultipleInelasticStress
    inelastic_models = "trial_plasticity"
    max_iterations = 500
    absolute_tolerance = 1e-05
  []
  [hill_tensor]
    type = ADHillConstants
    # F G H L M N
    hill_constants = "1.0 4.0 5.0 0.5 0.5 0.5"
    base_name = trial_plasticity
  []

  [trial_plasticity]
    type = ADHillPlasticityStressUpdate
    # internal_solve_output_on = always
    # F G H L M N
    hardening_constant = 5000
    yield_stress = 20000000000000
    base_name = trial_plasticity
  []
[]

[BCs]
  [no_disp_x]
    type = ADDirichletBC
    variable = disp_x
    boundary = bottom
    value = 0.0
  []

  [no_disp_y]
    type = ADDirichletBC
    variable = disp_y
    boundary = bottom
    value = 0.0
  []

  [no_disp_z]
    type = ADDirichletBC
    variable = disp_z
    boundary = bottom
    value = 0.0
  []

  [Pressure]
    [Side1]
      boundary = top
      function = pull
    []
  []
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient

  solve_type = NEWTON
  petsc_options_iname = '-ksp_gmres_restart -pc_type -sub_pc_type'
  petsc_options_value = '101                asm      lu'

  line_search = 'none'
  nl_rel_tol = 1e-07
  nl_abs_tol = 1.0e-15
  l_max_its = 90
  num_steps = 40
  dt = 5.0e1
  start_time = 0
  automatic_scaling = true
[]

[Postprocessors]
  [max_disp_x]
    type = ElementExtremeValue
    variable = disp_x
  []
  [max_disp_y]
    type = ElementExtremeValue
    variable = disp_y
  []
  [max_hydro]
    type = ElementAverageValue
    variable = hydrostatic_stress
  []
  [dt]
    type = TimestepSize
  []
  [num_lin]
    type = NumLinearIterations
    outputs = console
  []
  [num_nonlin]
    type = NumNonlinearIterations
    outputs = console
  []
[]

[Outputs]
  csv = true
  perf_graph = true
[]
