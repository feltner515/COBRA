# 2D with mixed conditions on stress/strain

[GlobalParams]
  displacements = 'disp_x disp_y'
  large_kinematics = false
  macro_gradient = hvar
  homogenization_constraint = homogenization
[]

[Mesh]
  [base]
    type = FileMeshGenerator
    file = '2d.exo'
  []

  [sidesets]
    type = SideSetsFromNormalsGenerator
    input = base
    normals = '-1 0 0
                1 0 0
                0 -1 0
                0 1 0'
    fixed_normal = true
    new_boundary = 'left right bottom top'
  []
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [hvar]
    family = SCALAR
    order = THIRD
  []
[]

[AuxVariables]
  [sxx]
    family = MONOMIAL
    order = CONSTANT
  []
  [syy]
    family = MONOMIAL
    order = CONSTANT
  []
  [sxy]
    family = MONOMIAL
    order = CONSTANT
  []
  [exx]
    family = MONOMIAL
    order = CONSTANT
  []
  [eyy]
    family = MONOMIAL
    order = CONSTANT
  []
  [exy]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [sxx]
    type = RankTwoAux
    variable = sxx
    rank_two_tensor = pk1_stress
    index_i = 0
    index_j = 0
  []
  [syy]
    type = RankTwoAux
    variable = syy
    rank_two_tensor = pk1_stress
    index_i = 1
    index_j = 1
  []
  [sxy]
    type = RankTwoAux
    variable = sxy
    rank_two_tensor = pk1_stress
    index_i = 0
    index_j = 1
  []
  [exx]
    type = RankTwoAux
    variable = exx
    rank_two_tensor = mechanical_strain
    index_i = 0
    index_j = 0
  []
  [eyy]
    type = RankTwoAux
    variable = eyy
    rank_two_tensor = mechanical_strain
    index_i = 1
    index_j = 1
  []
  [exy]
    type = RankTwoAux
    variable = exy
    rank_two_tensor = mechanical_strain
    index_i = 0
    index_j = 1
  []
[]

[UserObjects]
  [homogenization]
    type = HomogenizationConstraint
    constraint_types = ${constraint_types}
    targets = ${targets}
    execute_on = 'INITIAL LINEAR NONLINEAR'
  []
[]

[Kernels]
  [sdx]
    type = HomogenizedTotalLagrangianStressDivergence
    variable = disp_x
    component = 0
  []
  [sdy]
    type = HomogenizedTotalLagrangianStressDivergence
    variable = disp_y
    component = 1
  []
[]

[ScalarKernels]
  [enforce]
    type = HomogenizationConstraintScalarKernel
    variable = hvar
  []
[]

[Functions]
  [strain11]
    type = ParsedFunction
    expression = '4.0e-2*t'
  []
  [strain22]
    type = ParsedFunction
    expression = '-2.0e-2*t'
  []
  [strain12]
    type = ParsedFunction
    expression = '1.0e-2*t'
  []
  [stress11]
    type = ParsedFunction
    expression = '400*t'
  []
  [stress22]
    type = ParsedFunction
    expression = '-200*t'
  []
  [stress12]
    type = ParsedFunction
    expression = '100*t'
  []
[]

[BCs]
  [Periodic]
    [x]
      variable = disp_x
      auto_direction = 'x y'
    []
    [y]
      variable = disp_y
      auto_direction = 'x y'
    []
  []

  [fix1_x]
    type = DirichletBC
    boundary = "fix1"
    variable = disp_x
    value = 0
  []
  [fix1_y]
    type = DirichletBC
    boundary = "fix1"
    variable = disp_y
    value = 0
  []

  [fix2_y]
    type = DirichletBC
    boundary = "fix2"
    variable = disp_y
    value = 0
  []
[]

[Materials]
  [elastic_tensor_1]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 100000.0
    poissons_ratio = 0.3
    block = '1'
  []
  [elastic_tensor_2]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 120000.0
    poissons_ratio = 0.21
    block = '2'
  []
  [elastic_tensor_3]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 80000.0
    poissons_ratio = 0.4
    block = '3'
  []
  [compute_stress]
    type = ComputeLagrangianLinearElasticStress
  []
  [compute_strain]
    type = ComputeLagrangianStrain
    homogenization_gradient_names = 'homogenization_gradient'
  []
  [compute_homogenization_gradient]
    type = ComputeHomogenizedLagrangianStrain
  []
[]

[Postprocessors]
  [sxx]
    type = ElementAverageValue
    variable = sxx
    execute_on = 'initial timestep_end'
  []
  [syy]
    type = ElementAverageValue
    variable = syy
    execute_on = 'initial timestep_end'
  []
  [sxy]
    type = ElementAverageValue
    variable = sxy
    execute_on = 'initial timestep_end'
  []
  [exx]
    type = ElementAverageValue
    variable = exx
    execute_on = 'initial timestep_end'
  []
  [eyy]
    type = ElementAverageValue
    variable = eyy
    execute_on = 'initial timestep_end'
  []
  [exy]
    type = ElementAverageValue
    variable = exy
    execute_on = 'initial timestep_end'
  []
[]

[Executioner]
  type = Transient

  solve_type = 'newton'
  line_search = none

  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  l_max_its = 2
  l_tol = 1e-14
  nl_max_its = 30
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-10

  start_time = 0.0
  dt = 0.2
  dtmin = 0.2
  end_time = 1.0
[]

[Outputs]
  csv = true
[]
