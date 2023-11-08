[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 4
[]

[Variables]
  [v]
    type = MooseVariableFVReal
  []
  [lambda]
    type = MooseVariableScalar
  []
[]

[FVKernels]
  [diff]
    type = FVDiffusion
    variable = v
    coeff = coeff
  []
  [average]
    type = FVPointValueConstraint
    variable = v
    phi0 = 13
    lambda = lambda
    point = '0.3 0 0'
  []
[]

[FVBCs]
  [left]
    type = FVDirichletBC
    variable = v
    boundary = left
    value = 7
  []
[]

[Materials]
  [diff]
    type = ADGenericFunctorMaterial
    prop_names = 'coeff'
    prop_values = '1'
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'
[]

[Outputs]
  exodus = true
[]
