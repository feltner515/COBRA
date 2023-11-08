a=1.1

[GlobalParams]
  advected_interp_method = 'average'
[]

[Mesh]
  [./gen_mesh]
    type = GeneratedMeshGenerator
    dim = 1
    xmin = 0
    xmax = 1.1
    nx = 2
  [../]
[]

[Variables]
  [./u]
    family = MONOMIAL
    order = CONSTANT
    fv = true
    two_term_boundary_expansion = false
    type = MooseVariableFVReal
  [../]
  [./v]
    family = MONOMIAL
    order = CONSTANT
    fv = true
    two_term_boundary_expansion = true
    type = MooseVariableFVReal
  [../]
[]

[FVKernels]
  [./advection_u]
    type = FVAdvection
    variable = u
    velocity = '${a} 0 0'
  [../]
  [body_u]
    type = FVBodyForce
    variable = u
    function = 'forcing'
  []
  [./advection_v]
    type = FVAdvection
    variable = v
    velocity = '${a} 0 0'
  [../]
  [body_v]
    type = FVBodyForce
    variable = v
    function = 'forcing'
  []
[]

[FVBCs]
  [left_u]
    type = FVFunctionDirichletBC
    boundary = 'left'
    function = 'exact'
    variable = u
  []
  [right_u]
    type = FVConstantScalarOutflowBC
    variable = u
    velocity = '${a} 0 0'
    boundary = 'right'
  []
  [left_v]
    type = FVFunctionDirichletBC
    boundary = 'left'
    function = 'exact'
    variable = v
  []
  [right_v]
    type = FVConstantScalarOutflowBC
    variable = v
    velocity = '${a} 0 0'
    boundary = 'right'
  []
[]

[Functions]
  [exact]
    type = ParsedFunction
    expression = 'cos(x)'
  []
  [forcing]
    type = ParsedFunction
    expression = '-${a} * sin(x)'
  []
[]

[Executioner]
  type = Steady
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_shift_type -pc_factor_mat_solver_type'
  petsc_options_value = 'lu       NONZERO               mumps'
[]

[Outputs]
  exodus = true
  csv = true
[]

[Postprocessors]
  [./L2u]
    type = ElementL2Error
    variable = u
    function = exact
    outputs = 'console csv'
    execute_on = 'timestep_end'
  [../]
  [./L2v]
    type = ElementL2Error
    variable = v
    function = exact
    outputs = 'console csv'
    execute_on = 'timestep_end'
  [../]
  [h]
    type = AverageElementSize
    outputs = 'console csv'
    execute_on = 'timestep_end'
  []
[]
