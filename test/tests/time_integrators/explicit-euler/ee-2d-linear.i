[Mesh]
  type = GeneratedMesh
  dim = 2
  xmin = -1
  xmax = 1
  ymin = -1
  ymax = 1
  nx = 10
  ny = 10
  elem_type = QUAD4
[]

[Functions]
  [./ic]
    type = ParsedFunction
    expression = 0
  [../]

  [./forcing_fn]
    type = ParsedFunction
    expression = (x+y)
  [../]

  [./exact_fn]
    type = ParsedFunction
    expression = t*(x+y)
  [../]
[]

[Variables]
  [./u]
    order = FIRST
    family = LAGRANGE

    [./InitialCondition]
      type = FunctionIC
      function = ic
    [../]
  [../]
[]

[Kernels]
  [./ie]
    type = TimeDerivative
    variable = u
    lumping = true
    implicit = true
  [../]

  [./diff]
    type = Diffusion
    variable = u
    implicit = false
  [../]

  [./ffn]
    type = BodyForce
    variable = u
    function = forcing_fn
    implicit = false
  [../]
[]

[BCs]
  active = 'all'

  [./all]
    type = FunctionDirichletBC
    variable = u
    preset = false
    boundary = '0 1 2 3'
    function = exact_fn
    implicit = true
  [../]
[]

[Postprocessors]
  [./l2_err]
    type = ElementL2Error
    variable = u
    function = exact_fn
  [../]
[]

[Executioner]
  type = Transient
  scheme = 'explicit-euler'
  solve_type = 'LINEAR'

  start_time = 0.0
  num_steps = 20
  dt = 0.00005
[]

[Outputs]
  exodus = true
  [./console]
    type = Console
    max_rows = 10
  [../]
[]
