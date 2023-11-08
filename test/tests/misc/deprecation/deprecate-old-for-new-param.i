[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 10
  ny = 10
[]

[Variables]
  [u]
  []
[]

[Functions]
  [diff_func_x]
    type = ParsedFunction
    expression = 1/t
  []
  [diff_func_y]
    type = ParsedFunction
    expression = 't*t + x'
  []
[]

[Kernels]
  [diff]
    type = VectorMatDiffusion
    variable = u
    coef = diffusion
  []
  [td]
    type = TimeDerivative
    variable = u
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = '0'
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = '1'
  []
[]

[Materials]
  [gfm]
    type = GenericFunctionVectorMaterial
    block = 0
    prop_names = diffusion
    prop_values = 'diff_func_x diff_func_y 0'
  []
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = 0.1

  solve_type = 'PJFNK'

  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
[]

[Outputs]
  exodus = true
[]
