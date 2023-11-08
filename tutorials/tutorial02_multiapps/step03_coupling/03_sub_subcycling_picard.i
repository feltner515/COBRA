[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 10
  ny = 10
[]

[Variables]
  [v]
  []
[]

[AuxVariables]
  [ut]
  []
[]

[Kernels]
  [diff]
    type = Diffusion
    variable = v
  []
  [force]
    type = CoupledForce
    variable = v
    v = ut
    coef = 100
  []
  [td]
    type = TimeDerivative
    variable = v
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = v
    boundary = left
    value = 0
  []
  [right]
    type = DirichletBC
    variable = v
    boundary = right
    value = 1
  []
[]

[Executioner]
  type = Transient
  end_time = 2
  dt = 0.05

  nl_abs_tol = 1e-10

  solve_type = 'PJFNK'

  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
[]

[Outputs]
  exodus = true
[]

[Postprocessors]
  [average_v]
    type = ElementAverageValue
    variable = v
  []
[]
