[Mesh]
  type = GeneratedMesh
  dim = 2
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
  nx = 2
  ny = 2
[]

[Variables]
  [u]
    order = FIRST
    family = LAGRANGE
  []
[]

[AuxVariables]
  [ten]
    order = FIRST
    family = LAGRANGE
    initial_condition = 1
  []
  [2k]
    order = FIRST
    family = LAGRANGE
    initial_condition = 2
  []
[]

[Kernels]
  [all]
    type = MultipleUpdateErrorKernel
    variable = u
    var1 = ten
    var2 = 2k
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = 1
    value = 0
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = 3
    value = 1
  []
[]

[Executioner]
  type = Steady
[]
