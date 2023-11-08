[Mesh]
  [./gmg]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 10
    ny = 10
    nz = 10
  []

  [./createNewSidesetOne]
    type = SideSetsFromBoundingBoxGenerator
    input = gmg
    boundaries_old = 'bottom top'
    boundary_new = 11
    bottom_left = '-1.1 -1.1 -1.1'
    top_right = '1.1 1.1 1.1'
    block_id = 0
    boundary_id_overlap = true
  [../]
[]

[Variables]
  [./u]
  [../]
[]

[Kernels]
  [./diff]
    type = Diffusion
    variable = u
  [../]
[]

[BCs]
  [./BCone]
    type = DirichletBC
    variable = u
    boundary = 'left'
    value = 1
  [../]
  [./BCtwo]
    type = DirichletBC
    variable = u
    boundary = 11
    value = 0
  [../]
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
[]

[Outputs]
  exodus = true
[]
