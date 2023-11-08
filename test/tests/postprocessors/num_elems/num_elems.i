[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 4
  ny = 4
  nz = 0
  zmax = 0
  elem_type = QUAD4
  uniform_refine = 1
[]

[Variables]
  [./u]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[AuxVariables]
  [./u_aux]
    order = FIRST
    family = LAGRANGE
  [../]
  [./v_aux]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[Functions]
  [./force]
    type = ParsedFunction
    expression = t
  [../]
[]

[Kernels]
  [./diff]
    type = Diffusion
    variable = u
  [../]
  [./force]
    type = BodyForce
    variable = u
    function = force
  [../]
[]

[BCs]
  [./right]
    type = DirichletBC
    variable = u
    boundary = right
    value = 1
  [../]
  [./left]
    type = DirichletBC
    variable = u
    boundary = left
    value = 0
  [../]
[]

[Executioner]
  type = Transient
  num_steps = 3
  dt = 1
  solve_type = PJFNK
[]

[Adaptivity]
  steps = 1
  marker = box
  max_h_level = 3
  [./Markers]
    [./box]
      bottom_left = '0.3 0.3 0'
      inside = refine
      top_right = '0.6 0.6 0'
      outside = do_nothing
      type = BoxMarker
    [../]
  [../]
[]

[Postprocessors]
  [./num_elems_active]
    type = NumElems
    elem_filter = 'ACTIVE'
    execute_on = 'initial timestep_end'
  [../]
  [./num_elems_total]
    type = NumElems
    elem_filter = 'TOTAL'
    execute_on = 'initial timestep_end'
  [../]
[]

[Outputs]
  csv = true
[]
