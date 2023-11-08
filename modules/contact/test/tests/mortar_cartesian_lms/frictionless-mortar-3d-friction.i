starting_point = 0.25
offset = 0.00

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
  diffusivity = 1e0
  scaling = 1e0
[]

[Mesh]
  second_order = false
  [top_block]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 3
    ny = 3
    nz = 3
    xmin = -0.25
    xmax = 0.25
    ymin = -0.25
    ymax = 0.25
    zmin = -0.25
    zmax = 0.25
    elem_type = HEX8
  []
  [rotate_top_block]
    type = TransformGenerator
    input = top_block
    transform = ROTATE
    vector_value = '0 0 0'
  []
  [top_block_sidesets]
    type = RenameBoundaryGenerator
    input = rotate_top_block
    old_boundary = '0 1 2 3 4 5'
    new_boundary = 'top_bottom top_back top_right top_front top_left top_top'
  []
  [top_block_id]
    type = SubdomainIDGenerator
    input = top_block_sidesets
    subdomain_id = 1
  []
  [bottom_block]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 10
    ny = 10
    nz = 2
    xmin = -.5
    xmax = .5
    ymin = -.5
    ymax = .5
    zmin = -.3
    zmax = -.25
    elem_type = HEX8
  []
  [bottom_block_id]
    type = SubdomainIDGenerator
    input = bottom_block
    subdomain_id = 2
  []
  [bottom_block_change_boundary_id]
    type = RenameBoundaryGenerator
    input = bottom_block_id
    old_boundary = '0 1 2 3 4 5'
    new_boundary = '100 101 102 103 104 105'
  []
  [combined]
    type = MeshCollectionGenerator
    inputs = 'top_block_id bottom_block_change_boundary_id'
  []
  [block_rename]
    type = RenameBlockGenerator
    input = combined
    old_block = '1 2'
    new_block = 'top_block bottom_block'
  []
  [bottom_right_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = block_rename
    new_boundary = bottom_right
    block = bottom_block
    normal = '1 0 0'
  []
  [bottom_left_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = bottom_right_sideset
    new_boundary = bottom_left
    block = bottom_block
    normal = '-1 0 0'
  []
  [bottom_top_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = bottom_left_sideset
    new_boundary = bottom_top
    block = bottom_block
    normal = '0 0 1'
  []
  [bottom_bottom_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = bottom_top_sideset
    new_boundary = bottom_bottom
    block = bottom_block
    normal = '0  0 -1'
  []
  [bottom_front_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = bottom_bottom_sideset
    new_boundary = bottom_front
    block = bottom_block
    normal = '0 1 0'
  []
  [bottom_back_sideset]
    type = SideSetsAroundSubdomainGenerator
    input = bottom_front_sideset
    new_boundary = bottom_back
    block = bottom_block
    normal = '0 -1 0'
  []
  [secondary]
    input = bottom_back_sideset
    type = LowerDBlockFromSidesetGenerator
    sidesets = 'top_bottom' # top_back top_left'
    new_block_id = '10001'
    new_block_name = 'secondary_lower'
  []
  [primary]
    input = secondary
    type = LowerDBlockFromSidesetGenerator
    sidesets = 'bottom_top'
    new_block_id = '10000'
    new_block_name = 'primary_lower'
  []
[]

[Variables]
  [disp_x]
    block = '1 2'
  []
  [disp_y]
    block = '1 2'
  []
  [disp_z]
    block = '1 2'
  []
  [lm_x]
    block = 'secondary_lower'
    use_dual = true
  []
  [lm_y]
    block = 'secondary_lower'
    use_dual = true
  []
  [lm_z]
    block = 'secondary_lower'
    use_dual = true
  []
[]

[ICs]
  [disp_z]
    block = 1
    variable = disp_z
    value = '${fparse offset}'
    type = ConstantIC
  []
  [disp_x]
    block = 1
    variable = disp_x
    value = 0
    type = ConstantIC
  []
  [disp_y]
    block = 1
    variable = disp_y
    value = 0
    type = ConstantIC
  []
[]

[Kernels]
  [disp_x]
    type = MatDiffusion
    variable = disp_x
  []
  [disp_y]
    type = MatDiffusion
    variable = disp_y
  []
  [disp_z]
    type = MatDiffusion
    variable = disp_z
  []
[]

[Constraints]
  [weighted_gap_lm]
    type = ComputeFrictionalForceCartesianLMMechanicalContact
    primary_boundary = 'bottom_top'
    secondary_boundary = 'top_bottom'
    primary_subdomain = 'primary_lower'
    secondary_subdomain = 'secondary_lower'
    lm_x = lm_x
    lm_y = lm_y
    lm_z = lm_z
    variable = lm_x # This can be anything really
    disp_x = disp_x
    disp_y = disp_y
    disp_z = disp_z
    use_displaced_mesh = true
    correct_edge_dropping = true
    c = 1e+02
    c_t = 1e+2
    mu = 0.10
  []
  [normal_x]
    type = CartesianMortarMechanicalContact
    primary_boundary = 'bottom_top'
    secondary_boundary = 'top_bottom'
    primary_subdomain = 'primary_lower'
    secondary_subdomain = 'secondary_lower'
    variable = lm_x
    secondary_variable = disp_x
    component = x
    use_displaced_mesh = true
    compute_lm_residuals = false
    correct_edge_dropping = true
  []
  [normal_y]
    type = CartesianMortarMechanicalContact
    primary_boundary = 'bottom_top'
    secondary_boundary = 'top_bottom'
    primary_subdomain = 'primary_lower'
    secondary_subdomain = 'secondary_lower'
    variable = lm_y
    secondary_variable = disp_y
    component = y
    use_displaced_mesh = true
    compute_lm_residuals = false
    correct_edge_dropping = true
  []
  [normal_z]
    type = CartesianMortarMechanicalContact
    primary_boundary = 'bottom_top'
    secondary_boundary = 'top_bottom'
    primary_subdomain = 'primary_lower'
    secondary_subdomain = 'secondary_lower'
    variable = lm_z
    secondary_variable = disp_z
    component = z
    use_displaced_mesh = true
    compute_lm_residuals = false
    correct_edge_dropping = true
  []
[]

[BCs]
  [botx]
    type = DirichletBC
    variable = disp_x
    boundary = 'bottom_left bottom_right bottom_front bottom_back'
    value = 0.0
  []
  [boty]
    type = DirichletBC
    variable = disp_y
    boundary = 'bottom_left bottom_right bottom_front bottom_back'
    value = 0.0
  []
  [botz]
    type = DirichletBC
    variable = disp_z
    boundary = 'bottom_left bottom_right bottom_front bottom_back'
    value = 0.0
  []
  [topx]
    type = DirichletBC
    variable = disp_x
    boundary = 'top_top'
    value = 0.0
  []
  [topy]
    type = DirichletBC
    variable = disp_y
    boundary = 'top_top'
    value = 0.0
  []
  [topz]
    type = FunctionDirichletBC
    variable = disp_z
    boundary = 'top_top'
    function = '-${starting_point} * sin(2 * pi / 40 * t) + ${offset}'
  []
[]

[Preconditioning]
  [vcp]
    type = VCP
    full = true
    lm_variable = 'lm_x lm_y lm_z'
    primary_variable = 'disp_x disp_y disp_z'
    preconditioner = 'LU'
    is_lm_coupling_diagonal = true
    adaptive_condensation = true
  []
[]

[Executioner]
  type = Transient
  end_time = 1
  dt = .5
  dtmin = .01
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_factor_shift_type -pc_factor_shift_amount'
  petsc_options_value = ' NONZERO               1e-10'
  l_max_its = 100
  nl_max_its = 30
  # nl_rel_tol = 1e-6
  nl_abs_tol = 1e-12
  line_search = 'none'
  snesmf_reuse_base = false
[]

[Debug]
  show_var_residual_norms = true
[]

[Outputs]
  perf_graph = true
  exodus = true
  csv = true
[]

[Postprocessors]
  active = 'num_nl cumulative contact'
  [num_nl]
    type = NumNonlinearIterations
  []
  [cumulative]
    type = CumulativeValuePostprocessor
    postprocessor = num_nl
  []
  [contact]
    type = ContactDOFSetSize
    variable = lm_z
    subdomain = 'secondary_lower'
    execute_on = 'nonlinear timestep_end'
  []
[]

[VectorPostprocessors]
  [contact-pressure]
    type = NodalValueSampler
    block = secondary_lower
    variable = lm_z
    sort_by = 'id'
    execute_on = NONLINEAR
  []
[]
