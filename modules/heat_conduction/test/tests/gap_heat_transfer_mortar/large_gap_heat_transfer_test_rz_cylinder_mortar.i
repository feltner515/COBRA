rpv_core_gap_size = 0.2

core_outer_radius = 2
rpv_inner_radius = '${fparse 2 + rpv_core_gap_size}'
rpv_outer_radius = '${fparse 2.5 + rpv_core_gap_size}'
rpv_width = '${fparse rpv_outer_radius - rpv_inner_radius}'

rpv_outer_htc = 10 # W/m^2/K
rpv_outer_Tinf = 300 # K

core_blocks = '1'
rpv_blocks = '3'

[Mesh]
  [gmg]
    type = CartesianMeshGenerator
    dim = 2
    dx = '${core_outer_radius} ${rpv_core_gap_size} ${rpv_width}'
    ix = '400 1 100'
    dy = 1
    iy = '5'
  []
  [set_block_id1]
    type = SubdomainBoundingBoxGenerator
    input = gmg
    bottom_left = '0 0 0'
    top_right = '${core_outer_radius} 1 0'
    block_id = 1
    location = INSIDE
  []
  [rename_core_bdy]
    type = SideSetsBetweenSubdomainsGenerator
    input = set_block_id1
    primary_block = 1
    paired_block = 0
    new_boundary = 'core_outer'
  []
  [set_block_id3]
    type = SubdomainBoundingBoxGenerator
    input = rename_core_bdy
    bottom_left = '${rpv_inner_radius} 0 0'
    top_right = '${rpv_outer_radius} 1 0'
    block_id = 3
    location = INSIDE
  []
  [rename_inner_rpv_bdy]
    type = SideSetsBetweenSubdomainsGenerator
    input = set_block_id3
    primary_block = 3
    paired_block = 0
    new_boundary = 'rpv_inner'
  []
  # comment out for test without gap
  [2d_mesh]
    type = BlockDeletionGenerator
    input = rename_inner_rpv_bdy
    block = 0
  []
  [secondary]
    type = LowerDBlockFromSidesetGenerator
    sidesets = 'rpv_inner'
    new_block_id = 10001
    new_block_name = 'secondary_lower'
    input = 2d_mesh
  []
  [primary]
    type = LowerDBlockFromSidesetGenerator
    sidesets = 'core_outer'
    new_block_id = 10000
    new_block_name = 'primary_lower'
    input = secondary
  []
  allow_renumbering = false
[]

[Problem]
  coord_type = RZ
[]

[Variables]
  [Tsolid]
    initial_condition = 500
  []
    [lm]
      order = FIRST
      family = LAGRANGE
      block = 'secondary_lower'
    []
[]

[Kernels]
  [heat_source]
    type = CoupledForce
    variable = Tsolid
    block = '${core_blocks}'
    v = power_density
  []
  [heat_conduction]
    type = HeatConduction
    variable = Tsolid
  []
[]

[BCs]
  [RPV_out_BC] # k \nabla T = h (T- T_inf) at RPV outer boundary
    type = ConvectiveFluxFunction # (Robin BC)
    variable = Tsolid
    boundary = 'right' # outer RPV
    coefficient = ${rpv_outer_htc}
    T_infinity = ${rpv_outer_Tinf}
  []
[]

[UserObjects]
  [radiation]
    type = GapFluxModelRadiation
    temperature = Tsolid
    boundary = 'rpv_inner'
    primary_emissivity = 0.8
    secondary_emissivity = 0.8
  []
  [conduction]
    type = GapFluxModelConduction
    temperature = Tsolid
    boundary = 'rpv_inner'
    gap_conductivity = 0.1
  []
[]

[Constraints]
  [ced]
    type = ModularGapConductanceConstraint
    variable = lm
    secondary_variable = Tsolid
    primary_boundary = 'core_outer'
    primary_subdomain = 10000
    secondary_boundary = 'rpv_inner'
    secondary_subdomain = 10001
    gap_flux_models = 'radiation conduction'
    gap_geometry_type = 'CYLINDER'
  []
[]

[AuxVariables]
  [power_density]
    block = '${core_blocks}'
    initial_condition = 50e3
  []
[]

[Materials]
  [simple_mat]
    type = HeatConductionMaterial
    thermal_conductivity = 34.6 # W/m/K
  []
[]

[Postprocessors]
  [Tcore_avg]
    type = ElementAverageValue
    variable = Tsolid
    block = '${core_blocks}'
  []
  [Tcore_max]
    type = ElementExtremeValue
    value_type = max
    variable = Tsolid
    block = '${core_blocks}'
  []
  [Tcore_min]
    type = ElementExtremeValue
    value_type = min
    variable = Tsolid
    block = '${core_blocks}'
  []
  [Trpv_avg]
    type = ElementAverageValue
    variable = Tsolid
    block = '${rpv_blocks}'
  []
  [Trpv_max]
    type = ElementExtremeValue
    value_type = max
    variable = Tsolid
    block = '${rpv_blocks}'
  []
  [Trpv_min]
    type = ElementExtremeValue
    value_type = min
    variable = Tsolid
    block = '${rpv_blocks}'
  []
  [ptot]
    type = ElementIntegralVariablePostprocessor
    variable = power_density
    block = '${core_blocks}'
  []
  [rpv_convective_out]
    type = ConvectiveHeatTransferSideIntegral
    T_solid = Tsolid
    boundary = 'right' # outer RVP
    T_fluid = ${rpv_outer_Tinf}
    htc = ${rpv_outer_htc}
  []
  [heat_balance] # should be equal to 0 upon convergence
    type = ParsedPostprocessor
    function = '(rpv_convective_out - ptot) / ptot'
    pp_names = 'rpv_convective_out ptot'
  []
  [flux_from_core] # converges to ptot as the mesh is refined
    type = SideDiffusiveFluxIntegral
    variable = Tsolid
    boundary = core_outer
    diffusivity = thermal_conductivity
  []
  [flux_into_rpv] # converges to rpv_convective_out as the mesh is refined
    type = SideDiffusiveFluxIntegral
    variable = Tsolid
    boundary = rpv_inner
    diffusivity = thermal_conductivity
  []

[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[VectorPostprocessors]
  [NodalTemperature]
    type = NodalValueSampler
    sort_by = id
    boundary = 'rpv_inner core_outer'
    variable = Tsolid
  []
[]

[Executioner]
  type = Steady

  petsc_options = '-snes_converged_reason -pc_svd_monitor'
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package -mat_mffd_err -pc_factor_shift_type '
                        '-pc_factor_shift_amount'
  petsc_options_value = ' lu       superlu_dist                  1e-5          NONZERO               '
                        '1e-15'
  snesmf_reuse_base = false
  nl_rel_tol = 1e-10
  nl_abs_tol = 1e-10
  l_max_its = 100

  line_search = none
[]

[Outputs]
  exodus = false
  csv = true
[]
