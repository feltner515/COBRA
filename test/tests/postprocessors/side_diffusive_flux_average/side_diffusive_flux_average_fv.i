[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 10
  ny = 10
[]

[Variables]
  [u]
    order = CONSTANT
    family = MONOMIAL
    fv = true
  []
[]

[FVKernels]
  [diff]
    type = FVDiffusion
    variable = u
    coeff = 1
  []
[]

[FVBCs]
  [left]
    type = FVDirichletBC
    variable = u
    boundary = left
    value = 0
  []
  [right]
    type = FVDirichletBC
    variable = u
    boundary = right
    value = 1
  []
[]

[Materials]
  [mat_props]
    type = GenericFunctorMaterial
    prop_names = diffusivity
    prop_values = 1
  []
[]

[Postprocessors]
  [avg_flux_right]
    # Computes flux integral on the boundary, which should be -1
    type = SideDiffusiveFluxAverage
    variable = u
    boundary = right
    functor_diffusivity = diffusivity
  []
[]

[Executioner]
  type = Steady
  solve_type = PJFNK
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
  nl_abs_tol = 1e-14
  nl_rel_tol = 1e-14
  l_abs_tol = 1e-14
  l_tol = 1e-6
[]

[Outputs]
  exodus = true
[]
