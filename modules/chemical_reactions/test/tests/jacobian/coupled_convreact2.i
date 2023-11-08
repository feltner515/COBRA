# Test the Jacobian terms for the CoupledConvectionReactionSub Kernel using
# activity coefficients not equal to unity

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 2
  ny = 2
[]

[Variables]
  [./a]
    order = FIRST
    family = LAGRANGE
  [../]
  [./b]
    order = FIRST
    family = LAGRANGE
  [../]
  [./pressure]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[ICs]
  [./pressure]
    type = RandomIC
    variable = pressure
    min = 1
    max = 5
  [../]
  [./a]
    type = RandomIC
    variable = a
    max = 1
    min = 0
  [../]
  [./b]
    type = RandomIC
    variable = b
    max = 1
    min = 0
  [../]
[]

[Kernels]
  [./diff]
    type = DarcyFluxPressure
    variable = pressure
  [../]
  [./diff_b]
    type = Diffusion
    variable = b
  [../]
  [./a1conv]
    type = CoupledConvectionReactionSub
    variable = a
    v = b
    log_k = 2
    weight = 1
    sto_v = 2.5
    sto_u = 2
    p = pressure
    gamma_eq = 2
    gamma_u = 2.5
    gamma_v = 1.5
  [../]
[]

[Materials]
  [./porous]
    type = GenericConstantMaterial
    prop_names = 'diffusivity conductivity porosity'
    prop_values = '1e-4 1e-4 0.2'
  [../]
[]

[Executioner]
  type = Steady
  solve_type = NEWTON
[]

[Outputs]
  perf_graph = true
[]

[Preconditioning]
  [./smp]
    type = SMP
    full = true
  [../]
[]
