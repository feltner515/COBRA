# yy is passed in from the parent app

[Mesh]
  [line]
    type = GeneratedMeshGenerator
    dim = 1
    nx = 4
    xmax = 2
  []
  [box]
    type = SubdomainBoundingBoxGenerator
    input = line
    bottom_left = '0 -0.1 -0.1'
    top_right = '1 0.1 0.1'
    # need a different block ID than what is in the parent app to make sure the transfer works properly
    block_id = 20
  []
[]

[AuxVariables]
  [A]
    family = MONOMIAL
    order = CONSTANT
  []

  [S]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [A_ak]
    type = ParsedAux
    variable = A
    use_xyzt = true
    expression = '2*x+4*${yy}'
    execute_on = 'TIMESTEP_BEGIN'
  []
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [td]
    type = TimeDerivative
    variable = u
  []

  [diff]
    type = Diffusion
    variable = u
  []
[]

[UserObjects]
  [A_avg]
    type = LayeredAverage
    block = 20
    num_layers = 2
    direction = x
    variable = A
    execute_on = TIMESTEP_END
  []
[]

[Executioner]
  type = Transient
[]

[Outputs]
  exodus = true
[]
