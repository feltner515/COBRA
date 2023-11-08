[Mesh]
  [cube]
    type = GeneratedMeshGenerator
    dim = 3
  []
  [extrude_top]
    type = SideSetExtruderGenerator
    input = cube
    sideset = 'top'
    extrusion_vector = '3 3 3'
    num_layers = 3
  []
[]
