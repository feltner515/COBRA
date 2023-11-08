[Mesh]
  [hex_1]
    type = PolygonConcentricCircleMeshGenerator
    num_sides = 6
    num_sectors_per_side = '2 2 2 2 2 2'
    background_intervals = 1
    ring_radii = 4.0
    ring_intervals = 1
    ring_block_ids = '10'
    ring_block_names = 'center'
    background_block_ids = 20
    background_block_names = background
    polygon_size = 5.0
    preserve_volumes = on
  []
  [pattern]
    type = PatternedHexMeshGenerator
    inputs = 'hex_1'
    pattern = '0 0;
              0 0 0;
               0 0'
    background_intervals = 2
    background_block_id = 25
    background_block_name = 'assem_block'
    hexagon_size = 18
  []
  [center_trim]
    type = HexagonMeshTrimmer
    input = pattern
    center_trim_starting_index = 0
    center_trim_ending_index = 2
    center_trimming_section_boundary = symmetric
  []
[]
