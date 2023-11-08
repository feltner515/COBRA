//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ReactorGeometryMeshBuilderBase.h"
#include "MooseEnum.h"

/**
 * Mesh generator for defining a reactor core using a Cartesian or hexagonal lattice with the option
 * to be 2-D or 3-D.
 */
class CoreMeshGenerator : public ReactorGeometryMeshBuilderBase
{
public:
  static InputParameters validParams();

  CoreMeshGenerator(const InputParameters & parameters);

  std::unique_ptr<MeshBase> generate() override;

protected:
  ///The names of the assemblies that compose the core
  const std::vector<MeshGeneratorName> _inputs;

  ///The name of "filler" assembly given in the input to represent an empty space in the core pattern
  const MeshGeneratorName _empty_key;

  ///Whether empty positions are to be used in the pattern
  bool _empty_pos = false;

  ///The 2D assembly layout of the core
  const std::vector<std::vector<unsigned int>> _pattern;

  ///Whether this mesh should be extruded to 3-D, the core is always assumed to be the last
  const bool _extrude;

  ///The geometry type for the reactor that is stored on the ReactorMeshParams object
  std::string _geom_type;

  /// Whether the core periphery should be meshed
  const bool _mesh_periphery;
  /// Which periphery meshgenerator to use
  const MooseEnum _periphery_meshgenerator;
  /// "region_id" extra-element integer of the periphery mesh elements
  const subdomain_id_type _periphery_region_id;
  /// outer circle boundary radius
  const Real _outer_circle_radius;
  /// Number of segments in the outer circle boundary
  const unsigned int _outer_circle_num_segments;
  /// The subdomain name for the generated mesh outer boundary.
  const std::string _periphery_block_name;
  /// Number of periphery layers
  const unsigned int _periphery_num_layers;
  /// Desired (maximum) triangle area
  const Real _desired_area;
  /// Desired (local) triangle area as a function of (x,y)
  std::string _desired_area_func;

  ///The number of dimensions the mesh is ultimately going to have (2 or 3, declared in the ReactorMeshParams object)
  int _mesh_dimensions;

  ///A mapping from pin-type IDs to region IDs used when assigning region IDs during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<std::vector<subdomain_id_type>>> _pin_region_id_map;

  ///A mapping from pin-type IDs to block names used when assigning block names during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<std::vector<std::string>>> _pin_block_name_map;

  ///A mapping from assembly-type IDs to region IDs in the assembly duct regions used when assigning region IDs during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<std::vector<subdomain_id_type>>> _duct_region_id_map;

  ///A mapping from assembly-type IDs to block names in the assembly duct regions used when assigning block names during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<std::vector<std::string>>> _duct_block_name_map;

  ///A mapping from assembly-type IDs to region IDs in the assembly background regions used when assigning region IDs during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<subdomain_id_type>> _background_region_id_map;

  ///A mapping from assembly-type IDs to block names in the assembly background regions used when assigning block names during the assembly stitching stage
  std::map<subdomain_id_type, std::vector<std::string>> _background_block_name_map;

  /// The final mesh that is generated by the subgenerators;
  /// This mesh is generated by the subgenerators with only element and boundary IDs changed.
  std::unique_ptr<MeshBase> * _build_mesh;
};