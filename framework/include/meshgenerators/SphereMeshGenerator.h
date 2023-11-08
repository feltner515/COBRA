//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MeshGenerator.h"

/**
 * Create a sphere volume mesh.
 */
class SphereMeshGenerator : public MeshGenerator
{
public:
  static InputParameters validParams();

  SphereMeshGenerator(const InputParameters & parameters);

  std::unique_ptr<MeshBase> generate() override;

protected:
  /// sphere radius
  const Real & _radius;

  /// number of radial elements
  const unsigned int & _nr;

  /// element type
  const MooseEnum _elem_type;

  /// number of smoothing operations
  const unsigned int & _n_smooth;
};
