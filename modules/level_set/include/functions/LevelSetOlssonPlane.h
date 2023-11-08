//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Function.h"

/**
 * Implementation of a level set function to represent a plane.
 */
class LevelSetOlssonPlane : public Function
{
public:
  static InputParameters validParams();

  LevelSetOlssonPlane(const InputParameters & parameters);

  using Function::value;
  virtual Real value(Real /*t*/, const Point & p) const override;

  virtual RealGradient gradient(Real /*t*/, const Point & p) const override;

protected:
  /// A point on the plane
  const RealVectorValue & _point;

  /// The normal vector to the plane
  const RealVectorValue & _normal;

  /// The interface thickness
  const Real & _epsilon;
};
