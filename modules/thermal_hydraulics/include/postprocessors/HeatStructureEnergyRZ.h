//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "HeatStructureEnergyBase.h"
#include "RZSymmetry.h"

/**
 * Computes the total energy for a cylindrical heat structure.
 */
class HeatStructureEnergyRZ : public HeatStructureEnergyBase, public RZSymmetry
{
public:
  HeatStructureEnergyRZ(const InputParameters & parameters);

protected:
  virtual Real computeQpIntegral();

public:
  static InputParameters validParams();
};
