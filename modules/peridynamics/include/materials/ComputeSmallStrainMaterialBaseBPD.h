//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ParametricMaterialBasePD.h"

/**
 * Base material class for bond based peridynamic solid mechanics models
 */
class ComputeSmallStrainMaterialBaseBPD : public ParametricMaterialBasePD
{
public:
  static InputParameters validParams();

  ComputeSmallStrainMaterialBaseBPD(const InputParameters & parameters);

protected:
  virtual void computeBondForce() override;

  /// Micro-modulus
  Real _Cij;
};
