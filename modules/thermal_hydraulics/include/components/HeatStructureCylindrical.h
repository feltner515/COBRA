//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "HeatStructureCylindricalBase.h"

/**
 * Component to model cylindrical heat structure
 */
class HeatStructureCylindrical : public HeatStructureCylindricalBase
{
public:
  HeatStructureCylindrical(const InputParameters & params);

  virtual void check() const override;

public:
  static InputParameters validParams();
};
