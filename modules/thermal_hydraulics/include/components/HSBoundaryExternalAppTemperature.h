//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "HSBoundary.h"

/**
 * Heat structure boundary condition to set temperature values computed by an external application
 */
class HSBoundaryExternalAppTemperature : public HSBoundary
{
public:
  HSBoundaryExternalAppTemperature(const InputParameters & params);

  virtual void addVariables() override;
  virtual void addMooseObjects() override;

protected:
  /// The variable name that stores the values of temperature computed by an external application
  const VariableName & _T_ext_var_name;

public:
  static InputParameters validParams();
};
