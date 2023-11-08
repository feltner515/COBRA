//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "GeneralUserObject.h"

/**
 * A naive way to check if all boundaries are gathered to every single processor.
 */
class CheckGhostedBoundaries : public GeneralUserObject
{
public:
  static InputParameters validParams();

  CheckGhostedBoundaries(const InputParameters & params);

  virtual void initialSetup(){};

  virtual void initialize(){};
  virtual void execute();
  virtual void finalize(){};

private:
  dof_id_type _total_num_bdry_sides;
};
