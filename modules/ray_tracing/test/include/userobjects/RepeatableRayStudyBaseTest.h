//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "RepeatableRayStudy.h"

class RepeatableRayStudyBaseTest : public RepeatableRayStudy
{
public:
  RepeatableRayStudyBaseTest(const InputParameters & parameters);

  static InputParameters validParams();

protected:
  void defineRays() override;
};

class RepeatableRayStudyDefineNoClaimTest : public RepeatableRayStudy
{
public:
  RepeatableRayStudyDefineNoClaimTest(const InputParameters & params);

  static InputParameters validParams();

protected:
  void defineRays() override {}
};
