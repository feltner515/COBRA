//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Marker.h"

class RandomHitUserObject;

class RandomHitMarker : public Marker
{
public:
  static InputParameters validParams();

  RandomHitMarker(const InputParameters & parameters);
  virtual ~RandomHitMarker(){};

protected:
  virtual MarkerValue computeElementMarker();

  const RandomHitUserObject & _random_hits;
};
