//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Limiter.h"
#include "MathFVUtils.h"

namespace Moose
{
namespace FV
{
/**
 * Implements a limiter which reproduces the upwind scheme, defined by
 * $\beta(r_f) = 0$
 */
template <typename T>
class UpwindLimiter : public Limiter<T>
{
public:
  T
  limit(const T &, const T &, const VectorValue<T> *, const RealVectorValue &) const override final
  {
    return 0;
  }
  bool constant() const override final { return true; }
  InterpMethod interpMethod() const override final { return InterpMethod::Upwind; }

  UpwindLimiter() = default;
};
}
}
