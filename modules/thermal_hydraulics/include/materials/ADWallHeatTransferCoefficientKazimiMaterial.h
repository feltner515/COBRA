//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "Material.h"

/**
 *  Computes wall heat transfer coefficient for liquid sodium using Kazimi-Carelli correlation
 */
class ADWallHeatTransferCoefficientKazimiMaterial : public Material
{
public:
  ADWallHeatTransferCoefficientKazimiMaterial(const InputParameters & parameters);

protected:
  virtual void computeQpProperties() override;

  /// Wall heat transfer coefficient
  ADMaterialProperty<Real> & _Hw;
  /// Density
  const ADMaterialProperty<Real> & _rho;
  /// Velocity
  const ADMaterialProperty<Real> & _vel;
  /// Hydraulic diameter
  const ADMaterialProperty<Real> & _D_h;
  /// Thermal conductivity
  const ADMaterialProperty<Real> & _k;
  /// Dynamic viscosity
  const ADMaterialProperty<Real> & _mu;
  /// Specific heat capacity
  const ADMaterialProperty<Real> & _cp;
  /// Fluid temperature
  const ADMaterialProperty<Real> & _T;
  /// Wall temperature
  const ADMaterialProperty<Real> & _T_wall;
  /// Pitch-to-Diameter ratio
  const Real & _PoD;

public:
  static InputParameters validParams();
};
