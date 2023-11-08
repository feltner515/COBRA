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
#include "RankTwoTensor.h"
#include "RankFourTensor.h"
#include "RotationTensor.h"
#include "DerivativeMaterialInterface.h"

/**
 * ComputeGeneralStressBase is the direct base class for stress calculator materials that may
 * leverage quantities based on the displaced mesh (like the UMAT plugins) rather than solely using
 * strain tensors computed by separate MOOSE material objects (those classes should directly derive
 * from ComputeStressBase, which in turn directly derives from ComputeGeneralStressBase).
 */
class ComputeGeneralStressBase : public DerivativeMaterialInterface<Material>
{
public:
  static InputParameters validParams();

  ComputeGeneralStressBase(const InputParameters & parameters);

protected:
  virtual void initQpStatefulProperties() override;
  virtual void computeQpProperties() override;

  /**
   * Compute the stress and store it in the _stress material property
   * for the current quadrature point
   **/
  virtual void computeQpStress() = 0;

  /// Base name prepended to all material property names to allow for
  /// multi-material systems
  const std::string _base_name;

  /// Mechanical strain material property
  const MaterialProperty<RankTwoTensor> & _mechanical_strain;
  /// Stress material property
  MaterialProperty<RankTwoTensor> & _stress;
  /// Elastic strain material property
  MaterialProperty<RankTwoTensor> & _elastic_strain;

  /// Extra stress tensor
  const MaterialProperty<RankTwoTensor> & _extra_stress;

  /// initial stress components

  /// derivative of stress w.r.t. strain (_dstress_dstrain)
  MaterialProperty<RankFourTensor> & _Jacobian_mult;


  const Function & _functions00;
  const Function & _functions10;
  const Function & _functions20;
  const Function & _functions01;
  const Function & _functions11;
  const Function & _functions21;
  const Function & _functions02;
  const Function & _functions12;
  const Function & _functions22;
  const Function & _functions00es;
  const Function & _functions10es;
  const Function & _functions20es;
  const Function & _functions01es;
  const Function & _functions11es;
  const Function & _functions21es;
  const Function & _functions02es;
  const Function & _functions12es;
  const Function & _functions22es;


  RankTwoTensor _initstress;
  RankTwoTensor _initstrain;

};
