//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "Compressor1PhaseInertiaAux.h"
#include "ADShaftConnectedCompressor1PhaseUserObject.h"

registerMooseObject("ThermalHydraulicsApp", Compressor1PhaseInertiaAux);

InputParameters
Compressor1PhaseInertiaAux::validParams()
{
  InputParameters params = AuxScalarKernel::validParams();
  params.addRequiredParam<UserObjectName>("compressor_uo", "Compressor user object name");
  params.addClassDescription(
      "Returns the moment of inertia computed in the 1-phase shaft-connected compressor.");
  return params;
}

Compressor1PhaseInertiaAux::Compressor1PhaseInertiaAux(const InputParameters & parameters)
  : AuxScalarKernel(parameters),
    _compressor_uo(getUserObject<ADShaftConnectedCompressor1PhaseUserObject>("compressor_uo"))
{
}

Real
Compressor1PhaseInertiaAux::computeValue()
{
  return MetaPhysicL::raw_value(_compressor_uo.getMomentOfInertia());
}
