import os
# import inputfilegenerator
import pandas as pd
import random
import string
import numpy as np
import math

import os
# import inputfilegenerator
# import pandas as pd
# import random
# import numpy as np
# def main():

def media_sampling(mediafile, archetype, massflowrate_kg, peeningtime, partarea):
    particledata=pd.read_csv(mediafile)
    particledata.rename(columns = {'MINOR DIAMETER':'minordiameter', 'MAJOR DIAMETER':'majordiameter'}, inplace = True)
    particledata.minordiameter=particledata.minordiameter
    particledata.majordiameter=particledata.majordiameter
    particledata.AREA=particledata.AREA
    areaeqdiameter=np.array(np.sqrt(4*particledata.AREA/np.pi))
    if int(archetype)==1:
        primary=np.empty((len(areaeqdiameter),1))
        secondary=np.empty((len(areaeqdiameter),1))
        for q in range (0,len(areaeqdiameter),1):
            primary[q]=((particledata.majordiameter[q])**2)/(particledata.minordiameter[q])
            secondary[q]=((particledata.minordiameter[q])**2)/(particledata.majordiameter[q])
        primary=primary.flatten()
        secondary=secondary.flatten()
    elif int(archetype)==2:
        x_rc=np.empty((len(areaeqdiameter),1))
        for q in range (0,len(areaeqdiameter),1):
            a=(np.pi-2)/8
            b=(particledata.majordiameter[q]-particledata.minordiameter[q])
            c=((4*particledata.majordiameter[q]*particledata.minordiameter[q]+(np.pi-2)*particledata.minordiameter[q]**2)/8)-particledata.AREA[q]
            x_rc[q]=(-b+np.sqrt((b**2)-4*a*c))/(2*a)
        primary=np.array(particledata.minordiameter).flatten()
        secondary=x_rc.flatten()
    else :
        primary=areaeqdiameter.flatten()
        secondary=areaeqdiameter.flatten()
    
    diameters=np.empty((len(primary),1))
    for n in range (0,len(primary),1):
        # total=int(primary[n])+int(secondary[n])
        # binom=random.randint(0, total)
        binom=random.uniform(0, 1)
        # if binom <= int(primary[n]):
        #     diameters[n]=primary[n]
        # else: 
        #     diameters[n]=secondary[n]
        if int(np.round(binom)) == 1:
            diameters[n]=primary[n]
        else: 
            diameters[n]=secondary[n]
    length = 5
    width = 5
    density=7.98*10**-9
    particlemass_kg=(length*width/partarea)*peeningtime*massflowrate_kg
    particlemass_mt=particlemass_kg/1000
    rollingmass=0
    m=0
    impacts=np.array([])
    areaeqRVE = np.array([])
    effectivedensity_RVE = np.array([])
    while rollingmass < particlemass_mt:
        impacts=np.append(impacts,random.sample(range(0, len(particledata)-1), 1))
        areaeqRVE = np.append(areaeqRVE, areaeqdiameter[int(impacts[m])])
        radius=areaeqdiameter[int(impacts[m])]/2000
        effectivedensity_RVE = np.append(effectivedensity_RVE, ((areaeqRVE[m]**3)/diameters[int(impacts[m])]**3))
        particlemass=(4/3)*np.pi*density*radius**3
        m=m+1
        rollingmass=rollingmass+particlemass
    impacts=impacts.reshape((len(impacts),1))
    particles_call=np.empty((len(impacts),1))
    for x in range (0,len(impacts),1):
        particles_call[x]=diameters[int(impacts[x])]
    # return particles_call
    IOE_particles = np.array([])
    IOE_effectivedensity = np.array([])
    x_coords = np.array([])
    y_coords = np.array([])
    for t in range (0,len(impacts),1):
        x=random.uniform(0, 5)
        y=random.uniform(0, 5)
        if (x < 0.75) & (y < 0.75) & (x > 0.25) & (y > 0.25):
            IOE_particles = np.append(IOE_particles, float(particles_call[t]))
            IOE_effectivedensity = np.append(IOE_effectivedensity, float(effectivedensity_RVE[t]))
            x_coords = np.append(x_coords, x)
            y_coords = np.append(y_coords, y)
    
    return IOE_particles, IOE_effectivedensity,  x_coords, y_coords

def velo_dist(velomean, velostd, IOE_particles, phimean, phistd, thetamean,thetastd):
    velocity = np.random.normal(velomean, velostd, len(IOE_particles))
    #fix these distributions
    phi = np.random.normal(phimean, phistd, len(IOE_particles))
    theta = np.random.normal(thetamean, thetastd, len(IOE_particles))
    velx = np.empty((len(IOE_particles),1))
    vely = np.empty((len(IOE_particles),1))
    velz = np.empty((len(IOE_particles),1))
    for n in range (0,len(IOE_particles),1):
        velx[n] = velocity[n] * np.sin(phi[n]) * np.cos(theta[n])
        vely[n] = velocity[n] * np.sin(phi[n]) * np.sin(theta[n])
        velz[n] = velocity[n] * np.cos(phi[n])
    return velx, vely, velz


def sphere_box_montecarlo(sphere_center, radius, box_coords, n):
    ## Randomly samples n points within a sphere and counts how many are within a box
    ## sphere_center: tuple of (x, y, z) coordinates of the center of the sphere
    ## radius: float radius of the sphere
    ## box_coords: tuple of (x1, y1, z1, x2, y2, z2) coordinates of the box, where (x1, y1, z1) is the bottom-left-front corner and (x2, y2, z2) is the top-right-back corner
    ## n: number of points to sample
    ## Returns the fraction of points that are within the box
    count = 0
    for p in range(0,int(n)):
        r = random.uniform(0, radius)
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 2*math.pi)
        cartesian = [r*math.sin(phi)*math.cos(theta), r*math.sin(theta)*math.sin(phi), r*math.cos(phi)]
        if ((cartesian[0] + sphere_center[0] > box_coords[0]) and (cartesian[0] + sphere_center[0] < box_coords[3]) and (cartesian[1] + sphere_center[1] > box_coords[1]) and (cartesian[1] + sphere_center[1] < box_coords[4]) and (cartesian[2] + sphere_center[2] > box_coords[2]) and (cartesian[2] + sphere_center[2] < box_coords[5])):
            count += 1
    return (count-1)/n


def bash_gen(filename, n_files):
    "generate a bash script to run a series of simualtions with restarts"
    f = open('{}.sh'.format(filename), 'w')
    f.write('~/original-moose2/restart/restart-opt -i {}_{}.i\n'.format(filename, int(0)))
    for n in range (1,n_files):
      f.write('~/original-moose2/restart/restart-opt -i {}_{}.i\n'.format(filename, int(n)))
    f.close()
# changes for init sim:
# create second order aux var for output displacements
# remove initial conditions on displacement
# make diriclelet BC's AD, increase n_r on shot to 3
#jacobian compute -> 
# investigate non-AD for increased efficiency
def zeros():
    f = open('Zeros.i', 'w')
    f.write('''[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]
[Mesh]
    [body]
        type = GeneratedMeshGenerator
        dim = 3
        nx = 10
        ny = 10
        nz = 10
        xmin = -100
        xmax = 100 
        ymin = -100
        ymax = 100
        zmin = -100
        zmax = 100
    []
    [body_sides]
        type = RenameBoundaryGenerator
        input = body
        old_boundary = '0 1 2 3'
        new_boundary = '10 11 12 13'
    []
    [body_id]
        type = SubdomainIDGenerator
        input = body_sides
        subdomain_id = 1
    []
[]
[Variables]
    [disp_x]
    []
    [disp_y]
    []
    [disp_z]
    []
[]
[Modules/TensorMechanics/Master]
    [all]
        add_variables = false
        strain = FINITE
        block = '1'
        use_automatic_differentiation = false
    []

[]
[BCs]
    [material_base_y]
        type = DirichletBC
        variable = disp_y
        boundary =  10
        value = 0
    []
[]
[Materials]
    [./tensor]
        type = ComputeIsotropicElasticityTensor
        block = '1'
        youngs_modulus = 1.0e7
        poissons_ratio = 0.25


        # use_displaced_mesh = true
    [../]
    [stress]
        type = ComputeFiniteStrainElasticStress
        block = '1'
        outputs = exodus
    []
[]
[Preconditioning]
  [./SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'
  petsc_options = '-snes_ksp_ew'

  petsc_options_iname = '-pc_type -snes_linesearch_type -pc_factor_shift_type -pc_factor_shift_amount'
  petsc_options_value = 'lu       basic                 NONZERO               1e-15'
  line_search = 'none'
  automatic_scaling = true
  nl_abs_tol = 1.5e-07
  nl_rel_tol = 1.5e-07
  l_max_its = 40
  start_time = 0.0
  [TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 10
    linear_iteration_ratio = 4
    dt = 0.01
  []
  dtmin = 1e-12
  end_time = 1
[]
# [VectorPostprocessors]
#   # [stress_00_1]
#   #   type = MaterialVectorPostprocessor
#   #   material = radial_return_stress
#   #   elem_ids = '1 2 3 4 5 6 7'
#   #   block = '1'
#   #   execute_on = 'FINAL'
#   # []
#   [shearstress]
#     type = NodalValueSampler
#     variable = stress_xy
#     block = '1'
#     sort_by = 'id'
#   []

# []

[Outputs]
    exodus = true
    # csv = true
    [./out2]
        type = Exodus
        # discontinuous = true
        elemental_as_nodal = true
        execute_elemental_on = NONE
        # block = '1'
        
    [../]
    # [out_3]
    #   file_base = 'input'
    #   type = CSV 
    #   show = 'stress_xy'
    #   execute_on = 'FINAL'     
    # []
[]''')
    f.close()
    return

def initialize(filename):
    open('{}.i'.format(filename),'w').close()
    f=open('{}.i'.format(filename),'w')
    f.write('''[GlobalParams]
    displacements= 'disp_x disp_y disp_z'
[]
[Variables]
  [disp_x]
    order = SECOND
    family = LAGRANGE
  []
  [disp_y]
    order = SECOND
    family = LAGRANGE
  []
  [disp_z]
    order = SECOND
    family = LAGRANGE
  []
[]\n''')
    f.close()
def userobjects_initial(filename):
  f = open('{}.i'.format(filename), 'a')
  f.write('''[UserObjects]
  [soln_zeros]
  type = SolutionUserObject
  execute_on = INITIAL
  mesh = Zeros_out2.e
  timestep = LATEST
  use_displaced_mesh = false
  system_variables = 'stress_00'
  []
[]

[Functions]
  [subs_solution_fcn_hardening_variable]
  type = SolutionFunction
  solution = soln_zeros # soln 
  []
  [subs_solution_fcn_stress_00]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_00
  [../]
  [subs_solution_fcn_stress_01]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_01
  [../]
  [subs_solution_fcn_stress_02]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_02

  [../]
  [subs_solution_fcn_stress_11]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_11
  [../]
  [subs_solution_fcn_stress_12]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_12
  [../]
  [subs_solution_fcn_stress_22]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_22
  [../]
  [subs_solution_fcn_stress_10]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_10
  [../]
  [subs_solution_fcn_stress_20]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_20
  [../]
  [subs_solution_fcn_stress_21]
  type = SolutionFunction
  solution = soln_zeros # soln_stress
  from_variable = stress_00 # subs_stress_21
  [../]
  [subs_solution_fcn_elastic_strain_00]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_00
  [../]
  [subs_solution_fcn_elastic_strain_01]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_01
  [../]
  [subs_solution_fcn_elastic_strain_02]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_02
  [../]
  [subs_solution_fcn_elastic_strain_11]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_11
  [../]
  [subs_solution_fcn_elastic_strain_12]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_12
  [../]
  [subs_solution_fcn_elastic_strain_22]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_22
  [../]
  [subs_solution_fcn_elastic_strain_10]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_10
  [../]
  [subs_solution_fcn_elastic_strain_20]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_20
  [../]
  [subs_solution_fcn_elastic_strain_21]
  type = SolutionFunction
  solution = soln_zeros # soln_elastic_strain
  from_variable = stress_00 # subs_elastic_strain_21
  [../]
  [shot_solution_fcn_hardening_variable]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  []
  [shot_solution_fcn_stress_00]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_01]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_02]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_11]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_12]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_22]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_10]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_20]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_stress_21]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_00]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_01]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_02]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_11]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_12]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_22]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  [../]
  [shot_solution_fcn_elastic_strain_10]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  []
  [shot_solution_fcn_elastic_strain_20]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  []
  [shot_solution_fcn_elastic_strain_21]
  type = SolutionFunction
  solution = soln_zeros # soln_zeros
  from_variable = stress_00 # stress_00
  []

  [displacementx_soln]
  type = SolutionFunction
  solution = soln_zeros # soln_displacement
  from_variable = stress_00 # subs_disp_00
  [../]
  [displacementy_soln]
  type = SolutionFunction
  solution = soln_zeros # soln_displacement
  from_variable = stress_00 # subs_disp_11
  []
  [displacementz_soln]
  type = SolutionFunction
  solution = soln_zeros # soln_displacement
  from_variable = stress_00 # subs_disp_22
  []
  [subs_solution_fcn_plastic_strain00]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_00
  [../]
  [subs_solution_fcn_plastic_strain01]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_01
  [../]
  [subs_solution_fcn_plastic_strain02]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_02
  [../]
  [subs_solution_fcn_plastic_strain11]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_11
  [../]
  [subs_solution_fcn_plastic_strain12]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_12
  [../]
  [subs_solution_fcn_plastic_strain22]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_22
  [../]
  [subs_solution_fcn_plastic_strain10]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_10
  [../]
  [subs_solution_fcn_plastic_strain20]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_20
  [../]
  [subs_solution_fcn_plastic_strain21]
  type = SolutionFunction
  solution = soln_zeros # soln_plastic_strain
  from_variable = stress_00 # subs_plastic_strain_21
  [../]
  [solution_fcn_eff_inel_strain]
  type = SolutionFunction
  solution = soln_zeros # soln_eff_inelast_strain
  from_variable = stress_00 # subs_effective_plastic_strain
  []
  [subs_solution_fcn_inel_strain00]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_00
  [../]
  [subs_solution_fcn_inel_strain01]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_01
  [../]
  [subs_solution_fcn_inel_strain02]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_02
  [../]
  [subs_solution_fcn_inel_strain11]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_11
  [../]
  [subs_solution_fcn_inel_strain12]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_12
  [../]
  [subs_solution_fcn_inel_strain22]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_22
  [../]
  [subs_solution_fcn_inel_strain10]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_10
  [../]
  [subs_solution_fcn_inel_strain20]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_20
  [../]
  [subs_solution_fcn_inel_strain21]
  type = SolutionFunction
  solution = soln_zeros # soln_inelast_strain
  from_variable = stress_00 # subs_combined_inelastic_strain_21
  [../]

  []\n''')
  f.close()
def userobjects(filename, restartbase):
  f = open('{}.i'.format(filename), 'a')
  f.write('''[UserObjects]
  [soln]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_hardening_variable'
  []
  [soln_displacement]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_disp_00 subs_disp_11 subs_disp_22'
  []
  [soln_stress]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_stress_00 subs_stress_01 subs_stress_02 subs_stress_11 subs_stress_12 subs_stress_22 subs_stress_10 subs_stress_20 subs_stress_21'
  []
  [soln_elastic_strain]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_elastic_strain_00 subs_elastic_strain_01 subs_elastic_strain_02 subs_elastic_strain_11 subs_elastic_strain_12 subs_elastic_strain_22 subs_elastic_strain_10 subs_elastic_strain_20 subs_elastic_strain_21'
  []
  [soln_zeros]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = Zeros_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'stress_00'
  []
  [soln_plastic_strain]
    type = SolutionUserObject
    
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_plastic_strain_00 subs_plastic_strain_01 subs_plastic_strain_02 subs_plastic_strain_11 subs_plastic_strain_12 subs_plastic_strain_22 subs_plastic_strain_10 subs_plastic_strain_20 subs_plastic_strain_21'
  []
  # [soln_inelast_strain]
  #   type = SolutionUserObject
  #   execute_on = INITIAL
  #   mesh = {}_out2.e
  #   timestep = LATEST
  #   use_displaced_mesh = false
  #   system_variables = 'subs_combined_inelastic_strain_00 subs_combined_inelastic_strain_01 subs_combined_inelastic_strain_02 subs_combined_inelastic_strain_11 subs_combined_inelastic_strain_12 subs_combined_inelastic_strain_22 subs_combined_inelastic_strain_10 subs_combined_inelastic_strain_20 subs_combined_inelastic_strain_21'
  # []
  [soln_eff_inelast_strain]
    type = SolutionUserObject
    execute_on = INITIAL
    mesh = {}_out2.e
    timestep = LATEST
    use_displaced_mesh = false
    system_variables = 'subs_effective_plastic_strain'
  []
[]

[Functions]
  [subs_solution_fcn_hardening_variable]
    type = SolutionFunction
    solution = soln 
  []
  [subs_solution_fcn_stress_00]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_00
  [../]
  [subs_solution_fcn_stress_01]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_01
  [../]
  [subs_solution_fcn_stress_02]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_02
    
  [../]
  [subs_solution_fcn_stress_11]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_11
  [../]
  [subs_solution_fcn_stress_12]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_12
  [../]
  [subs_solution_fcn_stress_22]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_22
  [../]
  [subs_solution_fcn_stress_10]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_10
  [../]
  [subs_solution_fcn_stress_20]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_20
  [../]
  [subs_solution_fcn_stress_21]
    type = SolutionFunction
    solution = soln_stress
    from_variable = subs_stress_21
  [../]
  [subs_solution_fcn_elastic_strain_00]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_00
  [../]
  [subs_solution_fcn_elastic_strain_01]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_01
  [../]
  [subs_solution_fcn_elastic_strain_02]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_02
  [../]
  [subs_solution_fcn_elastic_strain_11]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_11
  [../]
  [subs_solution_fcn_elastic_strain_12]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_12
  [../]
  [subs_solution_fcn_elastic_strain_22]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_22
  [../]
  [subs_solution_fcn_elastic_strain_10]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_10
  [../]
  [subs_solution_fcn_elastic_strain_20]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_20
  [../]
  [subs_solution_fcn_elastic_strain_21]
    type = SolutionFunction
    solution = soln_elastic_strain
    from_variable = subs_elastic_strain_21
  [../]
  [shot_solution_fcn_hardening_variable]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  []
  [shot_solution_fcn_stress_00]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_01]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_02]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_11]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_12]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_22]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_10]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_20]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_stress_21]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_00]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_01]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_02]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_11]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_12]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_22]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  [../]
  [shot_solution_fcn_elastic_strain_10]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  []
  [shot_solution_fcn_elastic_strain_20]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  []
  [shot_solution_fcn_elastic_strain_21]
    type = SolutionFunction
    solution = soln_zeros
    from_variable = stress_00
  []
  
  [displacementx_soln]
    type = SolutionFunction
    solution = soln_displacement
    from_variable = subs_disp_00
  [../]
  [displacementy_soln]
    type = SolutionFunction
    solution = soln_displacement
    from_variable = subs_disp_11
  []
  [displacementz_soln]
    type = SolutionFunction
    solution = soln_displacement
    from_variable = subs_disp_22
  []
  [subs_solution_fcn_plastic_strain00]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_00
  [../]
  [subs_solution_fcn_plastic_strain01]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_01
  [../]
  [subs_solution_fcn_plastic_strain02]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_02
  [../]
  [subs_solution_fcn_plastic_strain11]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_11
  [../]
  [subs_solution_fcn_plastic_strain12]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_12
  [../]
  [subs_solution_fcn_plastic_strain22]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_22
  [../]
  [subs_solution_fcn_plastic_strain10]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_10
  [../]
  [subs_solution_fcn_plastic_strain20]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_20
  [../]
  [subs_solution_fcn_plastic_strain21]
    type = SolutionFunction
    solution = soln_plastic_strain
    from_variable = subs_plastic_strain_21
  [../]
  [solution_fcn_eff_inel_strain]
    type = SolutionFunction
    solution = soln_eff_inelast_strain
    from_variable = subs_effective_plastic_strain
  []
  [subs_solution_fcn_inel_strain00]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_00
  [../]
  [subs_solution_fcn_inel_strain01]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_01
  [../]
  [subs_solution_fcn_inel_strain02]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_02
  [../]
  [subs_solution_fcn_inel_strain11]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_11
  [../]
  [subs_solution_fcn_inel_strain12]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_12
  [../]
  [subs_solution_fcn_inel_strain22]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_22
  [../]
  [subs_solution_fcn_inel_strain10]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_10
  [../]
  [subs_solution_fcn_inel_strain20]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_20
  [../]
  [subs_solution_fcn_inel_strain21]
    type = SolutionFunction
    solution = soln_plastic_strain # soln_inelast_strain
    from_variable = subs_plastic_strain_21
  [../]
  
[]\n'''.format(restartbase, restartbase, restartbase, 
             restartbase, restartbase, restartbase, 
             restartbase, restartbase, restartbase, 
             restartbase, restartbase, restartbase, 
             restartbase, restartbase))
  f.close()
def ics(filename, velx, vely, velz):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[ICs]
  [disp_x_ic]
    type = FunctionIC
    variable = disp_x
    function = displacementx_soln
    block = '1'
  []
  [disp_y_ic]
    type = FunctionIC
    variable = disp_y
    function = displacementy_soln
    block = '1'
  []
  [disp_z_ic]
    type = FunctionIC
    variable = disp_z
    function = displacementz_soln
    block = '1'
  []
  [vel_x]
    type = ConstantIC
    variable = vel_x
    value = {}
  []
  [vel_y]
    type = ConstantIC
    variable = vel_y
    value = {}
  []
  [vel_z]
    type = ConstantIC
    variable = vel_z
    value = {}
  []
[]
[Problem]
  ignore_zeros_in_jacobian = true
[]\n'''.format(float(velx), float(vely), float(velz)))
   f.close()
def mesh(filename, impact_x, impact_y, roc):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[Mesh]
    [subs_0]
      type = GeneratedMeshGenerator
      dim = 2 
      nx = 15
      ny = 15
      xmin = 0.25
      xmax = 0.75
      ymin = 0.25
      ymax = 0.75
      elem_type = QUAD9
    []
    [subs]
      type = AdvancedExtruderGenerator
      input = subs_0
      direction  = '0 0 1'
      num_layers = '2 6 2 10'
      heights = '0.4 0.3 0.05 0.25'
    []

    [subs_sides]
        type=RenameBoundaryGenerator
        input=subs
        old_boundary = '0 1 2 3 4 5'
        new_boundary = '10 11 12 13 14 15'
    []
    [subs_id]
        type=SubdomainIDGenerator
        input=subs_sides
        subdomain_id = 1
    []
    [shot]
        type = SphereMeshGenerator
        radius = {}
        n_smooth = 0
        nr = 2
        elem_type = HEX27
    []
    [shot_sides]
      type=RenameBoundaryGenerator
      input=shot
      old_boundary = '0'
      new_boundary = '20'
    []
    [shot_id]
        type=SubdomainIDGenerator
        input=shot_sides
        subdomain_id = 2
    []
    [shot_split]
      type = SubdomainBoundingBoxGenerator
      input = shot_id
      block_id = 3
      bottom_left = '-1 -1 0'
      top_right = '1 1 1'
    []
    [shot_delete]
      type = BlockDeletionGenerator
      input = shot_split
      block = 3
      new_boundary = 21
    []
    [translate_shot]
        type=TransformGenerator
        transform= TRANSLATE
        input=shot_delete
        vector_value = '{} {} {}'
    []
    [shot_sides2]
      type = SideSetsFromBoundingBoxGenerator
      input = translate_shot
      bottom_left = '0.25 0.25 1'
      top_right = '0.75 0.75 1.1'
      boundary_new = 22
      boundaries_old = '20'
      block_id = 2
    []

    [cmbn]
        type=MeshCollectionGenerator
        inputs='subs_id shot_sides2'
    []
    allow_renumbering=true
    patch_update_strategy = iteration
    ghosting_patch_size = 100
[]
[Adaptivity]
  initial_marker = box
  initial_steps = 1
  switch_h_to_p_refinement = true
  [./Markers]
    [./box]
      type = BoxMarker
      bottom_left = '{} {} 0.8'
      top_right = '{} {} 1'
      inside = refine
      outside = do_nothing
    [../]
  [../]
[]\n'''.format(roc, impact_x, impact_y, 1.01+roc, impact_x-0.075, impact_y-0.075, impact_x+0.075,impact_y+0.075))
   f.close()
   
def TMkernelBC(filename):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[Modules/TensorMechanics/Master]
    [./block1]
      add_variables = false
      strain = FINITE
      block = '1'
      use_automatic_differentiation = true
      base_name = 'subs'
    [../]
    [./block2]
      add_variables = false
      strain = FINITE
      block = '2'
      use_automatic_differentiation = true
      base_name = 'shot'
    [../]
[]

[BCs]
    [./symm_y_material]
        type = ADDirichletBC
        variable = disp_y
        boundary = '10 12'
        value = 0.0
    [../]
    [symm_x_material]
        type = ADDirichletBC
        variable = disp_x
        boundary = '11 13'
        value = 0.0
    [../]
    [./material_base_z]
        type = ADDirichletBC
        variable = disp_z
        boundary = 14
        value = 0.0
    [../]

[]\n''')

   f.close()   
def contact(filename):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[Contact]
  [./dummy_name]
    primary = 22
    secondary = 15
    model = coulomb
    formulation = penalty
    normalize_penalty = true
    friction_coefficient = 0.5
    penalty = 8e6
    tangential_tolerance = 0.005
    automatic_pairing_distance = 0.001
  [../]
[]
[Dampers]
  [./contact_slip]
    type = ContactSlipDamper
    secondary = 15
    primary = 22
  [../]
  [jacobian_damper]
    type = ReferenceElementJacobianDamper
    displacements = 'disp_x disp_y disp_z'
    max_increment = 0.003
    min_damping = 0.00001
  []
[]\n''')
   f.close()
def auxkernels(filename):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[Kernels]
    [inertia_z]
        type = ADInertialForce
        variable = disp_z
        velocity = vel_z
        acceleration = accel_z
        beta = 0.25
        gamma = 0.5
        alpha = 0
        eta = 0.0
        block = '2'
        density = ADdensity
    []
    [inertia_x]
        type = ADInertialForce
        variable = disp_x
        velocity = vel_x
        acceleration = accel_x
        beta = 0.25
        gamma = 0.5
        alpha = 0
        eta = 0.0
        block = '2'
        density = ADdensity
    []
    [inertia_y]
        type = ADInertialForce
        variable = disp_y
        velocity = vel_y
        acceleration = accel_y
        beta = 0.25
        gamma = 0.5
        alpha = 0
        eta = 0.0
        block = '2'
        density = ADdensity
    []
[]
[AuxKernels]
    [accel_z]
      type = NewmarkAccelAux
      variable = accel_z
      displacement = disp_z
      velocity = vel_z
      beta = 0.25
      execute_on = 'timestep_end'
      block = '2'
    []
    [vel_z]
      type = NewmarkVelAux
      variable = vel_z
      acceleration = accel_z
      gamma = 0.5
      execute_on = 'timestep_end'
      block = '2'
    []
    [vel_x]
        type = NewmarkVelAux
        variable = vel_x
        acceleration = accel_x
        gamma = 0.5
        execute_on = 'timestep_end'
        block = '2'
    []
    [accel_x]
        type = NewmarkAccelAux
        variable = accel_x
        displacement = disp_x
        velocity = vel_x
        beta = 0.25
        execute_on = 'timestep_end'
        block = '2'
    []
    [vel_y]
        type = NewmarkVelAux
        variable = vel_y
        acceleration = accel_y
        gamma = 0.5
        execute_on = 'timestep_end'
        block = '2'
    []
    [accel_y]
        type = NewmarkAccelAux
        variable = accel_y
        displacement = disp_y
        velocity = vel_y
        beta = 0.25
        execute_on = 'timestep_end'
        block = '2'
    []

    [kinetic_energy]
      type = ADKineticEnergyAux
      block = '2'
      variable = kinetic_energy
      newmark_velocity_x = vel_x
      newmark_velocity_y = vel_y
      newmark_velocity_z = vel_z
      density = ADdensity
    []
    [disp_x_subs]
      type = ParsedAux
      coupled_variables = 'disp_x'
      variable = subs_disp_00
      expression = 'disp_x'
    []
    [disp_y_subs]
      type = ParsedAux
      coupled_variables = 'disp_y'
      variable = subs_disp_11
      expression = 'disp_y'
    []
    [disp_z_subs]
      type = ParsedAux
      coupled_variables = 'disp_z'
      variable = subs_disp_22
      expression = 'disp_z'
    []
[]
[AuxVariables]
    [vel_x]
      block = '2'
    []
    [accel_x]
      block = '2'
    []
    [vel_y]
      block = '2'
    []
    [accel_y]
      block = '2'
    []
    [vel_z]
      block = '2'
    []
    [accel_z]
      block = '2'
    []
    [kinetic_energy]
      order = CONSTANT
      family = MONOMIAL
    []
    [elastic_energy]
      order = CONSTANT
      family = MONOMIAL
    []
    [subs_disp_00]
      order = SECOND
      family = LAGRANGE
    []
    [subs_disp_11]
      order = SECOND
      family = LAGRANGE
    []
    [subs_disp_22]
      order = SECOND
      family = LAGRANGE
    []
[]\n''')
   f.close()

def materials(filename, density_shot):
   f = open('{}.i'.format(filename), 'a')
   f.write('''
[Materials]
  [./tensor]
    type = ADComputeIsotropicElasticityTensor
    block = '2'
    youngs_modulus = 310000
    poissons_ratio = 0.31
    base_name = 'shot'
    

    # use_displaced_mesh = true
  [../]
  [./stress]
    type = ADComputeFiniteStrainElasticStress
    block = '2'
    base_name = 'shot'
  [../]


  [./tensor_2]
    type = ADComputeIsotropicElasticityTensor
    block = '1'
    youngs_modulus = 210000
    poissons_ratio = 0.31
    base_name = 'subs'
    # use_displaced_mesh = true
    # base_name = 'block1_sim0'
    outputs = exodus
  [../]

  [./power_law_hardening]
    type = ADIsotropicPowerLawHardeningStressUpdate
    # automatic_differentiation_return_mapping = true
    strength_coefficient = 640 #K
    strain_hardening_exponent = 0.15 #n
    block = '1'
    base_name = 'subs'
    # base_name = 'block1_sim0'

    # use_displaced_mesh = true
    # output_properties = true
    outputs = exodus

  [../]
  [./radial_return_stress]
    type = ADComputeMultipleInelasticStress
    inelastic_models = 'power_law_hardening'
    #tangent_operator = elastic
    block = '1'
    base_name = 'subs'
    # base_name = 'block1_sim0'

    # output_properties = 'stress_00'
    outputs = exodus
  [../]
  [density]
    type = ADGenericConstantMaterial
    block = '1'
    prop_names = 'density'
    prop_values = '7.98e-9'
    # base_name = 'subs'
    # output_properties = true
    outputs = exodus
   []
   [density_shot]
    type = ADGenericConstantMaterial
    block = '2'
    prop_names = 'density'
    prop_values = '{}'
    outputs = exodus
    # base_name = 'shot'
   []
   [ADdensity_shot]
    type = ADGenericConstantMaterial
    block = '2'
    prop_names = 'ADdensity'
    prop_values = '{}'
    # base_name = 'shot'
   []

[]\n'''.format(float(density_shot), float(density_shot)))
   f.close()
def executioner(filename, filebase):
   f = open('{}.i'.format(filename), 'a')
   f.write('''[Preconditioning]
  [./SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'
  petsc_options = '-snes_ksp_ew'

  petsc_options_iname = '-pc_type -snes_linesearch_type -pc_factor_shift_type -pc_factor_shift_amount'
  petsc_options_value = 'lu       basic                 NONZERO               1e-15'
  line_search = 'none'
  automatic_scaling = true
  nl_abs_tol = 6e-07
  nl_rel_tol = 1e-50
  l_max_its = 25
  nl_max_its = 200
  start_time = 0.0
  n_max_nonlinear_pingpong = 5
  [TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 70
    linear_iteration_ratio = 25
    dt = 1e-7
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.5
    growth_factor = 1.5
  []
  dtmin = 1e-12
  dtmax = 2.5e-7
  end_time = 6e-6
[]

[Outputs]
    exodus = true
    # csv = true
    [./out2]
        type = Exodus
        # discontinuous = true
        elemental_as_nodal = true
        execute_elemental_on = NONE
        # block = '1'
        
    [../]
    [out3]
      type = Exodus
      file_base = '{}'
      execute_on = FINAL
      refinements = 2
      elemental_as_nodal = true
      execute_elemental_on = none
    []
[]'''.format(filebase))
   f.close()


def initialfile(filename, impact_x, impact_y, roc, velx, vely, velz, shot_density,  filebase):
    zeros()
    initialize(filename)
    userobjects_initial(filename)
    ics(filename, velx, vely, velz)
    mesh(filename, impact_x, impact_y, roc)
    TMkernelBC(filename)
    contact(filename)
    auxkernels(filename)
    materials(filename, shot_density)
    executioner(filename, filebase)
    return 0


def restartfile(filename, impact_x, impact_y, roc, velx, vely, velz, density_shot, filebase, restartbase):
    initialize(filename)
    userobjects(filename, restartbase)
    ics(filename, velx, vely, velz)
    mesh(filename, impact_x, impact_y, roc)
    TMkernelBC(filename)
    contact(filename)
    auxkernels(filename)
    materials(filename, density_shot)
    executioner(filename, filebase)
    return 0

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def write_to_database(filename, n, impact_x, impact_y, roc, velx, vely, velz, filebase, mediafile, archetype, massflowrate_kg, peeningtime, partarea, velo_mean, velo_std):
    db = open("database.csv", "a")
    db.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(filename, n, impact_x, impact_y, roc, velx, vely, velz, filebase, mediafile, archetype, massflowrate_kg, peeningtime, partarea, velo_mean, velo_std))
    db.close()
    return 0

def main(n_trials, mediafile, archetype, massflowrate_kg, peeningtime, partarea, velo_mean, velo_std, density = 7.98E-9, thetamean = 0, thetastd = 0.001, phimean = 0, phistd = 0.001):
  for n in range(0,n_trials):
      filename = get_random_string(10)
      IOE_particles, IOE_effectivedensity,  x_coords, y_coords = media_sampling(mediafile, archetype, massflowrate_kg, peeningtime, partarea)
      velx, vely, velz  = velo_dist(velomean=velo_mean, velostd=velo_std, thetamean = thetamean, thetastd= thetastd, phimean=phimean, phistd=phistd, IOE_particles=IOE_particles)
      density_scale = np.zeros((len(IOE_particles),1))
      for p in range(0,len(IOE_particles)):
          density_scale[p] = sphere_box_montecarlo(radius = IOE_particles[p]/2000, sphere_center=[x_coords[0], y_coords[p], 2], box_coords=[0.25,0.25,0, 0.75,0.75,10], n=1E6)
      particledensity = np.zeros((len(IOE_particles),1))
      for p in range(0,len(IOE_particles)):
          particledensity[p] = 2*density_scale[p]*density*IOE_effectivedensity[p]
      #now we have the mass density for each particle
      bash_gen(filename=filename, n_files=len(IOE_particles))
      initialfile(filename='{}/{}_{}'.format(filename,filename,int(0)), impact_x = x_coords[0], impact_y = y_coords[0], roc = IOE_particles[0]/2000, velx=velx[0], vely=vely[0], velz=velz[0], shot_density=particledensity[0], filebase = '{}_{}'.format(filename,int(0)))
      write_to_database(filename=filename, n=0, impact_x = x_coords[0], impact_y = y_coords[0], roc = IOE_particles[0]/2000, velx=velx[0], vely=vely[0], velz=velz[0], filebase = 'init', mediafile = mediafile, archetype = archetype, massflowrate_kg = massflowrate_kg, peeningtime = peeningtime, partarea = partarea, velo_mean = velo_mean, velo_std = velo_std)
      for p in range (1,len(IOE_particles)):
          restartfile(filename='{}/{}_{}'.format(filename,filename,int(p)), impact_x = x_coords[p], impact_y = y_coords[p], roc = IOE_particles[p]/2000, velx=velx[p], vely=vely[p], velz=velz[p], density_shot=particledensity[p], filebase = '{}_{}'.format(filename,int(p)), restartbase = '{}_{}'.format(filename,int(p-1)))
          write_to_database(filename=filename, n=p, impact_x = x_coords[p], impact_y = y_coords[p], roc = IOE_particles[p]/2000, velx=velx[p], vely=vely[p], velz=velz[p], filebase = '{}_{}'.format(filename,int(p-1)), mediafile = mediafile, archetype = archetype, massflowrate_kg = massflowrate_kg, peeningtime = peeningtime, partarea = partarea, velo_mean = velo_mean, velo_std = velo_std)



#need to come up with a good to name files, do database, then restart from each previous simulation, and do bash file generation