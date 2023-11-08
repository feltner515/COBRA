CDF      
      
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes         num_elem      
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1        num_nod_per_el1       num_side_ss1      num_side_ss2      num_nod_ns1       num_nod_ns2       num_nod_var       num_info  J         api_version       @�
=   version       @�
=   floating_point_word_size            	file_size               int64_status             title         !stagnation_temperature_aux_out.e       maximum_name_length                    
time_whole                            sP   	eb_status                             �   eb_prop1               name      ID              �   	ns_status         	                    �   ns_prop1      	         name      ID              �   	ss_status         
                    �   ss_prop1      
         name      ID              �   coordx                            �   coordy                            �   coordz                            �   eb_names                       $      �   ns_names      	                 D      	   ss_names      
                 D      	X   
coor_names                         d      	�   node_num_map                          
    connect1                  	elem_type         EDGE2               
   elem_num_map                          
   elem_ss1                          
$   side_ss1                          
(   elem_ss2                          
,   side_ss2                          
0   node_ns1                          
4   node_ns2                          
8   vals_nod_var1                                sX   vals_nod_var2                                sp   vals_nod_var3                                s�   vals_nod_var4                                s�   vals_nod_var5                                s�   name_nod_var                       �      
<   info_records                      hl      
�                                         ?�      ?�                                                                                          left                             right                              right                            left                                                                                                                                                                                specific_internal_energy         specific_volume                  stagnation_temperature           u                                velocity                            ####################                                                             # Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                   -i                                                                               stagnation_temperature_aux.i                                                                                                                                      ### Version Info ###                                                             Framework Information:                                                           MOOSE version:           git commit 5ec52e9 on 2016-11-01                        PETSc Version:           3.6.4                                                   Current Time:            Tue Nov  1 10:19:11 2016                                Executable Timestamp:    Tue Nov  1 10:15:50 2016                                                                                                                                                                                                  ### Input File ###                                                                                                                                                []                                                                                 block                          = INVALID                                         coord_type                     = XYZ                                             fe_cache                       = 0                                               kernel_coverage_check          = 1                                               material_coverage_check        = 1                                               name                           = 'MOOSE Problem'                                 restart_file_base              = INVALID                                         rz_coord_axis                  = Y                                               type                           = FEProblem                                       use_legacy_uo_aux_computation  = INVALID                                         use_legacy_uo_initialization   = INVALID                                         element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            control_tags                   = INVALID                                         dimNearNullSpace               = 0                                               dimNullSpace                   = 0                                               enable                         = 1                                               error_on_jacobian_nonzero_reallocation = 0                                       force_restart                  = 0                                               petsc_inames                   =                                                 petsc_options                  = INVALID                                         petsc_values                   =                                                 solve                          = 1                                               use_nonlinear                  = 1                                             []                                                                                                                                                                [AuxKernels]                                                                                                                                                        [./specific_internal_energy_ak]                                                    type                         = ConstantAux                                       block                        = INVALID                                           boundary                     = INVALID                                           control_tags                 = AuxKernels                                        enable                       = 1                                                 execute_on                   = LINEAR                                            seed                         = 0                                                 use_displaced_mesh           = 0                                                 value                        = 1.0262e+06                                        variable                     = specific_internal_energy                        [../]                                                                                                                                                             [./specific_volume_ak]                                                             type                         = ConstantAux                                       block                        = INVALID                                           boundary                     = INVALID                                           control_tags                 = AuxKernels                                        enable                       = 1                                                 execute_on                   = LINEAR                                            seed                         = 0                                                 use_displaced_mesh           = 0                                                 value                        = 0.0012192                                         variable                     = specific_volume                                 [../]                                                                                                                                                             [./stagnation_temperature_ak]                                                      type                         = StagnationTemperatureAux                          block                        = INVALID                                           boundary                     = INVALID                                           control_tags                 = AuxKernels                                        e                            = specific_internal_energy                          enable                       = 1                                                 execute_on                   = LINEAR                                            fp                           = eos                                               seed                         = 0                                                 use_displaced_mesh           = 0                                                 v                            = specific_volume                                   variable                     = stagnation_temperature                            vel                          = velocity                                        [../]                                                                                                                                                             [./velocity_ak]                                                                    type                         = ConstantAux                                       block                        = INVALID                                           boundary                     = INVALID                                           control_tags                 = AuxKernels                                        enable                       = 1                                                 execute_on                   = LINEAR                                            seed                         = 0                                                 use_displaced_mesh           = 0                                                 value                        = 10                                                variable                     = velocity                                        [../]                                                                          []                                                                                                                                                                [AuxVariables]                                                                                                                                                      [./specific_internal_energy]                                                       block                        = INVALID                                           family                       = LAGRANGE                                          initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                                                                                                             [./specific_volume]                                                                block                        = INVALID                                           family                       = LAGRANGE                                          initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                                                                                                             [./stagnation_temperature]                                                         block                        = INVALID                                           family                       = LAGRANGE                                          initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                                                                                                             [./velocity]                                                                       block                        = INVALID                                           family                       = LAGRANGE                                          initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                          []                                                                                                                                                                [BCs]                                                                                                                                                               [./left_u]                                                                         boundary                     = 0                                                 control_tags                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                                 value                        = 1                                               [../]                                                                                                                                                             [./right_u]                                                                        boundary                     = 1                                                 control_tags                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                                 value                        = 2                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      type                           = Steady                                          compute_initial_residual_before_preset_bcs = 0                                   control_tags                   =                                                 enable                         = 1                                               l_abs_step_tol                 = -1                                              l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         nl_abs_step_tol                = 1e-50                                           nl_abs_tol                     = 1e-50                                           nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 1e-50                                           nl_rel_tol                     = 1e-08                                           no_fe_reinit                   = 0                                               petsc_options                  = INVALID                                         petsc_options_iname            = INVALID                                         petsc_options_value            = INVALID                                         restart_file_base              =                                                 solve_type                     = PJFNK                                           splitting                      = INVALID                                       []                                                                                                                                                                [Executioner]                                                                      _fe_problem                    = 0x7f912b807800                                []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff_u]                                                                         type                         = Diffusion                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             displacements                  = INVALID                                         block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         patch_size                     = 40                                              second_order                   = 0                                               skip_partitioning              = 0                                               type                           = GeneratedMesh                                   uniform_refine                 = 0                                               bias_x                         = 1                                               bias_y                         = 1                                               bias_z                         = 1                                               centroid_partitioner_direction = INVALID                                         construct_node_list_from_side_list = 1                                           control_tags                   =                                                 dim                            = 1                                               distribution                   = DEFAULT                                         elem_type                      = INVALID                                         enable                         = 1                                               gauss_lobatto_grid             = 0                                               nemesis                        = 0                                               nx                             = 2                                               ny                             = 1                                               nz                             = 1                                               parallel_type                  = DEFAULT                                         partitioner                    = default                                         patch_update_strategy          = never                                           xmax                           = 1                                               xmin                           = 0                                               ymax                           = 1                                               ymin                           = 0                                               zmax                           = 1                                               zmin                           = 0                                             []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [Modules]                                                                                                                                                           [./FluidProperties]                                                                                                                                                 [./eos]                                                                            type                       = StiffenedGasFluidProperties                         beta                       = 0.00046                                             control_tags               = Modules/FluidProperties                             cv                         = 1816                                                enable                     = 1                                                   execute_on                 = TIMESTEP_END                                        gamma                      = 2.35                                                k                          = 0.6                                                 mu                         = 0.001                                               p_inf                      = 1e+09                                               q                          = -1.167e+06                                          q_prime                    = 0                                                   use_displaced_mesh         = 0                                                 [../]                                                                          [../]                                                                          []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               controls                       = 0                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = INVALID                                         gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     =                                                 tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                             []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              block                        = INVALID                                           eigen                        = 0                                                 family                       = LAGRANGE                                          initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           scaling                      = 1                                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                          []                                                                                                                                                                                                                 ?�      A/Q0    A/Q0    A/Q0    ?S��%ho?S��%ho?S��%ho@��׶z�@��׶z�@��׶z�?�      ?�     @       @$      @$      @$      