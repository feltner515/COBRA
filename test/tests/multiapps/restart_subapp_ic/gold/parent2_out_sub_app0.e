CDF      
      
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes         num_elem   
   
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1     
   num_nod_per_el1       num_side_ss1      num_side_ss2      num_nod_ns1       num_nod_ns2       num_nod_var       num_info           api_version       @�
=   version       @�
=   floating_point_word_size            	file_size               int64_status             title         master2_out_sub_app0.e     maximum_name_length                    
time_whole                            _�   	eb_status                             �   eb_prop1               name      ID              �   	ns_status         	                    �   ns_prop1      	         name      ID              �   	ss_status         
                    �   ss_prop1      
         name      ID              �   coordx                      X      �   coordy                      X         coordz                      X      l   eb_names                       $      �   ns_names      	                 D      �   ss_names      
                 D      	,   
coor_names                         d      	p   node_num_map                    ,      	�   connect1                  	elem_type         EDGE2         P      
    elem_num_map                    (      
P   elem_ss1                          
x   side_ss1                          
|   elem_ss2                          
�   side_ss2                          
�   node_ns1                          
�   node_ns2                          
�   vals_nod_var1                          X      _�   name_nod_var                       $      
�   info_records                      U       
�                                         ?�������?ə�����?�333333?ٙ�����?�      ?�333333?�ffffff?陙����?�������?�                                                                                                                                                                                                                          left                             right                              right                            left                                                                                                                                                              	   
                                                   	   	   
   
                              	   
   
               u                                   ####################@������                                                     # Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                   -i                                                                               master2.i                                                                                                                                                         ### Input File ###                                                                                                                                                []                                                                                 initial_from_file_timestep     = LATEST                                          initial_from_file_var          = INVALID                                         block                          = INVALID                                         coord_type                     = XYZ                                             fe_cache                       = 0                                               kernel_coverage_check          = 1                                               material_coverage_check        = 1                                               name                           = 'MOOSE Problem'                                 restart_file_base              = INVALID                                         rz_coord_axis                  = Y                                               type                           = FEProblem                                       use_legacy_uo_aux_computation  = INVALID                                         use_legacy_uo_initialization   = INVALID                                         element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            active_bcs                     = INVALID                                         active_kernels                 = INVALID                                         inactive_bcs                   = INVALID                                         inactive_kernels               = INVALID                                         start                          = 0                                               control_tags                   = INVALID                                         dimNearNullSpace               = 0                                               dimNullSpace                   = 0                                               enable                         = 1                                               error_on_jacobian_nonzero_reallocation = 0                                       petsc_inames                   =                                                 petsc_options                  = INVALID                                         petsc_values                   =                                                 solve                          = 1                                               use_nonlinear                  = 1                                             []                                                                                                                                                                [BCs]                                                                                                                                                               [./left]                                                                           boundary                     = left                                              control_tags                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                                 value                        = 0                                               [../]                                                                                                                                                             [./right]                                                                          boundary                     = right                                             control_tags                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 type                         = FunctionDirichletBC                               use_displaced_mesh           = 0                                                 variable                     = u                                                 diag_save_in                 = INVALID                                           function                     = u_fn                                              save_in                      = INVALID                                           seed                         = 0                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      type                           = Transient                                       abort_on_solve_fail            = 0                                               compute_initial_residual_before_preset_bcs = 0                                   control_tags                   =                                                 dt                             = 0.1                                             dtmax                          = 1e+30                                           dtmin                          = 2e-14                                           enable                         = 1                                               end_time                       = 1e+30                                           l_abs_step_tol                 = -1                                              l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         n_startup_steps                = 0                                               nl_abs_step_tol                = 1e-50                                           nl_abs_tol                     = 1e-50                                           nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 1e-50                                           nl_rel_tol                     = 1e-08                                           no_fe_reinit                   = 0                                               num_steps                      = 5                                               petsc_options                  = INVALID                                         petsc_options_iname            = INVALID                                         petsc_options_value            = INVALID                                         picard_abs_tol                 = 1e-50                                           picard_max_its                 = 1                                               picard_rel_tol                 = 1e-08                                           reset_dt                       = 0                                               restart_file_base              =                                                 scheme                         = INVALID                                         solve_type                     = PJFNK                                           splitting                      = INVALID                                         ss_check_tol                   = 1e-08                                           ss_tmin                        = 0                                               start_time                     = 0                                               time_period_ends               = INVALID                                         time_period_starts             = INVALID                                         time_periods                   = INVALID                                         timestep_tolerance             = 2e-14                                           trans_ss_check                 = 0                                               use_multiapp_dt                = 0                                               verbose                        = 0                                             []                                                                                                                                                                [Executioner]                                                                      _fe_problem                    = 0x7f93b382d600                                []                                                                                                                                                                [Functions]                                                                                                                                                         [./ffn]                                                                            type                         = ParsedFunction                                    control_tags                 = Functions                                         enable                       = 1                                                 vals                         = INVALID                                           value                        = x                                                 vars                         = INVALID                                         [../]                                                                                                                                                             [./u_fn]                                                                           type                         = ParsedFunction                                    control_tags                 = Functions                                         enable                       = 1                                                 vals                         = INVALID                                           value                        = t*x                                               vars                         = INVALID                                         [../]                                                                          []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff]                                                                           type                         = Diffusion                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                                                                                                             [./fn]                                                                             type                         = UserForcingFunction                               block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 function                     = ffn                                               implicit                     = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                                                                                                             [./td]                                                                             type                         = TimeDerivative                                    block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 lumping                      = 0                                                 save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             displacements                  = INVALID                                         block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         patch_size                     = 40                                              second_order                   = 0                                               skip_partitioning              = 0                                               type                           = GeneratedMesh                                   uniform_refine                 = 0                                               centroid_partitioner_direction = INVALID                                         control_tags                   =                                                 dim                            = 1                                               distribution                   = DEFAULT                                         elem_type                      = INVALID                                         enable                         = 1                                               nemesis                        = 0                                               nx                             = 10                                              ny                             = 1                                               nz                             = 1                                               partitioner                    = default                                         patch_update_strategy          = never                                           xmax                           = 1                                               xmin                           = 0                                               ymax                           = 1                                               ymin                           = 0                                               zmax                           = 1                                               zmin                           = 0                                             []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = INVALID                                         gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     =                                                 tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                             []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              block                        = INVALID                                           eigen                        = 0                                                 family                       = LAGRANGE                                          initial_condition            = 4.2                                               order                        = FIRST                                             outputs                      = INVALID                                           scaling                      = 1                                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                          []                                                                                  ?�      @������@������@������@������@������@������@������@������@������@������@������?�333333�5�r    ?��n(V��?�����`@F˶�D�@��4�8
@��]���@S��fJw@�+j&�@ ��Pe��?��=z�P^?�330ذ?�ffffff<L�:�   ?�ڨ�8�?�����?�K���?��E*c?���3?��ƴ<{o?�K�<�2?�j�'�?���)Qz�?�fffn��?陙����:���    ?��fK�?�6��O�z?�B�X��=?�����J?𗘠:n?�^�:���?�6���?�?�Ǌ�?�^��,�?陙�r��?�������8�jՐ   ?�6e����?ו=�1\�?��!;��G?�C	w?��0�9d?�3}
e�?�phɲ�2?�rWN[/?�sM��?���̓In?�������7O�p   ?�&ckn|b?�֏4NL?ۂ>�`��?ᮺD�oW?�!��QVL?���L��?�*>	��?인�B�?�b�x��?����۳k