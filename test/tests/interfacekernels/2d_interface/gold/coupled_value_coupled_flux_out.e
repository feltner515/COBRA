CDF      
   *   
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes      	   num_elem      
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1        num_nod_per_el1       num_el_in_blk2        num_nod_per_el2       num_side_ss1      num_side_ss2      num_side_ss3      num_side_ss4      num_side_ss5      num_side_ss6      num_side_ss7      num_side_ss8      num_side_ss9      num_side_ss10         num_side_ss11         num_side_ss12         num_nod_ns1       num_nod_ns2       num_nod_ns3       num_nod_ns4       num_nod_ns5       num_nod_ns6       num_nod_ns7       num_nod_ns8       num_nod_ns9       num_nod_ns10      num_nod_ns11      num_nod_ns12      num_nod_var       num_glo_var       num_info  �         api_version       @�
=   version       @�
=   floating_point_word_size            	file_size               int64_status             title         !coupled_value_coupled_flux_out.e       maximum_name_length                 ;   
time_whole                            �X   	eb_status                             �   eb_prop1               name      ID              �   	ns_status         	              0      �   ns_prop1      	         name      ID        0      �   	ss_status         
              0          ss_prop1      
         name      ID        0      0   coordx                      H      `   coordy                      H      �   eb_names                       D      �   ns_names      	                �      4   ss_names      
                �      �   
coor_names                         D      L   node_num_map                    $      �   connect1                  	elem_type         QUAD4         0      �   connect2                  	elem_type         QUAD4               �   elem_num_map                          �   elem_ss1                             side_ss1                             elem_ss2                             side_ss2                             elem_ss3                          $   side_ss3                          ,   elem_ss4                          4   side_ss4                          8   elem_ss5                          <   side_ss5                          D   elem_ss6                          L   side_ss6                          T   elem_ss7                          \   side_ss7                          d   elem_ss8                          l   side_ss8                          t   elem_ss9                          |   side_ss9                          �   	elem_ss10                             �   	side_ss10                             �   	elem_ss11                             �   	side_ss11                             �   	elem_ss12                             �   	side_ss12                             �   node_ns1                          �   node_ns2                          �   node_ns3                          �   node_ns4                          �   node_ns5                          �   node_ns6                           �   node_ns7      !                    �   node_ns8      "                    �   node_ns9      #                        	node_ns10         $                       	node_ns11         %                       	node_ns12         &                       vals_nod_var1                          H      �`   vals_nod_var2                          H      ��   name_nod_var      '                 D      $   name_glo_var      (                 D      h   vals_glo_var         (                    ��   info_records      )                |�      �                                                       	      
                                                                             
   	                          ?�      ?�              @       @       ?�              @                       ?�      ?�              ?�      @       @       @                                                                                                            master0_interface_to_0           right                            right_to_0                       top                              left                             top_to_0                         left_to_0                        bottom                           bottom_to_1                      left_to_1                        bottom_to_0                      bottom                           right                            master0_interface                bottom_to_0                      right_to_0                       master0_interface_to_0           top                              left                             top_to_0                         left_to_0                        bottom_to_1                      left_to_1                                                                                                                       	                                 	                                                                                                                                                                           	         	            	                  	                                 u                                v                                  u_int                            v_int                              ####################                                                             # Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                    -i coupled_value_coupled_flux.i --error --error-unused --error-override --no... -gdb-backtrace### Version Info ###                                               Framework Information:                                                           MOOSE Version:           git commit 410fa419c0 on 2018-06-26                     LibMesh Version:         05d16e2e395606dc186cfad9a764996da8d75c85                PETSc Version:           3.7.6                                                   Current Time:            Thu Jun 28 14:59:33 2018                                Executable Timestamp:    Thu Jun 28 14:01:39 2018                                                                                                                                                                                                  ### Input File ###                                                                                                                                                []                                                                                 inactive                       =                                                 initial_from_file_timestep     = LATEST                                          initial_from_file_var          = INVALID                                         element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            type                           = GAUSS                                         []                                                                                                                                                                [BCs]                                                                                                                                                               [./u]                                                                              boundary                     = 'left_to_0 bottom_to_0 right top'                 control_tags                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     =                                                   isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = VacuumBC                                          use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           alpha                        = 1                                                 diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                               [../]                                                                                                                                                             [./v]                                                                              boundary                     = 'left_to_1 bottom_to_1'                           control_tags                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     =                                                   isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = VacuumBC                                          use_displaced_mesh           = 0                                                 variable                     = v                                                 vector_tags                  = nontime                                           alpha                        = 1                                                 diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      inactive                       =                                                 isObjectAction                 = 1                                               type                           = Steady                                          compute_initial_residual_before_preset_bcs = 0                                   contact_line_search_allowed_lambda_cuts = 2                                      contact_line_search_ltol       = INVALID                                         control_tags                   =                                                 enable                         = 1                                               l_abs_step_tol                 = -1                                              l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         line_search_package            = petsc                                           mffd_type                      = wp                                              nl_abs_step_tol                = 1e-50                                           nl_abs_tol                     = 1e-50                                           nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 1e-50                                           nl_rel_tol                     = 1e-08                                           no_fe_reinit                   = 0                                               petsc_options                  = INVALID                                         petsc_options_iname            = INVALID                                         petsc_options_value            = INVALID                                         restart_file_base              =                                                 solve_type                     = NEWTON                                          splitting                      = INVALID                                       []                                                                                                                                                                [Executioner]                                                                    []                                                                                                                                                                [InterfaceKernels]                                                                                                                                                  [./interface]                                                                      inactive                     =                                                   isObjectAction               = 1                                                 type                         = PenaltyInterfaceDiffusion                         _moose_base                  = InterfaceKernel                                   boundary                     = master0_interface                                 control_tags                 = InterfaceKernels                                  diag_save_in                 = INVALID                                           diag_save_in_var_side        = INVALID                                           enable                       = 1                                                 implicit                     = 1                                                 neighbor_var                 = v                                                 penalty                      = 1e+06                                             save_in                      = INVALID                                           save_in_var_side             = INVALID                                           use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                          []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff_u]                                                                         inactive                     =                                                   isObjectAction               = 1                                                 type                         = CoeffParamDiffusion                               D                            = 4                                                 block                        = 0                                                 control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                                                                                                             [./diff_v]                                                                         inactive                     =                                                   isObjectAction               = 1                                                 type                         = CoeffParamDiffusion                               D                            = 2                                                 block                        = 1                                                 control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = v                                                 vector_tags                  = nontime                                         [../]                                                                                                                                                             [./source_u]                                                                       inactive                     =                                                   isObjectAction               = 1                                                 type                         = BodyForce                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           function                     = 1                                                 implicit                     = 1                                                 matrix_tags                  = system                                            postprocessor                = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 value                        = 1                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             inactive                       =                                                 displacements                  = INVALID                                         block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         isObjectAction                 = 1                                               second_order                   = 0                                               skip_partitioning              = 0                                               type                           = GeneratedMesh                                   uniform_refine                 = 0                                               allow_renumbering              = 1                                               bias_x                         = 1                                               bias_y                         = 1                                               bias_z                         = 1                                               centroid_partitioner_direction = INVALID                                         construct_node_list_from_side_list = 1                                           control_tags                   =                                                 dim                            = 2                                               elem_type                      = INVALID                                         enable                         = 1                                               gauss_lobatto_grid             = 0                                               ghosting_patch_size            = INVALID                                         max_leaf_size                  = 10                                              nemesis                        = 0                                               nx                             = 2                                               ny                             = 2                                               nz                             = 1                                               parallel_type                  = DEFAULT                                         partitioner                    = default                                         patch_size                     = 40                                              patch_update_strategy          = never                                           xmax                           = 2                                               xmin                           = 0                                               ymax                           = 2                                               ymin                           = 0                                               zmax                           = 1                                               zmin                           = 0                                             []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [MeshModifiers]                                                                                                                                                     [./break_boundary]                                                                 inactive                     =                                                   isObjectAction               = 1                                                 type                         = BreakBoundaryOnSubdomain                          boundaries                   = INVALID                                           control_tags                 = MeshModifiers                                     depends_on                   = interface                                         enable                       = 1                                                 force_prepare                = 0                                               [../]                                                                                                                                                             [./interface]                                                                      inactive                     =                                                   isObjectAction               = 1                                                 type                         = SideSetsBetweenSubdomains                         control_tags                 = MeshModifiers                                     depends_on                   = subdomain1                                        enable                       = 1                                                 force_prepare                = 0                                                 master_block                 = 0                                                 new_boundary                 = master0_interface                                 paired_block                 = 1                                               [../]                                                                                                                                                             [./subdomain1]                                                                     inactive                     =                                                   isObjectAction               = 1                                                 type                         = SubdomainBoundingBox                              block_id                     = 1                                                 block_name                   = INVALID                                           bottom_left                  = '(x,y,z)=(       0,        0,        0)'          control_tags                 = MeshModifiers                                     depends_on                   = INVALID                                           enable                       = 1                                                 force_prepare                = 0                                                 location                     = INSIDE                                            top_right                    = '(x,y,z)=(       1,        1,        0)'        [../]                                                                          []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               controls                       = 0                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = INVALID                                         gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         inactive                       =                                                 interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     =                                                 tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                             []                                                                                                                                                                [Postprocessors]                                                                                                                                                    [./u_int]                                                                          inactive                     =                                                   isObjectAction               = 1                                                 type                         = ElementIntegralVariablePostprocessor              allow_duplicate_execution_on_initial = 0                                         block                        = 0                                                 control_tags                 = Postprocessors                                    enable                       = 1                                                 execute_on                   = TIMESTEP_END                                      implicit                     = 1                                                 outputs                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                                                                                                             [./v_int]                                                                          inactive                     =                                                   isObjectAction               = 1                                                 type                         = ElementIntegralVariablePostprocessor              allow_duplicate_execution_on_initial = 0                                         block                        = 1                                                 control_tags                 = Postprocessors                                    enable                       = 1                                                 execute_on                   = TIMESTEP_END                                      implicit                     = 1                                                 outputs                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = v                                               [../]                                                                          []                                                                                                                                                                [Preconditioning]                                                                                                                                                   [./smp]                                                                            inactive                     =                                                   isObjectAction               = 1                                                 type                         = SMP                                               control_tags                 = Preconditioning                                   coupled_groups               = INVALID                                           enable                       = 1                                                 full                         = 1                                                 ksp_norm                     = unpreconditioned                                  mffd_type                    = wp                                                off_diag_column              = INVALID                                           off_diag_row                 = INVALID                                           pc_side                      = default                                           petsc_options                = INVALID                                           petsc_options_iname          = INVALID                                           petsc_options_value          = INVALID                                           solve_type                   = INVALID                                         [../]                                                                          []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              block                        = 0                                                 eigen                        = 0                                                 family                       = LAGRANGE                                          inactive                     =                                                   initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           scaling                      = 1                                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                                                                                                             [./v]                                                                              block                        = 1                                                 eigen                        = 0                                                 family                       = LAGRANGE                                          inactive                     =                                                   initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           scaling                      = 1                                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                          []                                                                                                                                                                                                                                                         ?�              ?� ��\��?����`e?� ��Z�?�j�9��C?�O��פ?�O��x,?�j�9�J$?��4��l"?�����?� �~��~?��XN�?� �~�ǲ                                        @0S��X�?�Q��&Uy