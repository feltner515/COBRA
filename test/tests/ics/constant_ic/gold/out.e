CDF      
      
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes      	   num_elem      
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1        num_nod_per_el1       num_side_ss1      num_side_ss2      num_side_ss3      num_side_ss4      num_nod_ns1       num_nod_ns2       num_nod_ns3       num_nod_ns4       num_nod_var       num_info  K         api_version       @§
=   version       @§
=   floating_point_word_size            	file_size               int64_status             title         out.e      maximum_name_length                     
time_whole                            u   	eb_status                             θ   eb_prop1               name      ID              μ   	ns_status         	                    π   ns_prop1      	         name      ID              	    	ss_status         
                    	   ss_prop1      
         name      ID              	    coordx                      H      	0   coordy                      H      	x   eb_names                       $      	ΐ   ns_names      	                       	δ   ss_names      
                       
h   
coor_names                         D      
μ   node_num_map                    $      0   connect1                  	elem_type         QUAD4         @      T   elem_num_map                             elem_ss1                          €   side_ss1                          ¬   elem_ss2                          ΄   side_ss2                          Ό   elem_ss3                          Δ   side_ss3                          Μ   elem_ss4                          Τ   side_ss4                          ά   node_ns1                          δ   node_ns2                          π   node_ns3                          ό   node_ns4                             vals_nod_var1                          H      u   vals_nod_var2                          H      ud   name_nod_var                       D         info_records                      hΌ      X                                                                 ?ΰ      ?ΰ              ?π      ?π      ?ΰ              ?π                      ?ΰ      ?ΰ              ?ΰ      ?π      ?π      ?π                                          bottom                           right                            top                              left                             bottom                           left                             right                            top                                                                                                                             	                                             	                                                                                 	         	         u                                u_aux                              ####################@"@"@"@"@"@"@"@"# Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                    /Users/icenct/projects/moose/test/moose_test-opt -i constant_ic_test.i --err... or --error-unused --error-override --no-gdb-backtrace### Version Info ###                                                                                         Framework Information:                                                           MOOSE Version:           git commit 69928b2ab5 on 2020-10-02                     LibMesh Version:         053cc54aaf127bec1dd617a7302b65455a205908                PETSc Version:           3.13.3                                                  SLEPc Version:           3.13.3                                                  Current Time:            Fri Oct  2 18:04:47 2020                                Executable Timestamp:    Fri Oct  2 15:21:27 2020                                                                                                                                                                                                  ### Input File ###                                                                                                                                                []                                                                                 inactive                       = (no_default)                                    custom_blocks                  = (no_default)                                    custom_orders                  = (no_default)                                    element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            type                           = GAUSS                                         []                                                                                                                                                                [AuxVariables]                                                                                                                                                      [./u_aux]                                                                          family                       = LAGRANGE                                          inactive                     = (no_default)                                      isObjectAction               = 1                                                 order                        = FIRST                                             scaling                      = INVALID                                           type                         = MooseVariableBase                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                           block                        = INVALID                                           components                   = 1                                                 control_tags                 = AuxVariables                                      eigen                        = 0                                                 enable                       = 1                                                 fv                           = 0                                                 initial_condition            = INVALID                                           outputs                      = INVALID                                           use_dual                     = 0                                                                                                                                  [./InitialCondition]                                                               inactive                   = (no_default)                                        isObjectAction             = 1                                                   type                       = ConstantIC                                          block                      = INVALID                                             boundary                   = INVALID                                             control_tags               = AuxVariables/u_aux                                  enable                     = 1                                                   ignore_uo_dependency       = 0                                                   value                      = 9.3                                                 variable                   = u_aux                                             [../]                                                                          [../]                                                                          []                                                                                                                                                                [BCs]                                                                                                                                                               [./left]                                                                           boundary                     = 3                                                 control_tags                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     = (no_default)                                      isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           diag_save_in                 = INVALID                                           preset                       = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 value                        = 0                                               [../]                                                                                                                                                             [./right]                                                                          boundary                     = 1                                                 control_tags                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     = (no_default)                                      isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           diag_save_in                 = INVALID                                           preset                       = 1                                                 save_in                      = INVALID                                           seed                         = 0                                                 value                        = 1                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      auto_preconditioning           = 1                                               inactive                       = (no_default)                                    isObjectAction                 = 1                                               type                           = Steady                                          accept_on_max_picard_iteration = 0                                               auto_advance                   = INVALID                                         automatic_scaling              = INVALID                                         compute_initial_residual_before_preset_bcs = 0                                   compute_scaling_once           = 1                                               contact_line_search_allowed_lambda_cuts = 2                                      contact_line_search_ltol       = INVALID                                         control_tags                   = (no_default)                                    custom_abs_tol                 = 1e-50                                           custom_rel_tol                 = 1e-08                                           direct_pp_value                = 0                                               disable_picard_residual_norm_check = 0                                           enable                         = 1                                               l_abs_tol                      = 1e-50                                           l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         line_search_package            = petsc                                           max_xfem_update                = 4294967295                                      mffd_type                      = wp                                              nl_abs_step_tol                = 0                                               nl_abs_tol                     = 1e-50                                           nl_div_tol                     = -1                                              nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 0                                               nl_rel_tol                     = 1e-10                                           num_grids                      = 1                                               petsc_options                  = INVALID                                         petsc_options_iname            = INVALID                                         petsc_options_value            = INVALID                                         picard_abs_tol                 = 1e-50                                           picard_custom_pp               = INVALID                                         picard_force_norms             = 0                                               picard_max_its                 = 1                                               picard_rel_tol                 = 1e-08                                           relaxation_factor              = 1                                               relaxed_variables              = (no_default)                                    resid_vs_jac_scaling_param     = 0                                               restart_file_base              = (no_default)                                    scaling_group_variables        = INVALID                                         skip_exception_check           = 0                                               snesmf_reuse_base              = 1                                               solve_type                     = PJFNK                                           splitting                      = INVALID                                         time                           = 0                                               update_xfem_at_timestep_begin  = 0                                               verbose                        = 0                                             []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff]                                                                           inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = Diffusion                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             displacements                  = INVALID                                         inactive                       = (no_default)                                    use_displaced_mesh             = 1                                               include_local_in_ghosting      = 0                                               output_ghosting                = 0                                               block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         isObjectAction                 = 1                                               second_order                   = 0                                               skip_partitioning              = 0                                               type                           = FileMesh                                        uniform_refine                 = 0                                               allow_renumbering              = 1                                               centroid_partitioner_direction = INVALID                                         construct_node_list_from_side_list = 1                                           control_tags                   = INVALID                                         dim                            = 1                                               enable                         = 1                                               final_generator                = INVALID                                         ghosting_patch_size            = INVALID                                         max_leaf_size                  = 10                                              nemesis                        = 0                                               parallel_type                  = DEFAULT                                         partitioner                    = default                                         patch_size                     = 40                                              patch_update_strategy          = never                                                                                                                            [./square]                                                                         inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = GeneratedMeshGenerator                            bias_x                       = 1                                                 bias_y                       = 1                                                 bias_z                       = 1                                                 boundary_id_offset           = 0                                                 boundary_name_prefix         = INVALID                                           control_tags                 = Mesh                                              dim                          = 2                                                 elem_type                    = INVALID                                           enable                       = 1                                                 extra_element_integers       = INVALID                                           gauss_lobatto_grid           = 0                                                 nx                           = 2                                                 ny                           = 2                                                 nz                           = 1                                                 xmax                         = 1                                                 xmin                         = 0                                                 ymax                         = 1                                                 ymin                         = 0                                                 zmax                         = 1                                                 zmin                         = 0                                               [../]                                                                          []                                                                                                                                                                [Mesh]                                                                                                                                                              [./square]                                                                       [../]                                                                          []                                                                                                                                                                [Mesh]                                                                                                                                                              [./square]                                                                       [../]                                                                          []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               controls                       = 0                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = out                                             gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         inactive                       = (no_default)                                    interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         perf_graph                     = 0                                               print_linear_converged_reason  = 1                                               print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_nonlinear_converged_reason = 1                                             print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     = (no_default)                                    tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                               xml                            = 0                                             []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              family                       = LAGRANGE                                          inactive                     = (no_default)                                      isObjectAction               = 1                                                 order                        = FIRST                                             scaling                      = INVALID                                           type                         = MooseVariableBase                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                           block                        = INVALID                                           components                   = 1                                                 control_tags                 = Variables                                         eigen                        = 0                                                 enable                       = 1                                                 fv                           = 0                                                 initial_condition            = INVALID                                           outputs                      = INVALID                                           use_dual                     = 0                                                                                                                                  [./InitialCondition]                                                               inactive                   = (no_default)                                        isObjectAction             = 1                                                   type                       = ConstantIC                                          block                      = INVALID                                             boundary                   = INVALID                                             control_tags               = Variables/u                                         enable                     = 1                                                   ignore_uo_dependency       = 0                                                   value                      = 6.2                                                 variable                   = u                                                 [../]                                                                          [../]                                                                          []                                                                                        @ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@ΜΜΜΜΜΝ@"@"@"@"@"@"@"@"@"?π              ?ΰ     ?ΰ             ?π      ?π      ?ΰ             ?π      @"@"@"@"@"@"@"@"@"