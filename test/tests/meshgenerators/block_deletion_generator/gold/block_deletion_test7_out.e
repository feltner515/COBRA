CDF      
      
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes         num_elem      
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1        num_nod_per_el1       num_side_ss1      num_side_ss2      num_side_ss3      num_side_ss4      num_nod_ns1       num_nod_ns2       num_nod_ns3       num_nod_ns4       num_nod_var       num_info  '         api_version       @�
=   version       @�
=   floating_point_word_size            	file_size               int64_status             title         block_deletion_test7_out.e     maximum_name_length                    
time_whole                            k�   	eb_status                             �   eb_prop1               name      ID              �   	ns_status         	                    �   ns_prop1      	         name      ID              �   	ss_status         
                    �   ss_prop1      
         name      ID              �   coordx                      �      	   coordy                      �      	�   eb_names                       $      
�   ns_names      	                 �      
�   ss_names      
                 �      4   
coor_names                         D      �   node_num_map                    `      �   connect1                  	elem_type         QUAD4         �      \   elem_num_map                    8      <   elem_ss1                          t   side_ss1                          �   elem_ss2                          �   side_ss2                          �   elem_ss3                          �   side_ss3                          �   elem_ss4                          �   side_ss4                          �   node_ns1                          �   node_ns2                          �   node_ns3                             node_ns4                             vals_nod_var1                          �      k�   name_nod_var                       $      ,   info_records                      ]X      P                                                         ?�      @       @       ?�      @      @      @      @              ?�              @       @      @      ?�              @       @      @      ?�              @       @      @                      ?�      ?�              ?�              ?�      ?�      @       @       @       @       @       @      @      @      @      @      @      @      @      @      @                                          right                            top                              left                             bottom                           bottom                           right                            top                              left                                                                                                                            	   
                                                                                 	      
               
                              
         
                                                                                                	   
                                       
                                                                                          	                     u                                   ####################                                                             # Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                    -i block_deletion_test7.i### Version Info ###                                   Framework Information:                                                           MOOSE Version:           git commit 0af734d158 on 2018-11-01                     LibMesh Version:         ab2cf97250f90e88b1da3c9fb40931cf46af7ba9                PETSc Version:           3.8.3                                                   Current Time:            Fri Nov  2 15:06:16 2018                                Executable Timestamp:    Thu Nov  1 10:28:19 2018                                                                                                                                                                                                  ### Input File ###                                                                                                                                                []                                                                                 inactive                       =                                                 initial_from_file_timestep     = LATEST                                          initial_from_file_var          = INVALID                                         element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            type                           = GAUSS                                         []                                                                                                                                                                [BCs]                                                                                                                                                               [./top]                                                                            boundary                     = bottom                                            control_tags                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     =                                                   isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = DirichletBC                                       use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           diag_save_in                 = INVALID                                           save_in                      = INVALID                                           seed                         = 0                                                 value                        = 1                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      inactive                       =                                                 isObjectAction                 = 1                                               type                           = Transient                                       abort_on_solve_fail            = 0                                               compute_initial_residual_before_preset_bcs = 0                                   contact_line_search_allowed_lambda_cuts = 2                                      contact_line_search_ltol       = INVALID                                         control_tags                   =                                                 dt                             = 10                                              dtmax                          = 1e+30                                           dtmin                          = 2e-14                                           enable                         = 1                                               end_time                       = 10                                              l_abs_step_tol                 = -1                                              l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         line_search_package            = petsc                                           max_xfem_update                = 4294967295                                      mffd_type                      = wp                                              n_startup_steps                = 0                                               nl_abs_step_tol                = 1e-50                                           nl_abs_tol                     = 1e-50                                           nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 1e-50                                           nl_rel_tol                     = 1e-08                                           no_fe_reinit                   = 0                                               num_steps                      = 4294967295                                      petsc_options                  = INVALID                                         petsc_options_iname            = '-pc_type -pc_hypre_type'                       petsc_options_value            = 'hypre boomeramg'                               picard_abs_tol                 = 1e-50                                           picard_max_its                 = 1                                               picard_rel_tol                 = 1e-08                                           relaxation_factor              = 1                                               relaxed_variables              =                                                 reset_dt                       = 0                                               restart_file_base              =                                                 scheme                         = implicit-euler                                  snesmf_reuse_base              = 1                                               solve_type                     = NEWTON                                          splitting                      = INVALID                                         ss_check_tol                   = 1e-08                                           ss_tmin                        = 0                                               start_time                     = 0                                               steady_state_detection         = 0                                               steady_state_start_time        = 0                                               steady_state_tolerance         = 1e-08                                           time_period_ends               = INVALID                                         time_period_starts             = INVALID                                         time_periods                   = INVALID                                         timestep_tolerance             = 2e-14                                           trans_ss_check                 = 0                                               update_xfem_at_timestep_begin  = 0                                               use_multiapp_dt                = 0                                               verbose                        = 0                                             []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff]                                                                           inactive                     =                                                   isObjectAction               = 1                                                 type                         = Diffusion                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                                                                                                             [./dt]                                                                             inactive                     =                                                   isObjectAction               = 1                                                 type                         = TimeDerivative                                    block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 lumping                      = 0                                                 matrix_tags                  = 'system time'                                     save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = time                                            [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             inactive                       =                                                 displacements                  = INVALID                                         block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         isObjectAction                 = 1                                               second_order                   = 0                                               skip_partitioning              = 0                                               type                           = MeshGeneratorMesh                               uniform_refine                 = 0                                               allow_renumbering              = 1                                               centroid_partitioner_direction = INVALID                                         construct_node_list_from_side_list = 1                                           control_tags                   =                                                 dim                            = 1                                               enable                         = 1                                               ghosting_patch_size            = INVALID                                         max_leaf_size                  = 10                                              nemesis                        = 0                                               parallel_type                  = DEFAULT                                         partitioner                    = default                                         patch_size                     = 40                                              patch_update_strategy          = never                                         []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [MeshGenerators]                                                                                                                                                    [./SubdomainBoundingBox1]                                                          inactive                     =                                                   isObjectAction               = 1                                                 type                         = SubdomainBoundingBoxGenerator                     block_id                     = 1                                                 block_name                   = INVALID                                           bottom_left                  = '(x,y,z)=(       0,        0,        0)'          control_tags                 = MeshGenerators                                    enable                       = 1                                                 input                        = gmg                                               location                     = INSIDE                                            top_right                    = '(x,y,z)=(       1,        1,        1)'        [../]                                                                                                                                                             [./SubdomainBoundingBox2]                                                          inactive                     =                                                   isObjectAction               = 1                                                 type                         = SubdomainBoundingBoxGenerator                     block_id                     = 1                                                 block_name                   = INVALID                                           bottom_left                  = '(x,y,z)=(       2,        2,        0)'          control_tags                 = MeshGenerators                                    enable                       = 1                                                 input                        = SubdomainBoundingBox1                             location                     = INSIDE                                            top_right                    = '(x,y,z)=(       3,        3,        1)'        [../]                                                                                                                                                             [./ed0]                                                                            inactive                     =                                                   isObjectAction               = 1                                                 type                         = BlockDeletionGenerator                            block_id                     = 1                                                 control_tags                 = MeshGenerators                                    enable                       = 1                                                 input                        = SubdomainBoundingBox2                             new_boundary                 = INVALID                                         [../]                                                                                                                                                             [./gmg]                                                                            inactive                     =                                                   isObjectAction               = 1                                                 type                         = GeneratedMeshGenerator                            bias_x                       = 1                                                 bias_y                       = 1                                                 bias_z                       = 1                                                 control_tags                 = MeshGenerators                                    dim                          = 2                                                 elem_type                    = INVALID                                           enable                       = 1                                                 gauss_lobatto_grid           = 0                                                 nx                           = 4                                                 ny                           = 4                                                 nz                           = 1                                                 xmax                         = 4                                                 xmin                         = 0                                                 ymax                         = 4                                                 ymin                         = 0                                                 zmax                         = 1                                                 zmin                         = 0                                               [../]                                                                          []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               controls                       = 0                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = INVALID                                         gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         inactive                       =                                                 interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         perf_graph                     = 0                                               print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     =                                                 tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                             []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              block                        = INVALID                                           eigen                        = 0                                                 family                       = LAGRANGE                                          inactive                     =                                                   initial_condition            = INVALID                                           order                        = FIRST                                             outputs                      = INVALID                                           scaling                      = 1                                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                         [../]                                                                          []                                                                                                                                                                                                                                                                                        @$      ?�      ?�      ?�$Wf��,?��Fi?�      ?��KBա?�      ?�ע'ɮ�?��,�Փ�?�Ȕu]�?�^�K��?��ɴ :?���k�?���Kv��?�xgX.[}?�V9X?�f��*t?�ʦe5[I?�#���?�?ݿe8�`�?�Π���?� �{��?ޱ�9��?�i�u�@