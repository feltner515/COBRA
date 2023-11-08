r1 = 2.0
r2 = 4.0
pi = 3.1415926535

th1_1 = 0.0
th1_2 = 15.0
th1_3 = 30.0
th1_4 = 37.5
th1_5 = 45.0

th2_1 = 0.0
th2_2 = 15.0
th2_3 = 30.0
th2_4 = 40.0
th2_5 = 45.0

z = 0.0

[Mesh]
  [fbpvg]
    type = FillBetweenPointVectorsGenerator
    positions_vector_1 = '${fparse r1*cos(th1_1/180.0*pi)} ${fparse r1*sin(th1_1/180.0*pi)} ${fparse z}
                          ${fparse r1*cos(th1_2/180.0*pi)} ${fparse r1*sin(th1_2/180.0*pi)} 0.0
                          ${fparse r1*cos(th1_3/180.0*pi)} ${fparse r1*sin(th1_3/180.0*pi)} 0.0
                          ${fparse r1*cos(th1_4/180.0*pi)} ${fparse r1*sin(th1_4/180.0*pi)} 0.0
                          ${fparse r1*cos(th1_5/180.0*pi)} ${fparse r1*sin(th1_5/180.0*pi)} 0.0'

    positions_vector_2 = '${fparse r2*cos(th2_1/180.0*pi)} ${fparse r2*sin(th2_1/180.0*pi)} 0.0
                          ${fparse r2*cos(th2_2/180.0*pi)} ${fparse r2*sin(th2_2/180.0*pi)} 0.0
                          ${fparse r2*cos(th2_3/180.0*pi)} ${fparse r2*sin(th2_3/180.0*pi)} 0.0
                          ${fparse r2*cos(th2_4/180.0*pi)} ${fparse r2*sin(th2_4/180.0*pi)} 0.0
                          ${fparse r2*cos(th2_5/180.0*pi)} ${fparse r2*sin(th2_5/180.0*pi)} 0.0'
    num_layers = 3
    input_boundary_1_id = 10
    input_boundary_2_id = 10
    begin_side_boundary_id = 10
    end_side_boundary_id = 10
    use_quad_elements = true
  []
[]
