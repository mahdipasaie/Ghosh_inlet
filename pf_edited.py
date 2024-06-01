import fenics as fe
import numpy as np



# Initial Condition :
class InitialConditions_pf(fe.UserExpression):

    def __init__(self,physical_parameters_dict, **kwargs):
        super().__init__(**kwargs)  # Initialize the base class

        self.omega = physical_parameters_dict['omega']
        self.dy = physical_parameters_dict['dy']
        self.y_solid = physical_parameters_dict['y_solid']

    def eval(self, values, x):
        xp = x[0]
        yp = x[1]
        # # Sinusoidal perturbation with an amplitude of 5 * dy
        perturbation_amplitude = 1*self.dy
        perturbation_wavelength = 2*self.dy  # Wavelength remains as dy
        perturbation =   perturbation_amplitude * np.sin(2 * np.pi * xp / perturbation_wavelength)

        if yp < self.y_solid - perturbation_amplitude :  # solid
            values[0] = 1
            values[1] = -1
        elif self.y_solid - perturbation_amplitude  <= yp <= self.y_solid + perturbation_amplitude:  # interface with perturbation
            values[0] = perturbation #random.uniform(-1, 1)
            values[1] = -1
        else:  # liquid
            values[0] = -1
            values[1] = -1

    def value_shape(self):
        return (2,)
    

def define_variables(mesh):
    
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Order parameter Phi
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U: dimensionless solute supersaturation
    P3 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity


    velocity_func_space = fe.FunctionSpace(mesh, P3)
    v_answer_on_pf_mesh = fe.Function(velocity_func_space)

    element = fe.MixedElement( [P1, P2] )

    function_space_pf = fe.FunctionSpace( mesh, element )

    Test_Functions = fe.TestFunctions(function_space_pf)
    test_1_pf, test_2_pf= Test_Functions



    solution_vector_pf = fe.Function(function_space_pf)  
    solution_vector_pf_0 = fe.Function(function_space_pf)  

    phi_answer, u_answer= fe.split(solution_vector_pf)  # Current solution
    phi_prev, u_prev = fe.split(solution_vector_pf_0)  # Previous solution


    num_subs = function_space_pf.num_sub_spaces()
    spaces_pf, maps_pf = [], []
    for i in range(num_subs):
        space_i, map_i = function_space_pf.sub(i).collapse(collapsed_dofs=True)
        spaces_pf.append(space_i)
        maps_pf.append(map_i)

    variables_dict = {
        "velocity_func_space": velocity_func_space,
        "v_answer_on_pf_mesh": v_answer_on_pf_mesh,
        "function_space_pf": function_space_pf,
        "Test_Functions": Test_Functions,
        "test_1_pf": test_1_pf,
        "test_2_pf": test_2_pf,
        "solution_vector_pf": solution_vector_pf,
        "solution_vector_pf_0": solution_vector_pf_0,
        "phi_answer": phi_answer,
        "u_answer": u_answer,
        "phi_prev": phi_prev,
        "u_prev": u_prev,
        "spaces_pf": spaces_pf,
        "maps_pf": maps_pf
    }

    return variables_dict

def calculate_dependent_variables(variables_dict_pf, physical_parameters_dict):

    phi_prev = variables_dict_pf["phi_prev"]


    # physical_parameters_dict
    w0 = physical_parameters_dict['w0']
    ep_4 = physical_parameters_dict['ep_4']

    # Define tolerance for avoiding division by zero errors
    tolerance_d = fe.sqrt(fe.DOLFIN_EPS)  # sqrt(1e-15)

    # Calculate gradient and derivatives for anisotropy function
    grad_phi = fe.grad(phi_prev)
    mgphi = fe.inner(grad_phi, grad_phi)
    dpx = fe.Dx(phi_prev, 0)
    dpy = fe.Dx(phi_prev, 1)
    dpx = fe.variable(dpx)
    dpy = fe.variable(dpy)

    # Normalized derivatives
    nmx = -dpx / fe.sqrt(mgphi)
    nmy = -dpy / fe.sqrt(mgphi)
    norm_phi_4 = nmx**4 + nmy**4

    # Anisotropy function
    a_n = fe.conditional(
        fe.lt(fe.sqrt(mgphi), fe.sqrt(fe.DOLFIN_EPS)),
        fe.Constant(1 - 3 * ep_4),
        1 - 3 * ep_4 + 4 * ep_4 * norm_phi_4
    )

    # Weight function based on anisotropy
    W_n = w0 * a_n

    # Derivatives of weight function w.r.t x and y
    D_w_n_x = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpx))
    D_w_n_y = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpy))

    return  {
        'D_w_n_x': D_w_n_x,
        'D_w_n_y': D_w_n_y,
        'mgphi': mgphi,
        'W_n': W_n
    }

def calculate_equation_1(variables_dict_pf, dependent_var_dict, physical_parameters_dict):

    phi_answer = variables_dict_pf["phi_answer"]
    u_answer = variables_dict_pf["u_answer"]
    phi_prev = variables_dict_pf["phi_prev"]
    v_test = variables_dict_pf["test_1_pf"]
    # dt = physical_parameters_dict['dt']
    dt = physical_parameters_dict["dt"]
    k_eq = physical_parameters_dict['k_eq']
    a1 = physical_parameters_dict['a1']
    w0 = physical_parameters_dict['w0']
    lamda = physical_parameters_dict['lamda']
    vel_answer = variables_dict_pf["v_answer_on_pf_mesh"]
    tau_0 = physical_parameters_dict['tau_0']
    # retrive dependent variables
    d_w_n_x = dependent_var_dict['D_w_n_x']
    d_w_n_y = dependent_var_dict['D_w_n_y']
    mgphi = dependent_var_dict['mgphi']
    w_n = dependent_var_dict['W_n']
    scaling_velocity = physical_parameters_dict['scaling_velocity']

    ##########remember to change this : 

    vel_answer = scaling_velocity* vel_answer


    term4_in = mgphi * w_n * d_w_n_x
    term5_in = mgphi * w_n * d_w_n_y

    term4 = -fe.inner(term4_in, v_test.dx(0)) * fe.dx
    term5 = -fe.inner(term5_in, v_test.dx(1)) * fe.dx

    term3 = -(w_n**2 * fe.inner(fe.grad(phi_answer), fe.grad(v_test))) * fe.dx

    term2 = (
        fe.inner(
            (phi_answer - phi_answer**3) - lamda * u_answer  * (1 - phi_answer**2) ** 2,
            v_test,
        ) * fe.dx
    )

    tau_n = (w_n / w0) ** 2 * tau_0
    term1 = -fe.inner((tau_n) * (phi_answer - phi_prev) / dt, v_test) * fe.dx
    # Advection term due to velocity of the fluid ( Negetive Cause on the LHS of Eq which goes RHS ) 
    term6 = - fe.inner((tau_n) * fe.dot(vel_answer, fe.grad(phi_answer)), v_test) * fe.dx  


    eq1 = term1 + term2 + term3 + term4 + term5 +  term6

    return eq1

def calculate_equation_2(variables_dict_pf, physical_parameters_dict, dependent_var_dict):

    phi_answer = variables_dict_pf["phi_answer"]
    u_answer = variables_dict_pf["u_answer"]
    u_prev = variables_dict_pf["u_prev"]
    phi_prev = variables_dict_pf["phi_prev"]
    q_test = variables_dict_pf["test_2_pf"]

    a1 = physical_parameters_dict['a1']
    # dt = physical_parameters_dict['dt']
    dt = physical_parameters_dict["dt"]
    k_eq = physical_parameters_dict['k_eq']
    vel_answer = variables_dict_pf["v_answer_on_pf_mesh"]
    opk = physical_parameters_dict['opk'](k_eq)
    omk = physical_parameters_dict['omk'](k_eq)
    at = physical_parameters_dict['at']
    lamda = physical_parameters_dict['lamda']
    a2 = physical_parameters_dict['a2']
    D = physical_parameters_dict['D'](a2, lamda)
    scaling_velocity = physical_parameters_dict['scaling_velocity']

    ##########remember to change this : 

    vel_answer = scaling_velocity* vel_answer

    ########################################
    tolerance_d = fe.sqrt(fe.DOLFIN_EPS)  

    grad_phi = fe.grad(phi_answer)
    abs_grad = fe.sqrt(fe.inner(grad_phi, grad_phi))

    norm = fe.conditional(
        fe.lt(abs_grad, tolerance_d), fe.as_vector([0, 0]), grad_phi / abs_grad
    )

    dphidt = (phi_answer - phi_prev) / dt

    term6 = -fe.inner(((opk) / 2 - (omk) * phi_answer / 2) * (u_answer - u_prev) / dt, q_test) * fe.dx
    term7 = -fe.inner(D * (1 - phi_answer) / 2 * fe.grad(u_answer), fe.grad(q_test)) * fe.dx
    # term8 = -at * (1 + (omk) * u_answer) * dphidt * fe.inner(norm, fe.grad(q_test)) * fe.dx
    term9 = (1 + (omk) * u_answer) * dphidt / 2 * q_test * fe.dx

    # Advection Term LHS goes to RHS ( Negative ):  V · {  [ (1 + k - (1 - k)ϕ) / 2]  ∇U -  [(1 + (1 - k)U) / 2]  ∇ϕ } : 
    term10_1 = fe.dot( vel_answer ,( opk - omk * phi_answer ) / 2 * fe.grad(u_answer)  ) # V ·  (1 + k - (1 - k)ϕ) / 2]  ∇U
    term10_2 =  fe.dot( vel_answer, - (1+ omk *u_answer) / 2 * grad_phi ) # -  V ·  [(1 + (1 - k)U) / 2]  ∇ϕ
    term10 = - fe.inner( term10_1 + term10_2, q_test ) * fe.dx # RHS ( Negative )

    # eq2 = term6 + term7 + term8 + term9 + term10
    eq2 = term6 + term7  + term9 + term10 # without anti-trapping term

    return eq2

def define_problem_pf(L, variables_dict_pf, physical_parameters_dict):

    solution_vector_pf = variables_dict_pf["solution_vector_pf"]
    abs_tol_pf = physical_parameters_dict["abs_tol_pf"]
    rel_tol_pf = physical_parameters_dict["rel_tol_pf"]
    linear_solver_pf = physical_parameters_dict['linear_solver_pf']
    nonlinear_solver_pf = physical_parameters_dict['nonlinear_solver_pf']
    preconditioner_pf = physical_parameters_dict['preconditioner_pf']
    maximum_iterations_pf = physical_parameters_dict['maximum_iterations_pf']

    J = fe.derivative(L, solution_vector_pf)  # Compute the Jacobian

    # Define the problem
    problem = fe.NonlinearVariationalProblem(L, solution_vector_pf, J=J)



    solver_pf = fe.NonlinearVariationalSolver(problem)

    solver_parameters = {
        'nonlinear_solver': nonlinear_solver_pf,
        'snes_solver': {
            'linear_solver': linear_solver_pf,
            'report': False,
            "preconditioner": preconditioner_pf,
            'error_on_nonconvergence': False,
            'absolute_tolerance': abs_tol_pf,
            'relative_tolerance': rel_tol_pf,
            'maximum_iterations': maximum_iterations_pf,
        }
    }


    solver_pf.parameters.update(solver_parameters)


    return solver_pf

def update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                                old_solution_vector_0_ns=None, variables_dict_pf= None):
    

    # define new solver after mesh refinement: 
    if old_solution_vector_pf is not None and old_solution_vector_0_pf is not None and old_solution_vector_0_ns is not None:

        variables_dict_pf = define_variables(mesh)

        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]
        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        u_prev = variables_dict_pf["u_prev"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        maps_pf = variables_dict_pf["maps_pf"]

        # Define dependent variables
        dependent_var_dict = calculate_dependent_variables(variables_dict_pf, physical_parameters_dict)

        # interpolate initial condition  after mesh refinement:
        fe.LagrangeInterpolator.interpolate(solution_vector_pf, old_solution_vector_pf)
        fe.LagrangeInterpolator.interpolate(solution_vector_pf_0, old_solution_vector_0_pf)

        # gettting the old solution vector for the Navier-stockes, velocity function on pf mesh:
        u_prev, p_prev= old_solution_vector_0_ns.split(deepcopy=True)
        fe.LagrangeInterpolator.interpolate(vel_answer_on_pf_mesh , u_prev)


        # Calculate equation 1 and 2
        eq1 = calculate_equation_1(variables_dict_pf, dependent_var_dict, physical_parameters_dict)
        eq2 = calculate_equation_2(variables_dict_pf, physical_parameters_dict, dependent_var_dict)

        # Define the combined weak form
        L = eq1 + eq2
        solver_pf = define_problem_pf(L, variables_dict_pf, physical_parameters_dict)

        return_dict = {
            "solver_pf": solver_pf,
            "variables_dict_pf": variables_dict_pf,
            "dependent_var_dict": dependent_var_dict,
            "eq1": eq1,
            "eq2": eq2,
            "L": L,
            "solution_vector_pf": solution_vector_pf,
            "solution_vector_pf_0": solution_vector_pf_0,
            "spaces_pf": spaces_pf,
            "maps_pf": maps_pf,
            "vel_answer_on_pf_mesh": vel_answer_on_pf_mesh,
        }

        return return_dict
    
    # define the initial condition for the first time step:
    if old_solution_vector_pf is None and old_solution_vector_0_pf is  None and old_solution_vector_0_ns is  None and variables_dict_pf is None:

    
        variables_dict_pf = define_variables(mesh)

        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]
        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        maps_pf = variables_dict_pf["maps_pf"]

        # Define dependent variables
        dependent_var_dict = calculate_dependent_variables(variables_dict_pf, physical_parameters_dict)

        # interpolate initial condition  after mesh refinement:
        initial_conditions = InitialConditions_pf(physical_parameters_dict)
        solution_vector_pf_0.interpolate(initial_conditions)
        solution_vector_pf.interpolate(initial_conditions)

        #Initialize velocity function on pf mesh:
        vel_answer_on_pf_mesh.interpolate(fe.Constant((0, 0)))


        # Calculate equation 1 and 2
        eq1 = calculate_equation_1(variables_dict_pf, dependent_var_dict, physical_parameters_dict)
        eq2 = calculate_equation_2(variables_dict_pf, physical_parameters_dict, dependent_var_dict)

        # Define the combined weak form
        L = eq1 + eq2
        solver_pf = define_problem_pf(L, variables_dict_pf, physical_parameters_dict)

        return_dict = {
            "solver_pf": solver_pf,
            "variables_dict_pf": variables_dict_pf,
            "dependent_var_dict": dependent_var_dict,
            "eq1": eq1,
            "eq2": eq2,
            "L": L,
            "solution_vector_pf": solution_vector_pf,
            "solution_vector_pf_0": solution_vector_pf_0,
            "spaces_pf": spaces_pf,
            "maps_pf": maps_pf,
            "vel_answer_on_pf_mesh": vel_answer_on_pf_mesh,
        }

        return return_dict
    
    
    # updte velocity on pf mesh :
    if variables_dict_pf is not None:

        
        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]
        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        u_prev = variables_dict_pf["u_prev"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        maps_pf = variables_dict_pf["maps_pf"]

        # Define dependent variables
        dependent_var_dict = calculate_dependent_variables(variables_dict_pf, physical_parameters_dict)


        #Initialize velocity function on pf mesh:
        u_prev, p_prev= old_solution_vector_0_ns.split(deepcopy=True)
        fe.LagrangeInterpolator.interpolate(vel_answer_on_pf_mesh , u_prev)


        # Calculate equation 1 and 2
        eq1 = calculate_equation_1(variables_dict_pf, dependent_var_dict, physical_parameters_dict)
        eq2 = calculate_equation_2(variables_dict_pf, physical_parameters_dict, dependent_var_dict)

        # Define the combined weak form
        L = eq1 + eq2
        solver_pf = define_problem_pf(L, variables_dict_pf, physical_parameters_dict)

        return_dict = {
            "solver_pf": solver_pf,
            "variables_dict_pf": variables_dict_pf,
            "dependent_var_dict": dependent_var_dict,
            "eq1": eq1,
            "eq2": eq2,
            "L": L,
            "solution_vector_pf": solution_vector_pf,
            "solution_vector_pf_0": solution_vector_pf_0,
            "spaces_pf": spaces_pf,
            "maps_pf": maps_pf,
            "vel_answer_on_pf_mesh": vel_answer_on_pf_mesh,
        }

        return return_dict
    



