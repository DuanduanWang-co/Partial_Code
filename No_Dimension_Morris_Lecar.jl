using DifferentialEquations
using Plots

function m_inf(V, V1, V2)
    0.5 * (1 + tanh((V - V1) / V2))
end

function w_inf(V, V3, V4)
    0.5 * (1 + tanh((V - V3) / V4))
end

function tau_w(V, phi, V3, V4)
    1 / (phi * cosh((V - V3) / (2 * V4)))
end

# Define Nernst potential
function nernst_potential(R, T, z, F, K_outside, K_inside)
    1000 * R * T / (z * F) * log(K_outside / K_inside)
end

# Define the Morris-Lecar model
function morris_lecar!(du, u, p, t)
    V, w = u
    C, g_Ca, g_K, g_L, V_L, phi, V1, V2, V3, V4, R, T, F, K_inside, I_ext, K_outside, V_Ca = p
    
    m = m_inf(V, V1, V2)
    w_eq = w_inf(V, V3, V4)
    tau = tau_w(V, phi, V3, V4)
    
    I_Ca = g_Ca * m * (V - V_Ca)
    I_K = g_K * w * (V - V_K)
    I_L = g_L * (V - V_L)
    
    du[1] = (I_ext - I_Ca - I_K - I_L) / C  
    du[2] = (w_eq - w) / tau           
end 

# Parameters for Morris-Lecar model
C = 20.0       
g_Ca = 4.0     
g_K = 8.0      
g_L = 2.0      
       
R = 8.314      
T = 310.15   #26.7
F = 96485.33212

RTF= R*T*1000/F # unit is mV
V_L = -60.0/RTF    
phi = 0.04     
V1 = -1.2/RTF      
V2 = 18.0/RTF     
V3 = 2.0/RTF      
V4 = 30.0/RTF
K_outside = 4.5432 
K_inside = 140.0 
Ca_outside = 1.7959
Ca_inside = 0.0001

V_Ca =  140/RTF
V_K = nernst_potential(R, T, 1, F, K_outside, K_inside)/RTF
V0 = -60.0/RTF
I_ext = 100/RTF

# Define parameters
p = [C, g_Ca, g_K, g_L, V_L, phi, V1, V2, V3, V4, R, T, F, K_inside, I_ext, K_outside, V_Ca, V_K]

# Solve the ODE problem
prob = ODEProblem(morris_lecar!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the results
plot(sol.t, sol[1, :], xlabel="Time", ylabel="Membrane Potential (V)", title="Morris-Lecar Model (I_ext=100/24.5, K_outside=4.5432)", label="V(t)")

