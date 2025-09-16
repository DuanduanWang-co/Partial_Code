using DifferentialEquations, Plots, CSV, DataFrames

function pacemaker_fhn!(du, u, h, p, t)
    x1, y1, x2, y2, x3, y3, z1, v1, z2, v2, z3, v3, z4, v4 = u
    a1, a2, a3, u11, u12, u21, u22, u31, u32, f1, f2, f3, d1, d2, d3, e1, e2, e3, K_SA_AV, K_AV_HP, τ = p[1:21]
    kk1, kk2, kk3, kk4, c1, c2, c3, c4, d1_ode, d2_ode, d3_ode, d4_ode, h1, h2, h3, h4, g1, g2, g3, g4, w11, w12, w21, w22, w31, w32, w41, w42 = p[22:end]
    KATDe = 4e-5
    KATRe = 4e-5
    KVNDe = 9e-5
    KVNRe = 6e-5

    # Time Delay
    y2_SA_AV = h(p, t - τ)[2]
    y3_AV_HP = h(p, t - τ)[4]

    # VDP Equations
    du[1] = y1
    du[2] = -a1 * y1 * (x1 - u11) * (x1 - u12) - f1 * x1 * (x1 + d1) * (x1 + e1)
    du[3] = y2
    du[4] = -a2 * y2 * (x2 - u21) * (x2 - u22) - f2 * x2 * (x2 + d2) * (x2 + e2) + K_SA_AV * (y2_SA_AV - y2)
    du[5] = y3
    du[6] = -a3 * y3 * (x3 - u31) * (x3 - u32) - f3 * x3 * (x3 + d3) * (x3 + e3) + K_AV_HP * (y3_AV_HP - y3)

    I_AT_De = du[1] < 0 ? 0 : KATDe * du[1]
    I_AT_Re = du[1] < 0 ? -KATRe * du[1] : 0
    I_VN_De = du[5] < 0 ? 0 : KVNDe * du[5]
    I_VN_Re = du[5] < 0 ? -KVNRe * du[5] : 0


    # P wave
    du[7] = kk1 * (-c1 * z1 * (z1 - w11) * (z1 - w12) - d1_ode * v1 * z1 + I_AT_De)
    du[8] = kk1 * h1 * (z1 - g1 * v1)

    # Ta wave
    du[9] = kk2 * (-c2 * z2 * (z2 - w21) * (z2 - w22) - d2_ode * v2 * z2 + I_AT_Re)
    du[10] = kk2 * h2 * (z2 - g2 * v2)

    # QRS wave
    du[11] = kk3 * (-c3 * z3 * (z3 - w31) * (z3 - w32) - d3_ode * v3 * z3 - 0.015 * v3+ I_VN_De)
    du[12] = kk3 * h3 * (z3 - g3 * v3)

    # T wave
    du[13] = kk4 * (-c4 * z4 * (z4 - w41) * (z4 - w42) - d4_ode * v4 * z4 + I_VN_Re)
    du[14] = kk4 * h4 * (z4 - g4 * v4)
end

# Define parameters
a1, a2, a3 = 40, 50, 50
u11, u21, u31 = 0.83, 0.83, 0.83
u12, u22, u32 = -0.83, -0.83, -0.83
f1, f2, f3 = 22, 8.4, 1.5
d1, d2, d3 = 3, 3, 3
e1, e2, e3 = 3.5, 5, 12
K_SA_AV, K_AV_HP = f1, f1
τ = 0.092

# FHN Parameters
kk1 = 2000.0; kk2 = 400.0; kk3 = 1300.0; kk4 = 2000.0
c1, c2, c3, c4 = 0.26, 0.26, 0.12, 0.10
d1_ode, d2_ode, d3_ode, d4_ode = 0.4, 0.4, 0.09, 0.1
h1, h2, h3, h4 = 0.004, 0.004, 0.008, 0.008
g1, g2, g3, g4 = 1.0, 1.0, 1.0, 1.0
w11, w12 = 0.13, 1.0
w21, w22 = 0.19, 1.0
w31, w32 = 0.12, 1.1
w41, w42 = 0.22, 0.8


p = [a1, a2, a3, u11, u12, u21, u22, u31, u32, f1, f2, f3, d1, d2, d3, e1, e2, e3, K_SA_AV, K_AV_HP, τ,
     kk1, kk2, kk3, kk4, c1, c2, c3, c4, d1_ode, d2_ode, d3_ode, d4_ode, h1, h2, h3, h4, g1, g2, g3, g4, w11, w12, w21, w22, w31, w32, w41, w42]

u0_combined = [0.1, 0.1, 0.2, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]
h(p, t) = u0_combined
prob_combined = DDEProblem(pacemaker_fhn!, u0_combined, h, (0.0, 10.0), p, constant_lags=(τ,))

sol_combined = solve(prob_combined, MethodOfSteps(Tsit5()))

SA = sol_combined[1, :]  # SA node potential (x1)
AV = sol_combined[3, :]  # AV node potential (x2)
PKJ = sol_combined[5, :] # PKJ potential (x3)
P_wave = sol_combined[7, :]   # P wave (z1)
Ta_wave = sol_combined[9, :]  # Ta wave (z2)
QRS_wave = sol_combined[11, :] # QRS wave (z3)
T_wave = sol_combined[13, :]  # T wave (z4)

ECG =  P_wave .- Ta_wave .+ QRS_wave .+ T_wave
ATR = P_wave .- Ta_wave 
VTR = QRS_wave .+ T_wave

t = sol_combined.t

# Plot
#plot(t, SA, label="SA Node", xlabel="Time", ylabel="Potential", title="SA Node Potential")
#plot(t, AV, label="AV Node", xlabel="Time", ylabel="Potential", title="AV Node Potential")
plot(t, P_wave, label = "P_wave-ecg")
#plot(t, Ta_wave, label = "Ta_wave")
#plot(t, QRS_wave, label = "QRS_wave")
#plot(t, T_wave, label = "T_wave")

plot(t, PKJ, label="PKJ Potential", xlabel="Time", ylabel="Potential", title="PKJ Potential")
plot(t, ECG, label="ECG", xlabel="Time", ylabel="ECG", title="ECG Signal")
plot(t, ATR, label = "ATR")
plot(t, VTR, label = "VTR")

plot(t, SA, label="SA Node", xlabel="Time", ylabel="Potential", title="SA Node Potential")
plot!(t, AV, label="AV Node", xlabel="Time", ylabel="Potential", title="AV Node Potential")
plot!(t, PKJ, label="PKJ Potential", xlabel="Time", ylabel="Potential", title="PKJ Potential")


df = DataFrame(Time = t, SA_Node = SA, AV_Node = AV, PKJ_Node = PKJ)

CSV.write("C:/Users/DELL/Desktop/t=0_2_4.csv", df)

df = DataFrame(Time = t, P = P_wave, Ta = Ta_wave, QRS = QRS_wave, T = T_wave, ECG = ECG)

CSV.write("C:/Users/DELL/Desktop/JuliaWaves.csv", df)

