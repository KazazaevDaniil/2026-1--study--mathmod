using DrWatson
@quickactivate "project"

using OrdinaryDiffEq      # лёгкий решатель для ОДУ (альтернатива DifferentialEquations)
using DataFrames
using Tables
using Plots
using LaTeXStrings
using BenchmarkTools

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function sir_ode!(du, u, p, t)
    S, I, R = u
    β, c, γ = p
    N = S + I + R
    @inbounds begin
        du[1] = -β * c * I / N * S   # dS/dt
        du[2] =  β * c * I / N * S - γ * I   # dI/dt
        du[3] =  γ * I                         # dR/dt
    end
    nothing
end

δt = 0.1
tmax = 40.0
tspan = (0.0, tmax)
u0 = [990.0, 10.0, 0.0]   # [S0, I0, R0]
p = [0.05, 10.0, 0.25]     # [β, c, γ]

R0 = (p[2] * p[1]) / p[3]   # R0 = (c * β) / γ

prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), saveat = δt)

df_ode = DataFrame(Tables.table(sol_ode'))
rename!(df_ode, ["S", "I", "R"])
df_ode[!, :t] = sol_ode.t
df_ode[!, :N] = df_ode.S + df_ode.I + df_ode.R   # общая численность (должна быть постоянной)

println("="^60)
println("Модель SIR (трёхпараметрическая)")
println("="^60)
println("Параметры:")
println("β (вероятность заражения при контакте) = ", p[1])
println("c (среднее число контактов в день)     = ", p[2])
println("γ (скорость выздоровления)             = ", p[3])
println("R0 = c·β/γ = ", round(R0, digits=3))
println("Средняя продолжительность болезни = ", round(1/p[3], digits=2), " дней")
println("\nНачальные условия:")
println("S0 = ", u0[1], ", I0 = ", u0[2], ", R0 = ", u0[3])
println()

plt1 = plot(df_ode.t, [df_ode.S df_ode.I df_ode.R],
            label = [L"S(t)" L"I(t)" L"R(t)"],
            xlabel = "Время, дни",
            ylabel = "Количество людей",
            title = "Модель SIR: динамика эпидемии",
            linewidth = 2,
            legend = :right,
            grid = true,
            size = (800, 500))

annotate!(plt1,
          maximum(df_ode.t) * 0.7,
          maximum(df_ode.N) * 0.8,
          text("Параметры:\nβ = $(p[1])\nc = $(p[2])\nγ = $(p[3])\nR₀ = $(round(R0, digits=2))", 8, :left))

savefig(plt1, plotsdir(script_name, "sir_main.png"))

peak_idx = argmax(df_ode.I)
peak_time = df_ode.t[peak_idx]
peak_value = df_ode.I[peak_idx]

plt2 = plot(df_ode.t, df_ode.I,
            label = L"I(t)",
            xlabel = "Время, дни",
            ylabel = "Количество инфицированных",
            title = "Динамика числа заражённых",
            color = :red,
            linewidth = 2,
            fill = (0, 0.3, :red),
            grid = true,
            size = (800, 400))
vline!(plt2, [peak_time], color = :black, linestyle = :dash, linewidth = 1, label = false)
annotate!(plt2, peak_time, peak_value * 1.05,
          text("Пик: $(round(peak_value, digits=1)) на $(round(peak_time, digits=1)) день", 8, :top))

savefig(plt2, plotsdir(script_name, "sir_infected.png"))

plt3 = plot(df_ode.t, df_ode.I,
            label = L"I(t)",
            xlabel = "Время, дни",
            ylabel = "Число инфицированных (лог. масштаб)",
            title = "Экспоненциальный рост (лог. шкала)",
            yscale = :log10,
            color = :red,
            linewidth = 2,
            grid = true,
            size = (800, 400))

savefig(plt3, plotsdir(script_name, "sir_log_scale.png"))

plt4 = plot(df_ode.t, [df_ode.S ./ df_ode.N .* 100,
                       df_ode.I ./ df_ode.N .* 100,
                       df_ode.R ./ df_ode.N .* 100],
            label = [L"S/N (\%)" L"I/N (\%)" L"R/N (\%)"],
            xlabel = "Время, дни",
            ylabel = "Доля популяции, %",
            title = "Динамика эпидемии (в процентах)",
            linewidth = 2,
            legend = :right,
            grid = true,
            size = (800, 500))

if R0 > 1
    herd_threshold = (1 - 1/R0) * 100
    hline!(plt4, [herd_threshold],
           color = :purple, linestyle = :dash,
           label = "Порог коллективного иммунитета ($(round(herd_threshold, digits=1))%)",
           linewidth = 1.5)
end

savefig(plt4, plotsdir(script_name, "sir_percentages.png"))

plt5 = plot(df_ode.S, df_ode.I,
            label = "Фазовая траектория",
            xlabel = L"S(t)",
            ylabel = L"I(t)",
            title = "Фазовый портрет SIR модели",
            color = :blue,
            linewidth = 2,
            grid = true,
            size = (800, 500),
            legend = :topright)

step = 50
for i in 1:step:length(df_ode.S)-step
    plot!(plt5, [df_ode.S[i], df_ode.S[i+step]],
                [df_ode.I[i], df_ode.I[i+step]],
                arrow = :closed, color = :blue, alpha = 0.3, label = false)
end

savefig(plt5, plotsdir(script_name, "sir_phase_portrait.png"))

df_ode[!, :Re] = R0 .* df_ode.S ./ df_ode.N

plt6 = plot(df_ode.t, df_ode.Re,
            label = L"R_e(t)",
            xlabel = "Время, дни",
            ylabel = "Эффективное репродуктивное число",
            title = "Изменение R_e во времени",
            color = :green,
            linewidth = 2,
            grid = true,
            size = (800, 400))
hline!(plt6, [1.0], color = :black, linestyle = :dash, linewidth = 1, label = "R_e = 1")

savefig(plt6, plotsdir(script_name, "sir_effective_R.png"))

println("\nБенчмарк решения ОДУ:")
@benchmark solve(prob_ode, Tsit5(), saveat = δt)

println("\n" * "="^60)
println("АНАЛИЗ РЕЗУЛЬТАТОВ")
println("="^60)
println("Общая численность популяции (контроль): N = ", round(df_ode.N[1], digits=1))
println("Пиковое число заражённых: I_max = ", round(peak_value, digits=1))
println("Время достижения пика: t_peak = ", round(peak_time, digits=1), " дней")
println("Итоговое число переболевших: R(∞) = ", round(df_ode.R[end], digits=1))
println("Доля переболевших: ", round(df_ode.R[end] / df_ode.N[1] * 100, digits=1), "%")

if R0 > 1
    println("\nТеоретический анализ:")
    println("  Порог коллективного иммунитета: ", round((1 - 1/R0) * 100, digits=1), "%")
    println("  Теоретический пик при S/N = 1/R0 = ", round(1/R0, digits=3))
end

println("\nВсе графики сохранены в: ", plotsdir(script_name))
