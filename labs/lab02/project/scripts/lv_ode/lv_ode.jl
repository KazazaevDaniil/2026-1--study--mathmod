using DrWatson
@quickactivate "project"

using OrdinaryDiffEq
using DataFrames
using Plots
using LaTeXStrings
using Statistics
using FFTW
using BenchmarkTools

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    @inbounds begin
        du[1] = α * x - β * x * y
        du[2] = δ * x * y - γ * y
    end
    nothing
end

p_lv = [0.1, 0.02, 0.01, 0.3]   # [α, β, δ, γ]
u0_lv = [40.0, 9.0]              # [x₀, y₀]
tspan_lv = (0.0, 200.0)
dt_save = 0.1

prob_lv = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, p_lv)
sol_lv = solve(prob_lv, Tsit5(), saveat = dt_save, abstol = 1e-10, reltol = 1e-8)

df_lv = DataFrame()
df_lv[!, :t] = sol_lv.t
df_lv[!, :prey] = [u[1] for u in sol_lv.u]
df_lv[!, :predator] = [u[2] for u in sol_lv.u]

df_lv[!, :dprey_dt] = p_lv[1] .* df_lv.prey - p_lv[2] .* df_lv.prey .* df_lv.predator
df_lv[!, :dpredator_dt] = p_lv[3] .* df_lv.prey .* df_lv.predator - p_lv[4] .* df_lv.predator
df_lv[!, :prey_pct_change] = df_lv.dprey_dt ./ df_lv.prey * 100
df_lv[!, :predator_pct_change] = df_lv.dpredator_dt ./ df_lv.predator * 100

println("="^60)
println("Модель Лотки-Вольтерры (хищник-жертва)")
println("="^60)
println("\nПараметры модели:")
println("α = ", p_lv[1])
println("β = ", p_lv[2])
println("δ = ", p_lv[3])
println("γ = ", p_lv[4])
println("\nНачальные условия: x₀ = ", u0_lv[1], ", y₀ = ", u0_lv[2])

x_star = p_lv[4] / p_lv[3]
y_star = p_lv[1] / p_lv[2]
println("\nСтационарные точки:")
println("x* = γ/δ = ", round(x_star, digits=3))
println("y* = α/β = ", round(y_star, digits=3))

plt1 = plot(df_lv.t, [df_lv.prey df_lv.predator],
            label = [L"Жертвы (x)" L"Хищники (y)"],
            xlabel = "Время", ylabel = "Популяция",
            title = "Динамика популяций",
            linewidth = 2, legend = :topright, grid = true,
            size = (900, 500), color = [:green :red])
hline!(plt1, [x_star], color = :green, linestyle = :dash, alpha = 0.5, label = L"x^*")
hline!(plt1, [y_star], color = :red, linestyle = :dash, alpha = 0.5, label = L"y^*")
savefig(plt1, plotsdir(script_name, "lv_dynamics.png"))

plt2 = plot(df_lv.prey, df_lv.predator,
            label = "Фазовая траектория",
            xlabel = L"x", ylabel = L"y",
            title = "Фазовый портрет",
            color = :blue, linewidth = 1.5,
            grid = true, size = (800, 600), legend = :topright)

step = 50
for i in 1:step:length(df_lv.prey)-step
    plot!(plt2, [df_lv.prey[i], df_lv.prey[i+step]],
                [df_lv.predator[i], df_lv.predator[i+step]],
                arrow = :closed, color = :blue, alpha = 0.3, label = false)
end

scatter!(plt2, [x_star], [y_star], color = :black, markersize = 8, label = L"(x^*, y^*)")

x_range = LinRange(0, maximum(df_lv.prey) * 1.1, 100)
y_nulcline = p_lv[1] ./ (p_lv[2] .* x_range)
plot!(plt2, x_range, y_nulcline, color = :red, linestyle = :dash, linewidth = 1.5,
      label = "Изоклина хищников (dy/dt=0)")
savefig(plt2, plotsdir(script_name, "lv_phase_portrait.png"))

plt3 = plot(df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt],
            label = [L"dx/dt" L"dy/dt"],
            xlabel = "Время", ylabel = "Скорость изменения",
            title = "Скорости изменения популяций",
            linewidth = 1.5, legend = :topright, grid = true,
            size = (900, 400), color = [:green :red])
savefig(plt3, plotsdir(script_name, "lv_derivatives.png"))

plt4 = plot(df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change],
            label = [L"dx/dt / x (\%)" L"dy/dt / y (\%)"],
            xlabel = "Время", ylabel = "Относительное изменение, %",
            title = "Относительные темпы роста",
            linewidth = 1.5, legend = :topright, grid = true,
            size = (900, 400), color = [:green :red])
savefig(plt4, plotsdir(script_name, "lv_relative_changes.png"))

function compute_fft(signal, dt)
    n = length(signal)
    spectrum = abs.(rfft(signal))
    freq = rfftfreq(n, 1/dt)
    return freq, spectrum
end

freq_prey, spectrum_prey = compute_fft(df_lv.prey .- mean(df_lv.prey), dt_save)
freq_predator, spectrum_predator = compute_fft(df_lv.predator .- mean(df_lv.predator), dt_save)

plt5 = plot(freq_prey, [spectrum_prey spectrum_predator],
            label = [L"Жертвы" L"Хищники"],
            xlabel = "Частота", ylabel = "Амплитуда",
            title = "Спектральный анализ (Фурье)",
            linewidth = 1.5, xscale = :log10, yscale = :log10,
            legend = :topright, grid = true, size = (800, 400),
            color = [:green :red])

if length(spectrum_prey) > 2
    idx_prey = argmax(spectrum_prey[2:end]) + 1
    dominant_freq_prey = freq_prey[idx_prey]
    period_prey = 1 / dominant_freq_prey
    println("\nДоминирующая частота жертв: ", round(dominant_freq_prey, digits=4))
    println("Период колебаний жертв: ", round(period_prey, digits=2))
end
savefig(plt5, plotsdir(script_name, "lv_spectrum.png"))

plt6 = plot(layout = (3,2), size = (1200,900))
plot!(plt6[1], df_lv.t, df_lv.prey, label = L"x(t)", color = :green, lw=2, title="Жертвы")
plot!(plt6[2], df_lv.t, df_lv.predator, label = L"y(t)", color = :red, lw=2, title="Хищники")
plot!(plt6[3], df_lv.prey, df_lv.predator, label=false, color=:blue, lw=1.5, title="Фазовый портрет")
scatter!(plt6[3], [x_star], [y_star], color=:black, markersize=5, label=L"(x^*,y^*)")
plot!(plt6[4], df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt],
      label=[L"dx/dt" L"dy/dt"], color=[:green :red], lw=1.5, title="Скорости")
plot!(plt6[5], freq_prey, spectrum_prey, label=L"x", color=:green, lw=1.5,
      title="Спектр жертв", xscale=:log10, yscale=:log10)
plot!(plt6[6], df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change],
      label=[L"dx/x" L"dy/y"], color=[:green :red], lw=1.5, title="Отн. изменения")
savefig(plt6, plotsdir(script_name, "lv_panel.png"))

println("\nОсновные статистики:")
println("Жертвы: min = ", round(minimum(df_lv.prey), digits=2),
        ", max = ", round(maximum(df_lv.prey), digits=2),
        ", mean = ", round(mean(df_lv.prey), digits=2))
println("Хищники: min = ", round(minimum(df_lv.predator), digits=2),
        ", max = ", round(maximum(df_lv.predator), digits=2),
        ", mean = ", round(mean(df_lv.predator), digits=2))

function find_first_peak(signal, time)
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            return time[i], signal[i]
        end
    end
    return NaN, NaN
end

peak_time_prey, peak_value_prey = find_first_peak(df_lv.prey, df_lv.t)
peak_time_predator, peak_value_predator = find_first_peak(df_lv.predator, df_lv.t)

if !isnan(peak_time_prey) && !isnan(peak_time_predator)
    phase_shift = peak_time_predator - peak_time_prey
    println("\nАнализ колебаний:")
    println("Первый пик жертв: t = ", round(peak_time_prey, digits=2),
            ", x = ", round(peak_value_prey, digits=2))
    println("Первый пик хищников: t = ", round(peak_time_predator, digits=2),
            ", y = ", round(peak_value_predator, digits=2))
    println("Сдвиг фаз (хищники отстают): ", round(phase_shift, digits=2))
end

println("\nБенчмарк решения:")
@benchmark solve(prob_lv, Tsit5(), saveat = dt_save)

println("\nМоделирование завершено успешно!")
