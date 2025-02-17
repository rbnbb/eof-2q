using Plots
using ColorSchemes

include("../src/eaee.jl")
include("plot-utils.jl")

# using PyPlot: figure
# pyplot()

if !Base.isinteractive() || !(@isdefined bell00p)  # avoid redefining const
    function _pretty_name(first_term::AbstractString, sign::AbstractString)
        second_term = first_term == "01" ? "10" : "11"
        return LaTeXString(
            L"$\frac{\ket{" * first_term * "}" * sign * raw"\ket{" * second_term *
            L"}}{\sqrt{2}}$",
        )
    end
    const bell00p = (
        s=_pretty_name("00", "+"), name="|00>+|11>", v=1 / sqrt(2) * [1.0 + 0im, 0, 0, 1]
    )
    const bell00m = (
        s=_pretty_name("00", "-"), name="|00>-|11>", v=1 / sqrt(2) * [1.0 + 0im, 0, 0, -1]
    )
    const bell01p = (
        s=_pretty_name("01", "+"), name="|01>+|10>", v=1 / sqrt(2) * [0, 1.0 + 0im, 1, 0]
    )
    const bell01m = (
        s=_pretty_name("01", "-"), name="|01>-|10>", v=1 / sqrt(2) * [0, 1.0 + 0im, -1, 0]
    )
end

"""
with pyplot
"""
function quick_heatmaps_for_eaee_n_dU(xs, ys, ees, nus)
    figure(1)
    p1 = heatmap(
        xs ./ pi, ys ./ pi, ees; xlabel=L"\theta/\pi", ylabel=L"\phi/\pi", title="EAEE"
    )
    display(p1)
    figure(2)
    p2 = heatmap(
        xs ./ pi,
        ys ./ pi,
        -nus;
        xlabel=L"\theta/\pi",
        ylabel=L"\phi/\pi",
        title="non-unitarity",
    )
    display(p2)
    return nothing
end

"""
    Compare the depenence on unitary parametrization of non-unitarity and eaee

requires `using Pyplot:figure` and pyplot backend.
"""
function compare_dU_eaee_once(phi=haar_rand_statevec(), noise=AmplitudeDamping(0.1))
    xs, ys, nus, es = compute_nu_ea_x_params(phi, noise)
    return quick_heatmaps_for_eaee_n_dU(xs, ys, es, nus)
end

# function plot_save()
#     pgfplotsx()
#     p = [0.2, 0.2, 0.1, 0.1]
#     d = 5
#     noises = [AmplitudeDamping, PhaseDamping, PhaseFlip, BitFlip]
#     psi = bell00m.v
#     name = bell00m.s
#     my_fname(str) = str[findall(a -> isuppercase(a), str)]
#     for j in eachindex(noises)
#         @show noise = noises[j](p[j])
#         plot()
#         study_quantum_channel(psi, noise, d, 1000; state=name, color_start=2)
#         savefig(my_fname(string(noises[j])) * string(p[j]) * "_n$d.pdf")
#     end
# end

"""
Plot entropy reduction with repeated applications of noise for different unravellings.

you might want to use
julia> plot(); noise= PhaseFlip(0.9); phi = bell00p; study_quantum_channel(phi.v, noise, 5;state=phi.s, color_start=1); phi = bell01p; study_quantum_channel(phi.v, noise, 5; state=phi.s, linestyle=:dashdotdot, color_start=2)
"""
function ef_nu_00_pi4_vs_application_plt(
    psi0::Statevector,
    noise::NoiseChannel,
    depth::Integer,
    numavgs=1;
    state="|ψ>",
    linestyle=:solid,
    color_start=2,
)
    eaees = zeros(depth + 1, 5)
    eaees_squared = zeros(depth + 1, 5)
    for _ in 1:numavgs
        S = compute_eaees_x_application(
            same_ee_state(psi0), noise, [(θ=0.0,), (θ=pi / 4,)], depth
        )
        eaees += S
        eaees_squared += S .^ 2
    end
    eaees = eaees ./ numavgs  # E[X]
    # V = E[X^2] - E[X]^2 = σ^2
    V = (eaees_squared / numavgs - eaees .^ 2)
    rootifnonzero(x) = isapprox(0, x; atol=default_atol()) ? 0 : sqrt(x)
    stds = rootifnonzero.(V)# / sqrt(numavgs)
    @show stds
    cols = palette(:Paired_10)[color_start:2:end]
    lab1 = LaTeXString(raw"$E_f\left(" * raw"\right)$")
    lab4 = L"\theta=0,\phi=0"
    lab5 = L"\theta=\frac{\pi}{4},\phi=0"
    kwargs = (
        title=LaTeXString(
            string(typeof(noise)) * raw" channel with $p=" *
            string(getproperty(noise, fieldnames(typeof(noise))[1])) * raw"$" *
            " for $numavgs rand 2q states",
        ),
        xlabel=raw"# application",
        ylabel=L"\overline{E}",
        titlefontsize=10,
        dpi=400,
        ylim=(-0.04, 1.04),
        legend_position=:bottomleft,
        legend=true,
        linestyle=linestyle,
        label=[lab1 "optimal" "numu" lab4 lab5],
        color=[cols[1] cols[2] cols[3] cols[4] cols[5]],
        linealpha=[1 1 1 1 1],
    )
    p = plot!(eaees[1:2:end, :]; ribbon=stds, kwargs...)
    do_stringy(txts) = reduce(*, (s * "\n" for s in string.(txts)))
    # for j in 1:depth
    #     annotate!(j+1, eaees[j+1,3], Plots.text(do_stringy(good_angles[j]), 6))
    # end
    return display(p)
end

"""
Return entropy of ensembles of pure trajectories for the quantum channel noise.

The return matrix has 2 + 2*length(angles4trajectories) columns.
The first column is the entanglement of formation
The 2nd is the entropy for optimal choice of Kraus operators at each branching
in the pure state trajectory tree, if do_optimal is false it will consist of 0s.
The following columns correspond to the entropy of unravellings  obtained by using,
globally (at each branching), the parametrizations provided for the Kraus
operators.
"""
function compute_eaees_x_application(
    psi0::Statevector, noise::NoiseChannel, angles4trajectories, depth=5; do_optimal=true
)::Matrix
    rho = psi0 * psi0'  # |psi0><psi0|
    # rho-ensembles to work with
    optimal_ensemble = [psi0]
    numu_ensemble = [psi0]
    more_ensembles = [[psi0] for _ in 1:length(angles4trajectories)]
    # first column is E_f, 2nd is optimized, 3rd is trivial, third is 
    eaees = zeros(depth + 1, 3 + length(angles4trajectories))
    Es = kraus_ops(noise)
    # we want to alternate between 1st and 2nd qubit depending on parity of j
    function twoqubitify(j::Integer, Es)
        return isodd(j) ? [kron(E, I(2)) for E in Es] : [kron(I(2), E) for E in Es]
    end
    # good_angles = []
    eaees[1, :] .= E_f(rho)
    for j in 1:depth  # apply noise depth times to psi, alternating qubits
        # density matrix calculation
        rho = apply_quantum_channel(rho, twoqubitify(j, Es))
        eaees[j + 1, 1] = E_f(rho)
        if do_optimal
            optimal_ensemble = apply_quantum_channel_optimally(
                optimal_ensemble, Es, E -> twoqubitify(j, E)
            )
            do_asserts() && @assert rho ≈ sum(v * v' for v in optimal_ensemble)
            eaees[j + 1, 2] = eaee_of(optimal_ensemble)
            numu_ensemble = apply_quantum_channel_numu(
                numu_ensemble, Es, E -> twoqubitify(j, E)
            )
            do_asserts() && @assert rho ≈ sum(v * v' for v in numu_ensemble)
            eaees[j + 1, 3] = eaee_of(numu_ensemble)
        end
        for k in eachindex(angles4trajectories)
            Fs = angles2su2mat(angles4trajectories[k]...) * Es
            more_ensembles[k] = apply_quantum_channel(more_ensembles[k], twoqubitify(j, Fs))
            do_asserts() && @assert rho ≈ sum(v * v' for v in more_ensembles[k])
            eaees[j + 1, 3 + k] = eaee_of(more_ensembles[k])
        end
    end
    return eaees
end

"""
Learn how entropy varies with the unitary freedom of Kraus operators.

The return data is a 3 element Tuple containing
rotation angles θ, phases β and excess ensemble averaged entanglement entropy
(that is E - E_f), respectively.
The quantum channel is applied once to each qubit of the initial state |psi0>.
Command:
xs, ys, zs = compute_eaee_x_params(same_ee_state(bell00m.v), PhaseFlip(0.01)); heatmap(xs./pi, ys./pi, zs; xlabel=L"\\theta/\\pi", ylabel=L"\\phi/\\pi", title="EAEE")
"""
function compute_eaee_x_params(
    psi0::Statevector, noise::NoiseChannel; thetas=0.0:0.025:0.5, betas=0.0:0.025:0.5
)
    function excess_entropy(psi0::Statevector, noise::NoiseChannel, θ::Number, β::Number)
        eaees = compute_eaees_x_application(psi0, noise, ((θ=θ, β=β),), 1; do_optimal=false)
        return eaees[end, 3] - eaees[end, 1]  # substract entanglement of formation
    end
    thetas *= pi
    betas *= pi
    ees = zeros(length(betas), length(thetas))
    for (j, t) in enumerate(thetas), (k, b) in enumerate(betas)
        ees[k, j] = excess_entropy(psi0, noise, t, b)
    end
    return thetas, betas, ees
end

"""
Learn how non-unitarity & entropy vary with the unitary freedom of Kraus operators.

The return data is a 4 element Tuple containing
rotation angles θ, phases β, non-unitarities and excess ensemble averaged entanglement entropy
(that is E - E_f), respectively.
The quantum channel is applied once to each qubit of the initial state |psi0>.
Command:
xs, ys, zs = compute_eaee_x_params(same_ee_state(bell00m.v), PhaseFlip(0.01)); heatmap(xs./pi, ys./pi, zs; xlabel=L"\\theta/\\pi", ylabel=L"\\phi/\\pi", title="EAEE")
"""
function compute_nu_ea_x_params(
    psi0::Statevector, noise::NoiseChannel; thetas=0.0:0.025:0.5, betas=0.0:0.025:0.5
)
    function excess_entropy(psi0::Statevector, noise::NoiseChannel, θ::Number, β::Number)
        eaees = compute_eaees_x_application(psi0, noise, ((θ=θ, β=β),), 1; do_optimal=false)
        return eaees[end, 4] - eaees[end, 1]  # substract entanglement of formation
    end
    thetas *= pi
    betas *= pi
    ees = zeros(length(betas), length(thetas))
    nus = zeros(length(betas), length(thetas))
    Es = kraus_ops(noise)
    for (j, t) in enumerate(thetas), (k, b) in enumerate(betas)
        ees[k, j] = excess_entropy(psi0, noise, t, b)
        nus[k, j] = averaged_non_unitarity(psi0, angles2su2mat(t, b) * Es)
    end
    return thetas, betas, nus, ees
end

function compute_ordered_numu_eaee_x_application(
    psi0::Statevector, noise::NoiseChannel, depth=3
)::Matrix
    rho = psi0 * psi0'  # |psi0><psi0|
    # rho-ensembles to work with
    optimal_ensemble = [psi0]
    numu_ensemble = [psi0]
    orderednumu_ensemble = [psi0]
    # first column is E_f, 2nd is 2-GEO optimized, 3rd is NUMU optimized, 4th is OrderedNUMU
    eaees = zeros(depth + 1, 4)
    Es = kraus_ops(noise)
    # we want to alternate between 1st and 2nd qubit depending on parity of j
    function twoqubitify(j::Integer, Es)
        return isodd(j) ? [kron(E, I(2)) for E in Es] : [kron(I(2), E) for E in Es]
    end
    eaees[1, :] .= E_f(rho)
    for j in 1:depth  # apply noise to each qubit depth times
        # density matrix calculation
        rho = apply_quantum_channel(rho, twoqubitify(1, Es))
        rho = apply_quantum_channel(rho, twoqubitify(2, Es))
        eaees[j + 1, 1] = E_f(rho)
        # 2-GEO optimal ensemble
        optimal_ensemble = apply_quantum_channel_optimally(
            optimal_ensemble, Es, E -> twoqubitify(1, E)
        )
        optimal_ensemble = apply_quantum_channel_optimally(
            optimal_ensemble, Es, E -> twoqubitify(2, E)
        )
        do_asserts() && @assert rho ≈ sum(v * v' for v in optimal_ensemble)
        eaees[j + 1, 2] = eaee_of(optimal_ensemble)
        # using NUMU
        numu_ensemble = apply_quantum_channel_numu(
            numu_ensemble, Es, E -> twoqubitify(1, E)
        )
        numu_ensemble = apply_quantum_channel_numu(
            numu_ensemble, Es, E -> twoqubitify(2, E)
        )
        do_asserts() && @assert rho ≈ sum(v * v' for v in numu_ensemble)
        eaees[j + 1, 3] = eaee_of(numu_ensemble)
        # using OrderedNUMU
        _ordered_numu_copy = copy(orderednumu_ensemble)
        _ordered_numu_copy = apply_quantum_channel_numu(
            _ordered_numu_copy, Es, E -> twoqubitify(1, E)
        )
        _ordered_numu_copy = apply_quantum_channel_numu(
            _ordered_numu_copy, Es, E -> twoqubitify(2, E)
        )
        orderednumu_ensemble = apply_quantum_channel_numu(
            orderednumu_ensemble, Es, E -> twoqubitify(2, E)
        )
        orderednumu_ensemble = apply_quantum_channel_numu(
            orderednumu_ensemble, Es, E -> twoqubitify(1, E)
        )

        do_asserts() && @assert rho ≈ sum(v * v' for v in numu_ensemble)
        eaees[j + 1, 3] = eaee_of(numu_ensemble)
        for k in eachindex(angles4trajectories)
            Fs = angles2su2mat(angles4trajectories[k]...) * Es
            more_ensembles[k] = apply_quantum_channel(more_ensembles[k], twoqubitify(j, Fs))
            do_asserts() && @assert rho ≈ sum(v * v' for v in more_ensembles[k])
            eaees[j + 1, 3 + k] = eaee_of(more_ensembles[k])
        end
    end
    return eaees
end

######### FUNCTIONS THAT PLOT STUFF #####

function find_4x4mat_for_two_site_AD(phi::Statevector)
    ops_1q = kraus_ops(AmplitudeDamping(0.01))
    Es = [kron(E, F) for E in ops_1q for F in ops_1q]
    rho = phi * phi'
    basic_ensemble::Ensemble = [phi]
    rho = apply_quantum_channel(rho, Es)
    basic_ensemble = apply_quantum_channel(basic_ensemble, Es)
    optimal_ensemble::Ensemble = optimal_decomp(rho)
    # do_asserts() && @assert rho ≈ ensemble2rho(optimal_ensemble)
    # do_asserts() && @assert rho ≈ ensemble2rho(basic_ensemble)
    # @show snap.(basic_ensemble)
    # @show snap.(optimal_ensemble)
    U = transformation_matrix(basic_ensemble, optimal_ensemble)
    return (basic_ensemble, optimal_ensemble, U)
    # return U
end

function average_4x4_unitaries(phi0::Statevector, numavgs=100)
    M_sum = zeros(ComplexF64, 4, 4)
    M_std = zeros(ComplexF64, 4, 4)
    for _ in 1:numavgs
        phi = same_ee_state(phi0)
        _, _, U = find_4x4mat_for_two_site_AD(phi)
        M_sum += U
        M_std += U .^ 2
    end
    M_sum = M_sum ./ numavgs
    M_std = M_std ./ numavgs - M_sum
    @info "Average" M_sum
    @info "Standard deviation" M_std
    return M_sum, M_std
end

"""
Compare the depenence on unitary parametrization of non-unitarity and eaee.
"""
function compare_dU_eaee_avg(numstates2avg=10, noise=AmplitudeDamping(0.1))
    phi = haar_rand_statevec()
    ees = Matrix{Float64}[]
    nus = Matrix{Float64}[]
    xs, ys, nu, ee = compute_nu_ea_x_params(phi, noise)
    push!(ees, ee)
    push!(nus, nu)
    for _ in 1:(numstates2avg - 1)
        phi = haar_rand_statevec()
        _, _, nu, ee = compute_nu_ea_x_params(phi, noise)
        push!(ees, ee)
        push!(nus, nu)
    end
    xs = reshape(collect(xs), (1, length(xs)))
    ys = reshape(collect(ys), (1, length(ys)))
    # @info "" data_summary(mean(nus)) data_summary(mean(ees))
    return xs,
    ys, mean(ees), std(ees) ./ sqrt(numstates2avg), mean(nus),
    std(nus) ./ sqrt(numstates2avg)
end

"""
    Make a scatter plot of E_f vs N_U

Each point is a Haar random state with E_f on one axis and the EAEE of the
state obtained by the NUMU method on the x axis. If we find a diagonal line
then we reach entanglement of formation. Allows a quick way to see how close
NUMU gets to best case for this toy 2 qubit case.
"""
function compare_nu_ef(noise::NoiseChannel, numstates=10)
    nus = zeros(Float64, numstates)
    efs = zeros(Float64, numstates)
    for j in 1:numstates
        psi0 = haar_rand_statevec()
        rho = psi0 * psi0'  # |psi0><psi0|
        # rho-ensembles to work with
        optimal_params = []
        nu_params = []
        ef_ensemble = [psi0]  # best, entanglement of formation
        nu_ensemble = [psi0]  # using NUMU method
        Es = kraus_ops(noise)
        @assert size(Es[1]) == (2, 2) "Kraus ops are not 2x2 matrices"
        # apply noise once on 1st qubit
        rho = apply_quantum_channel(rho, twoqubitify(j, Es))  # density matrix calculation
        efs[j] = E_f(rho)
        ef_ensemble = apply_quantum_channel_optimally(
            ef_ensemble,
            Es,
            E -> twoqubitify(j, E);
            save_pars=true,
            good_angles=optimal_params,
        )
        # θ_f, ϕ_f = optimal_params[1][1]
        nu_ensemble = apply_quantum_channel_numu(
            nu_ensemble, Es, E -> twoqubitify(j, E); save_pars=true, good_angles=nu_params
        )
        nus[j] = eaee_of(nu_ensemble)
        # θ_nu, ϕ_nu = nu_params[1][1]
        if do_asserts()
            @assert rho ≈ sum(v * v' for v in ef_ensemble)
            @assert rho ≈ sum(v * v' for v in nu_ensemble)
            @assert efs[j] ≈ eaee_of(ef_ensemble)
        end
    end
    return nus, efs
end

function plt_eaees_x_application(
    psi0::Statevector, noise::NoiseChannel, depth::Integer; plot_path=nothing, state=""
)
    labels = [L"\theta=0,\beta=0";;
        L"\theta=\frac{\pi}{4},\beta=0";;
        "NUMU";;
        "optim";;
        LaTeXString(raw"$E_f$")]
    # first column is canonical unravelling, 2nd is pi/4, 3rd is NUMU, 4th is E_f
    eaees = zeros(depth + 1, 5)
    # define state for each line
    rho = psi0 * conj.(transpose(psi0))  # |psi0><psi0|
    ensemble_std = [psi0]
    ensemble_pi4 = [psi0]
    ensemble_numu = [psi0]
    ensemble_opt = [psi0]
    Es = kraus_ops(noise)
    # we want to alternate between 1st and 2nd qubit depending on parity of j
    function twoqubitify(j::Integer, Es)
        return isodd(j) ? [kron(E, I(2)) for E in Es] : [kron(I(2), E) for E in Es]
    end
    # good_angles = []
    eaees[1, :] .= E_f(rho)  # begin with entanglement entropy of psi0
    for L in 2:(depth + 1)  # apply noise depth times to psi, alternating qubits
        # canonical unraveling
        ensemble_std = apply_quantum_channel(ensemble_std, twoqubitify(L, Es))
        eaees[L, 1] = eaee_of(ensemble_std)
        # pi/4 unraveling
        Fs = angles2su2mat(pi / 4, 0.0) * Es
        ensemble_pi4 = apply_quantum_channel(ensemble_pi4, twoqubitify(L, Fs))
        eaees[L, 2] = eaee_of(ensemble_pi4)
        # NUMU unraveling
        ensemble_numu = apply_quantum_channel_numu(
            ensemble_numu, Es, A -> twoqubitify(L, A)
        )
        eaees[L, 3] = eaee_of(ensemble_numu)
        # optimized unraveling
        ensemble_opt = apply_quantum_channel_optimally(
            ensemble_opt, Es, A -> twoqubitify(L, A)
        )
        eaees[L, 4] = eaee_of(ensemble_opt)
        # entanglement of formation
        rho = apply_quantum_channel(rho, twoqubitify(L, Fs))
        eaees[L, 5] = E_f(rho)
    end
    cols = palette(:tab10)[1:5]
    cols = [cols[4] cols[1] cols[5] cols[2] cols[3]]
    kwargs = (
        title=LaTeXString(
            string(typeof(noise)) * raw" with $p=" *
            string(getproperty(noise, fieldnames(typeof(noise))[1])) * raw"$ for " * state,
        ),
        xlabel=raw"# application",
        ylabel=L"\overline{S}",
        dpi=300,
        ylim=(-0.04, 1.04),
        legend=:bottomleft,
        # legend_position=:top,
        linestyle=[:solid :solid :dash :solid :dot],
        label=reshape([values(labels)...], (1, length(labels))),
        color=cols,
        grid=:dash,
        size=(400, 320),
        titlefontsize=11,
        guidefontsize=10,
        tickfontsize=9,
        legendfontsize=8,
        # linealpha=[1 0.8 0.8 0.8],
    )
    p = plot(eaees; kwargs...)
    do_stringy(txts) = reduce(*, (s * "\n" for s in string.(txts)))
    if !isnothing(plot_path)
        savefig(plot_path)
    end
    return p
end
