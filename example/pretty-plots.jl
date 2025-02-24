using CairoMakie
using LaTeXStrings
using ColorSchemes

include("../src/eaee.jl")

figout(s::AbstractString) = homedir() * "/physics/data/eof-2q/" * s

function bell_states()
    function _pretty_name(first_term::AbstractString, sign::AbstractString)
        second_term = first_term == "01" ? "10" : "11"
        return latexstring(
            raw"$\frac{|" * first_term * raw"\rangle" * sign * raw"|" * second_term *
            raw"\rangle}{\sqrt{2}}$",
        )
    end
    bell00p = (
        s=_pretty_name("00", "+"), name="|00>+|11>", v=1 / sqrt(2) * [1.0 + 0im, 0, 0, 1]
    )
    bell00m = (
        s=_pretty_name("00", "-"), name="|00>-|11>", v=1 / sqrt(2) * [1.0 + 0im, 0, 0, -1]
    )
    bell01p = (
        s=_pretty_name("01", "+"), name="|01>+|10>", v=1 / sqrt(2) * [0, 1.0 + 0im, 1, 0]
    )
    bell01m = (
        s=_pretty_name("01", "-"), name="|01>-|10>", v=1 / sqrt(2) * [0, 1.0 + 0im, -1, 0]
    )
    return (bell01m=bell01m, bell01p=bell01p, bell00m=bell00m, bell00p=bell00p)
end

function add_eaees_x_application!(ax, psi0::Statevector, noise::NoiseChannel, depth::Integer)
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
    cols = ColorSchemes.tab10[1:5]
    cols = [cols[4] cols[1] cols[5] cols[2] cols[3]]
    mshapes = [:circle, :dtriangle, :x, :hline, :vline, :utriangle ]
    labels = [L"\theta=0,\phi=0";;
        L"\theta=\frac{\pi}{4}, \phi=0";;
        "NUMU";;
        "optim";;
        L"$E_f$"]
    for j in axes(eaees, 2)  # columns = datarows
        ys = eaees[:, j]  # data series
        xs = 0:length(ys)-1
        lines!(ax, xs, ys; color=cols[j], label=labels[j], alpha=0.7)
        scatter!(ax, xs, ys; marker=mshapes[j], color=cols[j], markersize=9)
    end
    return nothing
end

function multipanel_bell_eaee_x_application()
    # fname = "eaeeX#_2q_$(state.name)_$noise.png"

    fig = Figure(;size=(672, 504))
    belles = bell_states()
    # belles = [belles.bell01m, belles.bell00m]
    gl = fig[1,1] = GridLayout()
    for (j, state) in enumerate(belles)
        psi = state.v  # current state ket
        # Row #1
        ax_pf = Axis(gl[j,1];)
        noise = PhaseFlip(0.08)
        title=latexstring( string(typeof(noise)) * raw" with $p=" * string(getproperty(noise, fieldnames(typeof(noise))[1])) * raw"$")
        Label(gl[0, 1], text=title, tellwidth=false)
        # fname = "eaeeX#_2q_$(state.name)_$noise.png"
        add_eaees_x_application!(ax_pf, psi, noise, 11)

        # Row #2
        ax_ad = Axis(gl[j,2];)
        noise = AmplitudeDamping(0.2)
        title=latexstring(string(typeof(noise)) * raw" with $p=" * string(getproperty(noise, fieldnames(typeof(noise))[1])) * raw"$")
        Label(gl[0, 2], text=title, tellwidth=false)
        add_eaees_x_application!(ax_ad, psi, noise, 11)
        Label(gl[j:j,0], text=state.s, rotation = pi/2, tellheight=false)  # name of bell state
        if j == 1
            hidexdecorations!(ax_pf; grid=false, ticks=false)
            hidexdecorations!(ax_ad; grid=false, ticks=false)
        else
            linkxaxes!(ax_pf, content(gl[1,1]))
            linkxaxes!(ax_ad, content(gl[1,2]))
        end
    end
    Legend(gl[3,1:2], content(gl[1,1]); orientation=:horizontal, tellheight=true)
    # savefig(p, plot_dir * "bell_states_eaee.png")
    return fig
end
