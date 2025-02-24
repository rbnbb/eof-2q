using JLD2
using Random
using LaTeXStrings
include("eaee.jl")

figname_rand_2q(Nt, noise) = "2q_$(Nt)_$(noise)"

function my_blue_palette()
    col_beg = RGB(173 / 255, 204 / 255, 251 / 255)
    col_end = RGB(38 / 255, 78 / 255, 148 / 255)
    # Create a custom color gradient
    return cgrad([col_beg, col_end])
end

function NU_analytic4pf(args; p=0.1, s=0.1)
    theta, phi = args
    f1 = (1 - p) * cos(theta)^2 + p * sin(theta)^2
    f2 = sqrt(p - p^2) * sin(2 * theta) * cos(2 * phi)
    numerator = f1 - f1^2 + f2^2 + s * f2 - 2 * s * f1 * f2
    denominator = f1 - f1^2 + s * f2 - 2 * s * f1 * f2 - s^2 * f2^2
    return -4 + 4 * numerator / denominator
end

function s_mat_el(psi::Statevector)
    # @assert length(psi) == 4
    sigma_mat = kron([1 0; 0 -1], [1 0; 0 1])
    return realify(psi' * sigma_mat * psi)
end

function _find_avg_s_mat_el(numavgs=100)
    sum = Vector{Float64}(undef, numavgs)
    for j in 1:numavgs
        psi = haar_rand_statevec()
        sum[j] = s_mat_el(psi)
    end
    # data_summary(sum)
    return mean(sum)
end

"""
Nonlinear means average of NU is not NU of average)
"""
function pr_NU4numtraj(pars, Nt)
    Random.seed!(1)
    nus2plot = zeros(size(pars)...)
    for _ in 1:Nt
        s = s_mat_el(haar_rand_statevec())
        nus2plot .+= NU_analytic4pf.(pars; s=s)
    end
    return nus2plot ./ Nt
end

function compute_or_load_nu_ee_data(Nt, noise)
    scratchdir = homedir() * "/data/NoisyQubitMPS/3dplots/"
    fname = figname_rand_2q
    datfile = scratchdir * fname(Nt, noise)
    mkpath(scratchdir)
    # (un)comment to reset catche
    # isfile(datfile) && rm(datfile)
    if isfile(datfile)
        data = load(datfile)
        d = Dict(k => Matrix{Float64}(v) for (k, v) in data)  # type the data
        xs, ys, ees, ees_err, dus, dus_err = d["xs"],
        d["ys"], d["ees"], d["ees_err"], d["dus"],
        d["dus_err"]
    else
        Random.seed!(1)
        xs, ys, ees, ees_err, dus, dus_err = compare_dU_eaee_avg(Nt, noise)
        jldsave(datfile; xs, ys, ees, ees_err, dus, dus_err)
    end
    xs = reshape(xs, (length(xs)))
    ys = reshape(ys, (length(ys)))
    return xs, ys, ees, ees_err, dus, dus_err
end

function old_pr_3d_plot(Nt=200; noise=PhaseFlip(0.1))
    datapath = homedir() * "/physics/data/NoisyQubitMPS/fig/06-24/3d_2q_theta_phi/"
    fname = figname_rand_2q
    xs, ys, ees, ees_err, dus, dus_err = compute_or_load_nu_ee_data(Nt, noise)
    # Old 3d-code
    # dus = sqrt.(dus)
    camera = (90, 20)
    p1 = nice3dplot(xs, ys, ees, ees_err; zlab=L"\overline{S-E_f}", camera)
    savefig(p1, datapath * fname(Nt, noise) * "_$(camera)_ea.pdf")
    p2 = nice3dplot(xs, ys, -dus, dus_err; zlab=L"\overline{d_U}", camera)
    return savefig(p2, datapath * fname(Nt, noise) * "_$(camera)_du.pdf")
end

function pr_du_ee_plot(Nt=200; noise=PhaseFlip(0.1))
    datapath = homedir() * "/physics/data/NoisyQubitMPS/fig/06-24/3d_2q_theta_phi/"
    fname = figname_rand_2q
    xs, ys, ees, ees_err, dus, dus_err = compute_or_load_nu_ee_data(Nt, noise)
    # Two 2D plots
    sel = 1:2:(length(ys) ÷ 2 + 1)
    # numshapes = div(length(sel), 2)
    # markershape = hcat(repeat([:dtriangle], inner=(1, numshapes)),
    #     repeat([:utriangle], inner=(1, length(sel) - numshapes)))
    markershape = repeat([:circ]; inner=(1, length(sel)))
    fs = 10
    default(;
        palette=palette(my_blue_palette(), length(sel)),
        legendfontsize=10,
        tickfontsize=fs,
        guidefontsize=fs,
        labelfontsize=12,
    )
    labels = (x=raw"$\phi$", y=L"\overline{S-E_f}", l=raw"$\theta=")
    # @info "" labels labels.l
    ea_phi = nice2dplt(xs, ees[:, sel], ees_err[:, sel], ys[sel], labels; markershape)
    # savefig(ea_phi, datapath * "o2d" * fname(Nt, noise) * "_ea_x_$(labels.x).pdf")
    labels = (x=raw"$\theta$", y=L"\overline{S-E_f}", l=raw"$\phi=")
    ea_theta = nice2dplt(ys, ees[sel, :]', ees_err[sel, :]', xs[sel], labels; markershape)
    # @info "" size(ees[:, sel])
    # savefig(ea_theta, datapath * "o2d" * fname(Nt, noise) * "_ea_x_$(labels.x).pdf")
    labels = (x=raw"$\phi$", y=L"-\overline{\mathcal{N}_U}", l=raw"$\theta=")
    nu_phi = nice2dplt(xs, -dus[:, sel], dus_err[:, sel], ys[sel], labels; markershape)
    if isa(noise, PhaseFlip)
        long_xs = range(min(xs...), max(xs...), 100)
        nu_args = [(theta, phi) for phi in long_xs, theta in ys[sel]]
        color = hcat(palette(:blues, length(ys[sel]))...)
        # plot!(nu_phi, long_xs, -pr_NU4numtraj(nu_args, Nt); linestyle=:solid, linewidth=1, color)
    end
    # savefig(nu_phi, datapath * "o2d" * fname(Nt, noise) * "_dU_x_$(labels.x).pdf")
    labels = (x=raw"$\theta$", y=L"-\overline{\mathcal{N}_U}", l=raw"$\phi=")
    nu_theta = nice2dplt(ys, -dus[sel, :]', dus_err[sel, :]', ys[sel], labels; markershape)
    if isa(noise, PhaseFlip)
        long_xs = range(min(xs...), max(xs...), 100)
        nu_args = [(theta, phi) for theta in long_xs, phi in ys[sel]]
        color = hcat(palette(:greys, length(ys[sel]))...)
        # @info "" size(dus[sel, :]) size(long_xs) size(nu_args) nu_args[1, :]
        # plot!(nu_theta, long_xs, -NU_analytic4pf.(nu_args); linestyle=:solid, linewidth=1, color)
    end
    # savefig(nu_theta, datapath * "o2d" * fname(Nt, noise) * "_dU_x_$(labels.x).pdf")
    # ps = [ea_theta, nu_theta, ea_phi, nu_phi]
    p = plot(ea_theta, ea_phi, nu_theta, nu_phi; layout=(2, 2), legend=false)
    display(p)
    return savefig(p, datapath * "rand$(Nt)_$(noise)_multi.pdf")
end

function nice2dplt(xs, ys, yerrs, ls, names; markershape=:circle)
    p1 = plot()
    pretty_name(x) = names.l * string(round(x / pi; digits=2)) * raw"\pi $"
    label = reshape([pretty_name(t) for t in ls], (1, length(ls)))
    kwargs = (
        markersize=2,
        msc=:grey,
        markeralpha=1,
        legend=:outerright,
        legend_column=1,
        size=(540, 380),
        grid=:false,
    )
    scatter!(
        xs,
        ys;
        yerror=yerrs,
        xtick=pitick(0, pi / 2, 8; mode=:text),
        label,
        markershape,
        kwargs...,
    )
    xlabel!(names.x)
    ylabel!(names.y)
    return p1
end

function quick_plot_n_merge()
    numavgs = 100000
    pr_du_ee_plot(numavgs; noise=PhaseFlip(0.1))
    pr_du_ee_plot(numavgs; noise=AmplitudeDamping(0.1))
    datapath = homedir() * "/physics/data/numu-1/pdfs/pr-repro/"
    scriptfile = datapath * "merge_ad_pf.sh"
    run(`$scriptfile  $numavgs`)
    return "all good"
end

function nice3dplot(xs, ys, zs, zerrs; zlab="", camera=(20, 20))
    p1 = plot3d()
    # plot!(xs, ys, zs; st=:wireframe, linewidth=0.5, color=:red, fillalpha=0.5)
    cols = reshape(ColorSchemes.get(ColorSchemes.thermal, zs, :extrema), length(zs), 1)
    @show size(cols)
    @show "NaNs? $(any(isnan.(zs)))"
    # z_errs = zs./10
    zerr = reshape(zerrs, length(zerrs))
    scatter!(
        surface2points(xs, ys, zs)...;
        xtick=pitick(0, pi / 2, 4; mode=:text),
        ytick=pitick(0, pi / 2, 4; mode=:text),
        zerr,
        ms=2,
        color=cols,
        msc=:grey,
        grid=false,
    )
    xlabel!(L"\theta")
    ylabel!(L"\phi")
    zlabel!(zlab)
    plot!(; camera, legend=false)
    return p1
end

function comprehensive_comparison_channels()
    for noise in [AmplitudeDamping]
    end
end
