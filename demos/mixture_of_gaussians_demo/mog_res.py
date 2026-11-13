import matplotlib
from matplotlib.colors import to_rgba
from .mog_main import *
from .mog_mdn import n_bootstrap_iter


def gather_results_for_mdn_svi():

    # true posterior
    true_posterior = calc_posterior()

    # prior proposal
    _, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    kl_prior_prop, _ = true_posterior.kl(prior_proposal, n_samples=10**6)

    # mdn trained with proposal
    _, approx_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    kl_mdn_prop, _ = true_posterior.kl(approx_posterior, n_samples=10**6)

    # number of simulations
    n_sims_prior_prop = n_bootstrap_iter * 200
    n_sims_mdn_prop = n_sims_prior_prop + 1000

    helper.save((kl_prior_prop, kl_mdn_prop, n_sims_prior_prop, n_sims_mdn_prop), plotsdir + 'mdn_svi_results.pkl')



##################################################################################
# PLOTTING FUNCTIONS
##################################################################################

def plot_results():

    kl_prior_prop, kl_mdn_prop, n_sims_prior_prop, n_sims_mdn_prop = helper.load(plotsdir + 'mdn_svi_results.pkl')

    # plot KL Divergence vs number of simulations
    fig, ax = plt.subplots(1, 1)
    ax.grid()
    l1 = ax.plot(kl_prior_prop, n_sims_prior_prop, 'o', label='MDN proposal prior')
    l2 = ax.plot(kl_mdn_prop, n_sims_mdn_prop, 'o', label='MDN trained with proposal')
    c1, c2 = l1[0].get_color(), l2[0].get_color()
    ax.annotate(f'KL: {kl_prior_prop:.3f}', (kl_prior_prop, n_sims_prior_prop), textcoords='offset points', xytext=(6, 6), fontweight='bold', color=c1)
    ax.annotate(f'KL: {kl_mdn_prop:.3f}', (kl_mdn_prop, n_sims_mdn_prop), textcoords='offset points', xytext=(6, 6), fontweight='bold', color=c2)

    ax.set_xlabel('KL Divergence')
    ax.set_ylabel('Number of Simulations')
    ax.set_title('KL Divergence vs Number of Simulations')
    plt.legend(loc='best')
    plt.show(block=True)


def plot_learnt_posteriors(save=False, debug=False):
    fontsize=10
    savepath = '../demos/mixture_of_gaussians_demo/results/figs/mog/'
    kl_prior_prop, kl_mdn_prop, _, _ = helper.load(plotsdir + 'mdn_svi_results.pkl')
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rc('text', usetex=False)

    # PLOT 1: Posterior Distributions
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.subplots_adjust(bottom=0.38) 
    xx = np.linspace(disp_lims[0], disp_lims[1], 1000)

    # true posterior
    true_posterior = calc_posterior()
    pp = true_posterior.eval(xx[:, np.newaxis], log=False)
    m_true, S_true = true_posterior.calc_mean_and_cov()
    ax.plot(xx, pp, label='True posterior')

    # proposal prior
    _, _, proposal_prior, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    pp = proposal_prior.eval(xx[:, np.newaxis], log=False)
    m_proposal_prior, S_proposal_prior = proposal_prior.m, proposal_prior.S
    ax.plot(xx, pp, label='Proposal prior')

    # mdn with proposal
    _, snpea_posterior = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    pp = snpea_posterior.eval(xx[:, np.newaxis], log=False)
    m_snpea, S_snpea = snpea_posterior.calc_mean_and_cov()
    ax.plot(xx, pp, label='MDN with proposal')

    # PLot table data
    table_col = ('Mean', 'Variance', 'KL(P||Q)')
    table_row = ['True Posterior', 'Proposal Prior', 'MDN with Proposal (SNPE-A)']
    stats = [(m_true, S_true, '-'), (m_proposal_prior, S_proposal_prior, kl_prior_prop), (m_snpea, S_snpea, kl_mdn_prop)]
    ax.grid(True)


    # Helper functions 
    def fmt(v):
        try:
            return '{:.3f}'.format(float(np.squeeze(v)))
        except (ValueError, TypeError):
            return ''
    
    cell_text = [[fmt(m), fmt(S), fmt(kl)] for m, S, kl in stats]
    colors = [line.get_color() for line in ax.get_lines()[:3]]
    row_colors = [to_rgba(c, 0.35) for c in colors]

    ax_tab = fig.add_axes([0.0, 0.03, 1.0, 0.22])
    ax_tab.axis('off')

    w, h = 0.3, 1.0  
    tbl = ax_tab.table(cellText=cell_text, rowLabels=table_row, colLabels=table_col, rowColours=row_colors, cellLoc='center', bbox=[(1 - (w-0.2)) / 2, (1 - (h-0.4)) / 2, w, h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    ax.set_xlim(disp_lims)
    ax.set_ylim([0.0, 4.0])
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Probability Density')
    
    if save: fig.savefig(savepath + 'posteriors.pdf')


    # PLOT 2: Conditional Density 
    # grids for 2d plotting
    xx = np.linspace(-15.0, 15.0, 1000)
    yy = np.linspace(mu_a - 2.0, mu_b + 2.0, 1000)
    X, Y = np.meshgrid(xx, yy)
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)
    levels = [0.75, 0.99]
    
    fig, ax = plt.subplots(1, 1)
    fig.canvas.setWindowTitle('Conditional Density Estimation with MDN-SVI trained with Proposal')

    net, _ = helper.load(netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl')
    Z = np.exp(net.eval([xy[:, 0:1], xy[:, 1:2]])).reshape(list(X.shape))
    a, ms, _ = net.eval_comps(xx[:, np.newaxis])
    cln = ax.contour(X, Y, helper.probs2contours(Z, levels), levels=levels, colors=['Red', 'Green'])        # Contour lines: The contour set (75% and 99% of mass)
    artists, _ = cln.legend_elements()
                       
    mm = np.sum(np.concatenate(ms, axis=1) * a, axis=1)             # The mean of the mixture at each x as an array of numbers
    mln = ax.plot(xx, mm, 'b')                                      # Mean line: The mean of the mixture at each x as a line
    vln = ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], color='k', linestyle='--')   # The dashed vertical line at x_obs = 0
    
    legend_handles = [mln[0], artists[0], artists[1], vln]
  
    ax.legend(
                legend_handles,
                ['Mean', r'$75\%$ of mass', r'$99\%$ of mass', r'$x_o=0$'],
                loc='upper left',
                labelspacing=0.4,
                handletextpad=0.1,
                borderaxespad=0.3,
                handlelength=2.0,
                fontsize=fontsize
            )

    ax.set_xlabel(r'$x$', labelpad=-1.0)
    ax.set_ylabel(r'$\theta$', labelpad=-15.0)
    ax.grid()
    if save: fig.savefig(savepath + 'mdn_proposal.pdf')

    # TEST 
    if debug:
        contour_values = helper.probs2contours(Z, levels)
        colors = ['Red', 'Green']

        vals, counts = np.unique(contour_values, return_counts=True)
        print(vals)     # distinct values present
        print(counts)   # number of grid cells holding each value

        for lev, col in zip(levels, colors):
            inside = (contour_values <= lev).astype(float)

            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(f"Contour Level {lev}")

            # Plot just this single level on the new axis
            ax.contour(X, Y, inside, levels=[0.5], colors=col)
            ax.set_title(f"Probability Boundary: {lev}")

    plt.show(block=True)


if '__main__' == __name__:

    # gather_results_for_mdn_svi()

    plot_results()
    plot_learnt_posteriors(save=False) # The probability density plot is not correct. Somehow need to fix this.