import os
os.environ["PYTENSOR_FLAGS"] = "exception_verbosity=high,optimizer=fast_compile"

import util.mdn as mdn
import util.LossFunction as lf
import util.Trainer as Trainer
from .mog_main import *

n_bootstrap_iter = 4


##################################################################################
# NETWORK TRAINING 
##################################################################################

def train_proposal_prior_with_bootstrapping(save=True):
    """Trains an MDN-SVI to return the posterior with bootstrapping."""
    
    n_samples = 200

    # create an mdn
    net = mdn.MDN_SVI(
            n_inputs=1, 
            n_hiddens=[20], 
            act_fun='tanh', 
            n_outputs=1, 
            n_components=1
        )
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior_proposal = None

    for iter in range(n_bootstrap_iter):

        # generate new data
        ms = np.empty(n_samples)
        xs = np.empty(n_samples)
        dist = np.empty(n_samples)

        for i in range(n_samples):

            ms[i] = mu_a - 1.0
            while ms[i] < mu_a or ms[i] > mu_b:
                if iter == 0:
                    sample = sim_prior()
                else:
                    sample = prior_proposal.gen()

                sample = np.asarray(sample).squeeze()
                ms[i] = sample.item()
                              
            xs[i] = sim_likelihood(ms[i])
            dist[i] = calc_dist(xs[i], x_obs)

            print('simulation {0}, distance = {1}'.format(i, dist[i]))

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_samples)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 12.0])
        plt.show(block=True)

        # train an mdn to give the posterior
        minibatch = 50
        maxiter = int(1000 * n_samples / minibatch)
        monitor_every = 100
        trainer = Trainer.Trainer(
            model=net,
            trn_data=[xs[:, np.newaxis], ms[:, np.newaxis]],
            trn_loss = net.mlprob,
            trn_target=net.y
        )
        trainer.train(
            maxiter=maxiter,
            minibatch=minibatch,
            show_progress=True,
            monitor_every=monitor_every
        )

        # calculate the approximate posterior
        mdn_mog = net.get_mog(np.asarray([x_obs]), n_samples=None)
        mdn_mog.prune_negligible_components(1.0e-3)
        approx_posterior = mdn_mog if iter == 0 else mdn_mog / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        # save the net and the approximate posterior
        if save:
            helper.save((net, approx_posterior, prior_proposal, dist), netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))


def train_mdn_with_proposal(save=True):
    """
    SNPE-A: Uses the prior proposal learnt by bootstrapping to train an mdn.
    """

    # load prior proposal
    net, _, prior_proposal, _ = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(n_bootstrap_iter-1))
    net = mdn.replicate_gaussian_mdn(net, 2)

    # generate data
    n_samples = 1000
    ms = prior_proposal.gen(n_samples)
    xs = sim_likelihood(ms[:, 0])[:, np.newaxis]

    # train an mdn to give the posterior
    minibatch = 100
    maxiter = int(10000 * n_samples / minibatch)
    monitor_every = 1000
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[xs, ms],
        trn_loss=net.mlprob + regularizer / n_samples,
        trn_target=net.y
    )
    trainer.train(
        maxiter=maxiter,
        minibatch=minibatch,
        show_progress=True,
        monitor_every=monitor_every
    )

    # calculate the approximate posterior
    mdn_mog = net.get_mog(np.asarray([x_obs]))
    mdn_mog.prune_negligible_components(1.0e-3)
    approx_posterior = mdn_mog / prior_proposal

    # save the net
    if save:
        filename = netsdir + 'mdn_svi_proposal_hiddens_20_tanh_sims_1k.pkl'
        helper.save((net, approx_posterior), filename)



##################################################################################
# PLOTTING FUNCTIONS
##################################################################################

def show_mdn_posterior(filename):
    """Shows the posterior learnt by the mdn saved in the given file."""

    # load mdn
    data = helper.load(netsdir + filename + '.pkl')
    net = data[0] if isinstance(data, tuple) else data


    # PLOT 1: Learnt posterior
    learnt_posterior = net.get_mog(np.asarray([x_obs]))
    m, S = learnt_posterior.calc_mean_and_cov()

    title = f'Learnt Posterior (MDN)\n Mean = {m.item():.4f}, Variance = {S[0][0]:.4f}'
    helper.plot_pdf_marginals(pdf=learnt_posterior, lims=disp_lims, title=title)

    # PLOT 2: Learnt posterior components
    if learnt_posterior.n_components > 1:
        for a, x in zip(learnt_posterior.a, learnt_posterior.xs):
            title = f'Learnt Posterior Component (MDN), a = {a:.2f}'
            helper.plot_pdf_marginals(pdf=x, lims=disp_lims, title=title)


    # grids for plotting
    xx = np.linspace(-12.0, 12.0, 100)
    yy = np.linspace(-10.0, 10.0, 100)
    X, Y = np.meshgrid(xx, yy)
    xy = np.stack([X.flatten(), Y.flatten()], axis=1)

    # PLOT 3: MDN's conditional probability density
    fig, ax = plt.subplots(1, 1)
    Z = np.exp(net.eval([xy[:, 0:1], xy[:, 1:2]])).reshape(list(X.shape))
    ax.grid()
    ax.contour(X, Y, Z)
    ax.set_xlabel('Likelihood, x')
    ax.set_ylabel('Prior, m')
    ax.set_title('MDN Conditional Probability Density, p(m|x)')
    fig.canvas.manager.set_window_title('PLOT 3')

    # show mdn's parameters
    a, m, s = net.eval_comps(xx[:, np.newaxis])
    s = list(map(lambda t: np.sqrt(1.0 / t[:, :, 0]), s))
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0.0, 1.0, net.n_components)]

    # PLOT 4: MDN's mixing coefficients
    fig, ax = plt.subplots(1, 1)
    ax.grid()
    for i in range(net.n_components):
        ax.plot(xx, a[:, i], color=cols[i], label='component {0}'.format(i))
    ax.set_ylim([-0.2, 1.2])
    ax.set_xlabel('x')
    ax.set_ylabel('mixing coefficient')
    ax.set_title('MDN Mixing Coefficients, a(x)')
    fig.canvas.manager.set_window_title('PLOT 4')
    ax.legend()

    # PLOT 5: MDN's means and standard deviations
    fig, axs = plt.subplots(1, net.n_components, sharey=True)
    axs = [axs] if net.n_components == 1 else axs
    for i in range(net.n_components):
        axs[i].grid()
        axs[i].plot(xx, m[i], '-', color=cols[i], label='m')
        axs[i].plot(xx, m[i] + s[i], ':', color=cols[i], label='m +/- s')
        axs[i].plot(xx, m[i] - s[i], ':', color=cols[i])
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('m')
        axs[i].set_title('component {0}'.format(i))
        axs[i].legend()
    fig.canvas.manager.set_window_title('PLOT 5')

    plt.show(block=True)




def show_mdn_with_proposal_posterior(filename, lims=disp_lims):
    """
    Shows the posterior learnt by the mdn saved in the given file.
    """

    _, ax = plt.subplots(1, 1)
    xx = np.linspace(lims[0], lims[1], 200)
    
    # show true posterior
    true_posterior = calc_posterior()
    true_m, true_S = true_posterior.calc_mean_and_cov()

    # show mdn's posterior
    _, learnt_posterior = helper.load(netsdir + filename + '.pkl')
    learnt_m, learnt_S = learnt_posterior.calc_mean_and_cov() 

    pp_true = true_posterior.eval(xx[:, np.newaxis], log=False)
    pp_learnt = learnt_posterior.eval(xx[:, np.newaxis], log=False)

    ax.plot(xx, pp_true, label='True posterior')
    ax.plot(xx, pp_learnt, label='Learnt posterior (SNPE-A)', linestyle='-', color='tab:orange')
    ax.text(xx[-1] - 3, pp_true[-1] + 1.4, f'True Posterior Mean: {true_m.item():.4f} \nTrue Posterior Variance: {true_S[0][0]:.4f}', color='tab:blue', va='center')
    ax.text(xx[-1] - 3, pp_learnt[-1] + 1, f'Learnt Posterior Mean: {learnt_m.item():.4f} \nLearnt Posterior Variance: {learnt_S[0][0]:.4f}', color='tab:orange', va='center')

    ax.set_xlim(lims)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Probability density')
    ax.set_title('True vs Learnt Posterior')
    ax.legend()
    ax.grid(True)
    plt.show()

    if learnt_posterior.n_components > 1:
        for a, x in zip(learnt_posterior.a, learnt_posterior.xs):
            helper.plot_pdf_marginals(pdf=x, lims=disp_lims, title= f'Learnt Posterior (SNPE-A), a = {a:.2f}')
    
    plt.show(block=True)



def show_mdn_with_bootstrapping_posterior():
    """
    Shows the posterior learnt by the mdn using bootstrapping.
    """

    # fig_dist = plt.figure()
    all_dist = np.array([])

    for iter in range(n_bootstrap_iter):

        # load approximate posterior
        _, approx_posterior, _, dist = helper.load(netsdir + 'mdn_svi_proposal_prior_{0}.pkl'.format(iter))
        m, S = approx_posterior.calc_mean_and_cov()

        # PLOT 1: Learnt posterior
        title = f'Learnt Posterior (Bootstrapping), Iteration = {iter+1}\n Mean = {m.item():.4f}, Variance = {S[0][0]:.4f}'
        helper.plot_pdf_marginals(pdf=approx_posterior, lims=disp_lims, title=title)

        all_dist = np.append(all_dist, dist)

        # PLOT 2: Distance histograms
        # ax = fig_dist.add_subplot(2, round(n_bootstrap_iter/2), iter+1)
        # ax.hist(dist, bins=int(np.sqrt(dist.size)))
        # ax.set_title(f'Distance Histograms, Iteration = {iter+1}')
        # ax.set_xlim([0.0, 12.0])
        # ax.grid()
        
    

    # PLOT 3: Distance
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_dist, '.')
    ax.set_xlabel('samples')
    ax.set_ylabel('distance')
    ax.grid()
    ax.set_title('Distance Trace Across Bootstrapping Iterations')
    fig.canvas.manager.set_window_title('PLOT 3')

    plt.show(block=True)


if __name__ == '__main__':
    # train_proposal_prior_with_bootstrapping(save=True)
    # train_mdn_with_proposal(save=True)

    # show_mdn_posterior(filename='mdn_svi_proposal_hiddens_20_tanh_sims_1k')
    show_mdn_with_proposal_posterior(filename='mdn_svi_proposal_hiddens_20_tanh_sims_1k')
    # show_mdn_with_bootstrapping_posterior()

