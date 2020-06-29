import theano
import theano.tensor as TT
from rllab.misc import ext
import numpy as np
import copy
import rllab.misc.logger as logger
from rllab.spaces.box import Box
from rllab.envs.env_spec import EnvSpec
from sandbox.finetuning.policies.concurrent_hier_policy2 import HierarchicalPolicy
from sandbox.finetuning.algos.hier_batch_polopt import BatchPolopt, \
    BatchSampler  # note that I use my own BatchPolopt class here
from sandbox.finetuning.algos.hier_batch_sampler import HierBatchSampler
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import lasagne.updates 

class Concurrent_PPO(BatchPolopt):
    """
    Designed to enable concurrent training of a SNN that parameterizes skills
    and also train the manager at the same time

    Note that, if I'm not trying to do the sample approximation of the weird log of sum term,
    I don't need to know which skill was picked, just need to know the action
    """

    # double check this constructor later
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.003,
                 step_size_2=0.0000001,
                 num_latents=6,
                 latents=None,  # some sort of iterable of the actual latent vectors
                 period=10,  # how often I choose a latent
                 truncate_local_is_ratio=None,
                 epsilon=0.1,
                 train_pi_iters=10,
                 use_skill_dependent_baseline=False,
                 mlp_skill_dependent_baseline=False,
                 freeze_manager=False,
                 freeze_skills=False,
                 **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                # optimizer_args = dict()
                optimizer_args = dict(batch_size=None)
            self.optimizer = FirstOrderOptimizer(learning_rate=step_size, max_epochs=train_pi_iters, **optimizer_args)
        self.step_size = step_size
        self.step_size_2 = step_size_2
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.epsilon = epsilon

        super(Concurrent_PPO, self).__init__(**kwargs)  # not sure if this line is correct
        self.num_latents = kwargs['policy'].latent_dim
        self.latents = latents
        self.period = period
        self.freeze_manager = freeze_manager
        self.freeze_skills = freeze_skills
        assert (not freeze_manager) or (not freeze_skills)

        # todo: fix this sampler stuff
        # import pdb; pdb.set_trace()
        self.sampler = HierBatchSampler(self, self.period)
        # self.sampler = BatchSampler(self)
        # i hope this is right
        self.diagonal = DiagonalGaussian(self.policy.low_policy.action_space.flat_dim)
        self.debug_fns = []

        assert isinstance(self.policy, HierarchicalPolicy)
        if self.policy is not None:
            self.period = self.policy.period
        assert self.policy.period == self.period
        # self.old_policy = copy.deepcopy(self.policy)

        # skill dependent baseline
        self.use_skill_dependent_baseline = use_skill_dependent_baseline
        self.mlp_skill_dependent_baseline = mlp_skill_dependent_baseline
        if use_skill_dependent_baseline:
            curr_env = kwargs['env']
            skill_dependent_action_space = curr_env.action_space
            new_obs_space_no_bi = curr_env.observation_space.shape[0] + 1  # 1 for the t_remaining
            skill_dependent_obs_space_dim = (new_obs_space_no_bi * (self.num_latents + 1) + self.num_latents,)
            skill_dependent_obs_space = Box(-1.0, 1.0, shape=skill_dependent_obs_space_dim)
            skill_dependent_env_spec = EnvSpec(skill_dependent_obs_space, skill_dependent_action_space)
            if self.mlp_skill_dependent_baseline:
                self.skill_dependent_baseline = GaussianMLPBaseline(env_spec=skill_dependent_env_spec)
            else:
                self.skill_dependent_baseline = LinearFeatureBaseline(env_spec=skill_dependent_env_spec)

    # initialize the computation graph
    # optimize is run on >= 1 trajectory at a time
    # assumptions: 1 trajectory, which is a multiple of p; that the obs_var_probs is valid
    def init_opt(self):
        assert isinstance(self.policy, HierarchicalPolicy)
        manager_surr_loss = 0
        skill_surr_loss = 0

        disc_rewards_var = ext.new_tensor('disc_rewards', ndim=1, dtype=theano.config.floatX)

        if not self.freeze_manager:
            obs_var_sparse = ext.new_tensor('sparse_obs', ndim=2, dtype=theano.config.floatX)
            latent_var_sparse = ext.new_tensor('sparse_latent', ndim=2, dtype=theano.config.floatX)
            advantage_var_sparse = ext.new_tensor('sparse_advantage', ndim=1,
                                                  dtype=theano.config.floatX)  # advantage every self.period timesteps
            manager_prob_var = ext.new_tensor('manager_prob_var', ndim=2, dtype=theano.config.floatX)
            #############################################################
            ### calculating the manager portion of the surrogate loss ###
            #############################################################

            latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
            actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
            old_actual_latent_probs = TT.sum(manager_prob_var * latent_var_sparse, axis=1)
            lr = TT.exp(TT.log(actual_latent_probs) - TT.log(old_actual_latent_probs))
            manager_surr_loss_vector = TT.minimum(lr * advantage_var_sparse,
                                                  TT.clip(lr, 1 - self.epsilon, 1 + self.epsilon) * advantage_var_sparse)
            manager_surr_loss = -TT.mean(manager_surr_loss_vector)

        if not self.freeze_skills:
            obs_var_raw = ext.new_tensor('obs', ndim=3, dtype=theano.config.floatX)  # todo: check the dtype
            action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1, )
            advantage_var = ext.new_tensor('advantage', ndim=1, dtype=theano.config.floatX)
            latent_var = ext.new_tensor('latents', ndim=2, dtype=theano.config.floatX)
            mean_var = ext.new_tensor('mean', ndim=2, dtype=theano.config.floatX)
            log_std_var = ext.new_tensor('log_std', ndim=2, dtype=theano.config.floatX)


            # undoing the reshape, so that batch sampling is ok
            obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])

            ############################################################
            ### calculating the skills portion of the surrogate loss ###
            ############################################################
            dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
            old_dist_info_var = dict(mean=mean_var, log_std=log_std_var)
            skill_lr = self.diagonal.likelihood_ratio_sym(action_var, old_dist_info_var, dist_info_var)
            skill_surr_loss_vector = TT.minimum(skill_lr * advantage_var,
                                                TT.clip(skill_lr, 1 - self.epsilon, 1 + self.epsilon) * advantage_var)
            skill_surr_loss = -TT.mean(skill_surr_loss_vector)

        surr_loss = manager_surr_loss / self.period + skill_surr_loss  # so that the relative magnitudes are correct

        if self.freeze_skills and not self.freeze_manager:
            input_list = [obs_var_sparse, advantage_var_sparse, latent_var_sparse, manager_prob_var]
        elif self.freeze_manager and not self.freeze_skills:
            input_list = [obs_var_raw, action_var, advantage_var, latent_var, mean_var, log_std_var]
        else:
            assert (not self.freeze_manager) or (not self.freeze_skills)
            input_list = [obs_var_raw, obs_var_sparse, action_var, advantage_var, advantage_var_sparse, latent_var,
                          latent_var_sparse, mean_var, log_std_var, manager_prob_var]

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list
        )

        experimental = True


        self.first_order_grad_lo = self.first_order_grad_lo_init(obs_var_raw, action_var, latent_var, advantage_var)
        self.first_order_grad_hi = self.first_order_grad_hi_init(obs_var_sparse, latent_var_sparse, advantage_var_sparse)

        if not experimental:
            self.second_order_grad_hi_lo = self.second_order_grad_hi_lo_init(obs_var_raw, action_var, latent_var, disc_rewards_var, obs_var_sparse, latent_var_sparse)
            self.second_order_grad_lo_hi = self.second_order_grad_lo_hi_init(obs_var_raw, action_var, latent_var, disc_rewards_var, obs_var_sparse, latent_var_sparse)

        else:
            self.second_order_grad_hi_lo = self.second_order_grad_hi_lo_exp_init(obs_var_raw, action_var, latent_var, advantage_var, obs_var_sparse, latent_var_sparse)
            self.second_order_grad_lo_hi = self.second_order_grad_lo_hi_exp_init(obs_var_raw, action_var, latent_var, advantage_var, obs_var_sparse, latent_var_sparse)


        self.hi_lo_magnitude = sum([x.norm(2) ** 2 for x in self.second_order_grad_hi_lo]) ** 0.5
        self.lo_hi_magnitude = sum([x.norm(2) ** 2 for x in self.second_order_grad_lo_hi]) ** 0.5 

        # self.hi_lo_clipped = lasagne.updates.total_norm_constraint(self.second_order_grad_hi_lo, 0.3)
        # self.lo_hi_clipped = lasagne.updates.total_norm_constraint(self.second_order_grad_lo_hi, 0.1)

        # self.hi_lo_clipped_magnitude = sum([x.norm(2) ** 2 for x in self.hi_lo_clipped]) ** 0.5
        # self.lo_hi_clipped_magnitude = sum([x.norm(2) ** 2 for x in self.lo_hi_clipped]) ** 0.5

        self.updates_1 = lasagne.updates.adam(self.second_order_grad_hi_lo, self.policy.manager.get_params(trainable=True), learning_rate=self.step_size_2)
        self.updates_2 = lasagne.updates.adam(self.second_order_grad_lo_hi, self.policy.low_policy.get_params(trainable=True), learning_rate=self.step_size_2)

        # Updates params, and outputs the magnitude of the gradient. 
        self.train_fn_1 = theano.function([obs_var_raw, action_var, latent_var, obs_var_sparse, latent_var_sparse, advantage_var], updates = self.updates_1, outputs=self.hi_lo_magnitude)
        self.train_fn_2 = theano.function([obs_var_raw, action_var, latent_var, advantage_var, obs_var_sparse, latent_var_sparse, advantage_var_sparse],updates = self.updates_2, outputs=self.lo_hi_magnitude)

        print("INIT HERE", self.step_size_2)

        return dict()

    # Returns d(theta_lo)[R]
    def first_order_grad_lo_init(self, obs_var_raw, action_var, latent_var, advantage_var):
        #input_list = [obs_var_raw, action_var, latent_var, advantage_var]

        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])

        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        log_probs = self.diagonal.log_likelihood_sym(action_var, dist_info_var)
        # todo: verify that dist_info_vars is in order

        low_surrogate_loss = TT.mean(log_probs * advantage_var)

        return theano.grad(low_surrogate_loss, self.policy.low_policy.get_params(trainable=True))

        #self.grad_likelihood_info_low = (low_ll_loss, input_list)

    # Returns d(theta_hi)[R]
    def first_order_grad_hi_init(self, obs_var_sparse, latent_var_sparse, advantage_var_sparse):
        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        manager_surr_loss = TT.mean(TT.log(actual_latent_probs) * advantage_var_sparse)

        return theano.grad(manager_surr_loss, self.policy.manager.get_params())

    # d(theta_hi) d(theta_lo)[R]
    # The final gradient has the shape of theta_hi 
    def second_order_grad_hi_lo_init(self, obs_var_raw, action_var, latent_var, disc_rewards_var, obs_var_sparse, latent_var_sparse):
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])
        disc_rewards_var_batched = TT.reshape(disc_rewards_var, [disc_rewards_var.shape[0] // 5000, 5000])

        ## Computing the cumulative likelihood for low policy
        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        log_probs_low = self.diagonal.log_likelihood_sym(action_var, dist_info_var)
        log_probs_low_batched = TT.reshape(log_probs_low, [log_probs_low.shape[0] // 5000, 5000])
        cum_log_probs_low_batched = TT.cumsum(log_probs_low_batched, axis=1)

        ## Computing the cumulative likelihood for high policy

        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        actual_latent_probs_batched = TT.reshape(actual_latent_probs, [actual_latent_probs.shape[0] // 5000, 5000])
        cum_latent_probs_batched = TT.cumsum(actual_latent_probs_batched, axis=1)

        surrogate_loss = TT.mean(disc_rewards_var_batched * cum_log_probs_low_batched * cum_latent_probs_batched)

        fo_grad_lo = self.first_order_grad_lo

        grad_lo = theano.grad(surrogate_loss, self.policy.low_policy.get_params())
        # Final result is grad_lo(R)^T * grad_hi (grad_low (R))
        # Compute this result in one step with Lop

        return theano.Lop(grad_lo, self.policy.manager.get_params(), fo_grad_lo)

    # d(theta_lo) d(theta_hi)[R]
    # The final gradient has the shape of theta_lo
    def second_order_grad_lo_hi_init(self, obs_var_raw, action_var, latent_var, disc_rewards_var, obs_var_sparse, latent_var_sparse):
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])
        disc_rewards_var_batched = TT.reshape(disc_rewards_var, [disc_rewards_var.shape[0] // 5000, 5000])

        ## Computing the cumulative likelihood for low policy
        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        log_probs_low = self.diagonal.log_likelihood_sym(action_var, dist_info_var)
        log_probs_low_batched = TT.reshape(log_probs_low, [log_probs_low.shape[0] // 5000, 5000])
        cum_log_probs_low_batched = TT.cumsum(log_probs_low_batched, axis=1)

        ## Computing the cumulative likelihood for high policy

        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)
        actual_latent_probs_batched = TT.reshape(actual_latent_probs, [actual_latent_probs.shape[0] // 5000, 5000])
        cum_latent_probs_batched = TT.cumsum(actual_latent_probs_batched, axis=1)

        surrogate_loss = TT.mean(disc_rewards_var_batched * cum_log_probs_low_batched * cum_latent_probs_batched)

        fo_grad_hi = self.first_order_grad_hi

        grad_hi = theano.grad(surrogate_loss, self.policy.manager.get_params())
        # Final result is grad_lo(R)^T * grad_hi (grad_low (R))
        # Compute this result in one step with Lop

        return theano.Lop(grad_hi, self.policy.low_policy.get_params(), fo_grad_hi)


    def second_order_grad_hi_lo_exp_init(self, obs_var_raw, action_var, latent_var, adv_var, obs_var_sparse, latent_var_sparse):
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])

        ## Computing the cumulative likelihood for low policy
        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        log_probs_low = self.diagonal.log_likelihood_sym(action_var, dist_info_var)

        ## Computing the cumulative likelihood for high policy

        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)

        surrogate_loss = TT.mean(adv_var * log_probs_low * actual_latent_probs)

        fo_grad_lo = self.first_order_grad_lo

        grad_lo = theano.grad(surrogate_loss, self.policy.low_policy.get_params())
        # Final result is grad_lo(R)^T * grad_hi (grad_low (R))
        # Compute this result in one step with Lop

        return theano.Lop(grad_lo, self.policy.manager.get_params(), fo_grad_lo)

    def second_order_grad_lo_hi_exp_init(self, obs_var_raw, action_var, latent_var, adv_var, obs_var_sparse, latent_var_sparse):
        obs_var = TT.reshape(obs_var_raw, [obs_var_raw.shape[0] * obs_var_raw.shape[1], obs_var_raw.shape[2]])

        ## Computing the cumulative likelihood for low policy
        dist_info_var = self.policy.low_policy.dist_info_sym(obs_var, state_info_var=latent_var)
        log_probs_low = self.diagonal.log_likelihood_sym(action_var, dist_info_var)

        ## Computing the cumulative likelihood for high policy

        latent_probs = self.policy.manager.dist_info_sym(obs_var_sparse)['prob']
        actual_latent_probs = TT.sum(latent_probs * latent_var_sparse, axis=1)

        surrogate_loss = TT.mean(adv_var * log_probs_low * actual_latent_probs)
        fo_grad_hi = self.first_order_grad_hi

        grad_hi = theano.grad(surrogate_loss, self.policy.manager.get_params())

        return theano.Lop(grad_hi, self.policy.low_policy.get_params(), fo_grad_hi)

    # do the optimization
    def optimize_policy(self, itr, samples_data):
        #KEYS:  dict_keys(['observations', 'actions', 'rewards', 'returns', 'advantages', 'env_infos', 'agent_infos', 'paths', 'skill_advantages'])
        print(len(samples_data['observations']), self.period)
        assert len(samples_data['observations']) % self.period == 0
        # note that I have to do extra preprocessing to the advantages, and also create obs_var_sparse

        if self.use_skill_dependent_baseline:
            input_values = tuple(ext.extract(
                samples_data, "observations", "actions", "advantages", "agent_infos", "rewards", "skill_advantages",))
        else:
            input_values = tuple(ext.extract(
                samples_data, "observations", "actions", "advantages", "agent_infos", "rewards"))

        obs_raw = input_values[0].reshape(input_values[0].shape[0] // self.period, self.period,
                                          input_values[0].shape[1])

        obs_sparse = input_values[0].take([i for i in range(0, input_values[0].shape[0], self.period)], axis=0)
        advantage_sparse = input_values[2].reshape([input_values[2].shape[0] // self.period, self.period])[:, 0]
        latents = input_values[3]['latents']
        latents_sparse = latents.take([i for i in range(0, latents.shape[0], self.period)], axis=0)
        mean = input_values[3]['mean']
        log_std = input_values[3]['log_std']
        prob = np.array(
            list(input_values[3]['prob'].take([i for i in range(0, latents.shape[0], self.period)], axis=0)),
            dtype=np.float32)
        if self.use_skill_dependent_baseline:
            advantage_var = input_values[5]
        else:
            advantage_var = input_values[2]
        # import ipdb; ipdb.set_trace()
        if self.freeze_skills and not self.freeze_manager:
            all_input_values = (obs_sparse, advantage_sparse, latents_sparse, prob)
        elif self.freeze_manager and not self.freeze_skills:
            all_input_values = (obs_raw, input_values[1], advantage_var, latents, mean, log_std)
        else:
            assert (not self.freeze_manager) or (not self.freeze_skills)
            all_input_values = (obs_raw, obs_sparse, input_values[1], advantage_var, advantage_sparse, latents,
                            latents_sparse, mean, log_std, prob)


        # disc_rewards = input_values[4]
        # disc = 1. 
        # for t in range(len(disc_rewards)):
        #     disc_rewards[t] *= disc
        #     disc *= 0.999

        #     if t % 5000 == 0:
        #         disc = 1.


        # todo: assign current parameters to old policy; does this work?
        # old_param_values = self.policy.get_param_values(trainable=True)
        # self.old_policy.set_param_values(old_param_values, trainable=True)
        # old_param_values = self.policy.get_param_values()
        # self.old_policy.set_param_values(old_param_values)

        loss_before = self.optimizer.loss(all_input_values)
        self.optimizer.optimize(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)

        ### Second order gradient corrections

        print("Start")

        if self.step_size_2 != 0:
            mag_1 = self.train_fn_1(obs_raw, input_values[1], latents, obs_sparse, latents_sparse, advantage_var)
            mag_2 = self.train_fn_2(obs_raw, input_values[1], latents, advantage_var, obs_sparse, latents_sparse, advantage_sparse)

            logger.record_tabular('Grad_hi_lo_Mag', mag_1)
            logger.record_tabular('Grad_lo_hi_Mag', mag_2)

        print("End")

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env
        )

    def log_diagnostics(self, paths):
        # paths obtained by self.sampler.obtain_samples
        BatchPolopt.log_diagnostics(self, paths)
        # self.sampler.log_diagnostics(paths)   # wasn't doing anything anyways

        # want to log the standard deviations
        # want to log the max and min of the actions
