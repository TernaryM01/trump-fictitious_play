import collections
import random
from typing import Sequence, Dict, Tuple, List, Any, Generator, Callable, Optional

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
import pickle
import gc

import flax.linen as nn
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python import policy
import pyspiel

from tqdm.notebook import tqdm

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from multiprocessing import Pool
from cfvfp_worker_proper import run_worker_traversals, get_dir

class DeepCFVFPSolver(policy.Policy):
    def __init__(self,
                 game_name: str,
                 
                 # Sequences of models and transformers, one per phase
                 q_value_network_models: Sequence[nn.Module],
                 avg_policy_network_models: Sequence[nn.Module],
                 info_state_tensor_transformers: Sequence[Callable[[jax.Array], jax.Array]],
                 action_transformers: Sequence[Callable[[jax.Array], jax.Array]],
                 phase_classifier_fn: Callable[[jax.Array], int], # Takes full infostate, returns phase

                 batch_size_q_value: Sequence[int],
                 batch_size_avg_policy: Sequence[int],
                 q_value_memory_capacity: Sequence[int],
                 avg_policy_memory_capacity: Sequence[int],
                 q_value_network_train_steps: Sequence[int],
                 avg_policy_network_train_steps: Sequence[int],

                 dummy_infostate: np.ndarray = None,

                 data_augmentors: Optional[List[Optional[Callable[[jax.Array], jax.Array]]]] = None,

                 revelation_transformer: Optional[Callable[[List[jax.Array], int, float], jax.Array]] = None,
                 revelation_intensity: Sequence[float] = [1.0, 0.0],
                 revelation_decay_mode: str = 'linear',

                 uniform: bool = False,

                 # Training parameters
                 num_iterations: int = 100,
                 num_traversals_per_player: int = 100,
                 num_iterations_q_per_pi: int = 1,
                #  branching_rate: float = 0.0,
                #  branching_depth_limit: int = 0,

                 learning_rate: float = 1e-3,
                 
                #  reinit_q_nets: bool = False,
                 average_weighting_mode: str = 'vanilla',

                 save_dir_buffers: str = "cfvfp_buffers",
                 save_dir_nets: str = "cfvfp_nets",
                 num_workers = None,
                 seed: int = 42):
        
        self._num_workers = num_workers if num_workers else mp.cpu_count() // 2  # number of physical cores
        self._uniform = uniform

        self._game = pyspiel.load_game(game_name)
        super(DeepCFVFPSolver, self).__init__(self._game, list(range(self._game.num_players())))
        if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError('Simultaneous games are not supported.')
        self._num_phases = len(action_transformers)
        if not (len(q_value_network_models) == len(avg_policy_network_models) == \
                len(info_state_tensor_transformers) == self._num_phases and \
                len(batch_size_q_value) == self._num_phases and \
                len(batch_size_avg_policy) == self._num_phases and \
                len(q_value_memory_capacity) == self._num_phases and \
                len(avg_policy_memory_capacity) == self._num_phases and \
                len(q_value_network_train_steps) == self._num_phases and \
                len(avg_policy_network_train_steps) == self._num_phases):
            raise ValueError("Mismatch of lengths of model and transformer sequences.")

        self._batch_size_q_value = [int(i) for i in batch_size_q_value]
        self._batch_size_avg_policy = [int(i) for i in batch_size_avg_policy]
        self._q_value_network_train_steps_iter = [int(i) for i in q_value_network_train_steps]
        self._avg_policy_network_train_steps_iter = [int(i) for i in avg_policy_network_train_steps]
        self._q_value_memory_capacity = [int(i) for i in q_value_memory_capacity]
        self._avg_policy_memory_capacity = [int(i) for i in avg_policy_memory_capacity]
        
        self._num_players = self._game.num_players()
        self._global_num_actions = self._game.num_distinct_actions()

        self._phase_classifier_fn = phase_classifier_fn
        self._action_transformers = action_transformers
        # Parse transformers to separate Q and policy transformers
        self._info_state_tensor_transformers_q = []
        self._info_state_tensor_transformers_pi = []
        for transformer in info_state_tensor_transformers:
            if isinstance(transformer, list) and len(transformer) == 2:
                self._info_state_tensor_transformers_q.append(transformer[0])
                self._info_state_tensor_transformers_pi.append(transformer[1])
            else:
                self._info_state_tensor_transformers_q.append(transformer)
                self._info_state_tensor_transformers_pi.append(transformer)

        self._data_augmentors = data_augmentors if data_augmentors else [None] * self._num_phases

        self._revelation_transformer = revelation_transformer
        self._revelation_intensity = revelation_intensity
        self._revelation_decay_mode = revelation_decay_mode

        self._num_traversals_per_player = int(num_traversals_per_player)
        # self.branching_rate = branching_rate
        # self.branching_depth_limit = branching_depth_limit

        # self._reinit_q_nets = reinit_q_nets
        self._average_weighting_mode = average_weighting_mode
        
        self._learning_rate = learning_rate
        self._master_rngkey = jax.random.PRNGKey(seed)
        self._rngkey, subkey_init_master = jax.random.split(self._master_rngkey)

        self.models_q: List[nn.Module] = list(q_value_network_models)
        self.models_pi: List[nn.Module] = list(avg_policy_network_models)

        # Parameters, optimizers, etc., are now lists indexed by phase_idx, then by player_idx
        self.params_q: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.params_pi: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.opt_states_q: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.opt_states_pi: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.optimizers_q: List[List[optax.GradientTransformation]]= [[] for _ in range(self._num_phases)]
        self.optimizers_pi: List[List[optax.GradientTransformation]] = [[] for _ in range(self._num_phases)]
        self.value_and_grad_fns_q: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.value_and_grad_fns_pi: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.jitted_updates_q: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.jitted_updates_pi: List[List[Any]] = [[] for _ in range(self._num_phases)]
        # self.jitted_inference_q: List[List[Any]] = [[] for _ in range(self._num_phases)]
        self.jitted_inference_pi: List[List[Any]] = [[] for _ in range(self._num_phases)]

        self._num_iterations_q_per_pi = int(num_iterations_q_per_pi)
        self._num_iterations = int(num_iterations)
        self._iteration = 1

        init_key_q_master, init_key_pi_master = jax.random.split(subkey_init_master)
        for phase in range(self._num_phases):
            phase_q_master_key = jax.random.fold_in(init_key_q_master, phase)
            phase_pi_master_key = jax.random.fold_in(init_key_pi_master, phase)
            q_phase_keys = jax.random.split(phase_q_master_key, self._num_players)
            pi_phase_keys = jax.random.split(phase_pi_master_key, self._num_players)

            # Dummy inputs for this phase's network
            # The transformer function for this phase should produce the correct shape
            dummy_full_info_state = np.ones([1, self._game.information_state_tensor_shape()[0]], dtype=np.float32) \
                if dummy_infostate is None else dummy_infostate
            dummy_info_state_transformed_q = self._info_state_tensor_transformers_q[phase](dummy_full_info_state)
            dummy_info_state_transformed_pi = self._info_state_tensor_transformers_pi[phase](dummy_full_info_state)
            
            # Policy networks output logits for global action space, mask applied later
            dummy_mask_global = jnp.ones([1, self._global_num_actions], dtype=bool)

            for i in range(self._num_players if not self._uniform else 1):
                q_params = self.models_q[phase].init(q_phase_keys[i], dummy_info_state_transformed_q)['params']
                self.params_q[phase].append(q_params)
                self.optimizers_q[phase].append(optax.adam(learning_rate))
                self.opt_states_q[phase].append(self.optimizers_q[phase][i].init(q_params))
                self.value_and_grad_fns_q[phase].append(jax.value_and_grad(self._loss_q_value_batch_factory(self.models_q[phase], self._action_transformers[phase])))
                self.jitted_updates_q[phase].append(self._get_jitted_q_value_update(self.optimizers_q[phase][i], self.value_and_grad_fns_q[phase][i]))
                # self.jitted_inference_q[phase].append(self._get_jitted_best_response(self.models_q[phase], self.action_transformers[phase]))
                
                pi_params = self.models_pi[phase].init(pi_phase_keys[i], dummy_info_state_transformed_pi, dummy_mask_global)['params']
                self.params_pi[phase].append(pi_params)
                self.optimizers_pi[phase].append(optax.adam(learning_rate))
                self.opt_states_pi[phase].append(self.optimizers_pi[phase][i].init(pi_params))
                self.value_and_grad_fns_pi[phase].append(jax.value_and_grad(self._loss_avg_policy_batch_factory(self.models_pi[phase], self._action_transformers[phase])))
                self.jitted_updates_pi[phase].append(self._get_jitted_avg_policy_update(self.optimizers_pi[phase][i], self.value_and_grad_fns_pi[phase][i]))
                self.jitted_inference_pi[phase].append(self._get_jitted_avg_policy(self.models_pi[phase], self._action_transformers[phase]))

        # Buffer Management

        self._q_value_target_memories_p: List[List[List[Dict[str, np.ndarray]]]] = \
            [[[] for _ in range(self._num_players)] for _ in range(self._num_phases)]
        self._best_response_memories_p: List[List[List[Dict[str, np.ndarray]]]] = \
            [[[] for _ in range(self._num_players)] for _ in range(self._num_phases)]
        
        self._save_dir_buffers = save_dir_buffers
        os.makedirs(self._save_dir_buffers, exist_ok=True)
        self._save_dir_nets = save_dir_nets
        os.makedirs(self._save_dir_nets, exist_ok=True)

    def _get_jitted_q_value_update(self, optimizer_instance: optax.GradientTransformation, 
                                   value_and_grad_fn_instance: Callable) -> Callable:
        """Returns a JIT-compiled function for a single Q-value network training step."""
        @jax.jit
        def update(params_q_value: Any, opt_state: Any,
                   info_states: jax.Array, actions_taken: jax.Array,
                   target_q_values: jax.Array):
            # value_and_grad_fn_instance is the result of jax.value_and_grad(self._loss_q_value_batch_factory(...))
            loss_val, grads = value_and_grad_fn_instance(
                params_q_value, info_states, actions_taken, target_q_values
            )
            updates, new_opt_state = optimizer_instance.update(grads, opt_state, params_q_value)
            new_params = optax.apply_updates(params_q_value, updates)
            return new_params, new_opt_state, loss_val
        return update

    def _get_jitted_avg_policy_update(self, optimizer_instance: optax.GradientTransformation, 
                                      value_and_grad_fn_instance: Callable) -> Callable:
        """Returns a JIT-compiled function for a single average policy network training step."""
        @jax.jit
        def update(params_avg_policy: Any, opt_state: Any, 
                   info_states: jax.Array, best_response_probs: jax.Array,
                   masks: jax.Array, sample_iterations: jax.Array = None):
            # value_and_grad_fn_instance is the result of jax.value_and_grad(self._loss_avg_policy_batch_factory(...))
            loss_val, grads = value_and_grad_fn_instance(
                params_avg_policy, info_states, best_response_probs,
                masks, sample_iterations
            )
            updates, new_opt_state = optimizer_instance.update(grads, opt_state, params_avg_policy)
            new_params = optax.apply_updates(params_avg_policy, updates)
            return new_params, new_opt_state, loss_val
        return update

    # def _get_jitted_best_response(self, q_model_instance, action_transformer_fn):
    #     @jax.jit
    #     def get_q_br(params_q_value: Any, info_state_transformed: jax.Array, legal_actions_mask_global: jax.Array):
    #         q_values_net_output = q_model_instance.apply({'params': params_q_value}, info_state_transformed)[0]
    #         q_values_global = action_transformer_fn(q_values_net_output)
            
    #         masked_q_values = jnp.where(legal_actions_mask_global, q_values_global, jnp.finfo(q_values_global.dtype).min)
    #         return jnp.argmax(masked_q_values) # Global action ID
    #     return get_q_br

    def _get_jitted_avg_policy(self, pi_model_instance, action_transformer_fn):
        @jax.jit
        def get_policy(params_avg_policy: Any, info_state_transformed: jax.Array, legal_actions_mask_global: jax.Array):
            masked_logits_net_output = pi_model_instance.apply(
                {'params': params_avg_policy}, info_state_transformed, legal_actions_mask_global # Pass global mask
            )
            # Assume model's output is already masked logits for global action space, or action_transformer handles it
            masked_logits_global = action_transformer_fn(masked_logits_net_output)
            avg_policy_probs_global = jax.nn.softmax(masked_logits_global, axis=-1)
            return jnp.squeeze(avg_policy_probs_global, axis=0)
        return get_policy

    def action_probabilities(self, state: pyspiel.State, player: int = None) -> Dict[int, float]:
        player = player if player is not None else state.current_player()
        if player < 0 or player >= self._num_players : 
            legal_actions = state.legal_actions(player) if player >=0 else state.legal_actions() if state.is_chance_node() else []
            if not legal_actions: return {}
            prob = 1.0 / len(legal_actions)
            return {action: prob for action in legal_actions}

        full_info_state_np = np.array(state.information_state_tensor(player), dtype=np.float32)
        
        phase = self._phase_classifier_fn(full_info_state_np)
        info_state_transformed_np = self._info_state_tensor_transformers_pi[phase](full_info_state_np)
        legal_actions_mask_global_np = np.array(state.legal_actions_mask(player), dtype=bool)

        avg_policy_probs_global = self.jitted_inference_pi[phase][player](
            self.params_pi[phase][player], 
            jnp.expand_dims(info_state_transformed_np, axis=0), 
            jnp.expand_dims(legal_actions_mask_global_np, axis=0)
        )
        avg_policy_probs_np_global = np.array(avg_policy_probs_global)
        
        legal_actions = state.legal_actions(player) 
        if not legal_actions: return {}
        
        return {action: float(avg_policy_probs_np_global[action]) for action in legal_actions if action < len(avg_policy_probs_np_global)}

    def solve(self) -> Tuple[List[List[Any]], Dict[str, Dict[int, List[float]]], Dict[str, Dict[int, List[float]]]]:
        # Determine the last complete iteration where all network files exist
        all_iterations = []
        for player in range(self._num_players if not self._uniform else 1):
            for phase in range(self._num_phases):
                q_iters = self._get_saved_iterations(player, phase, 'q')
                pi_iters = self._get_saved_iterations(player, phase, 'pi')
                common_iters = q_iters.intersection(pi_iters)
                all_iterations.append(common_iters)

        if all_iterations:
            common_iterations = set.intersection(*all_iterations)
            last_complete_iter = max(common_iterations) if common_iterations else 0
        else:
            last_complete_iter = 0

        starting_iter = last_complete_iter + 1 if last_complete_iter > 0 else 1

        # Load network states from the last complete iteration if it exists
        if last_complete_iter > 0:
            print(f"Resuming from iteration {last_complete_iter}, starting at iteration {starting_iter}")
            for player in range(self._num_players if not self._uniform else 1):
                for phase in range(self._num_phases):
                    # Load Q-value network state
                    q_params_path = os.path.join(get_dir(self._save_dir_nets, player, phase, 'q'), f"q_params_iter{last_complete_iter}.msgpack")
                    with open(q_params_path, 'rb') as f:
                        target = {'params': self.params_q[phase][player], 'opt_state': self.opt_states_q[phase][player]}
                        state_dict = flax.serialization.from_bytes(target, f.read())
                    self.params_q[phase][player] = state_dict['params']
                    self.opt_states_q[phase][player] = state_dict['opt_state']

                    # Load Average Policy network state
                    pi_params_path = os.path.join(get_dir(self._save_dir_nets, player, phase, 'pi'), f"pi_params_iter{last_complete_iter}.msgpack")
                    with open(pi_params_path, 'rb') as f:
                        target = {'params': self.params_pi[phase][player], 'opt_state': self.opt_states_pi[phase][player]}
                        state_dict = flax.serialization.from_bytes(target, f.read())
                    self.params_pi[phase][player] = state_dict['params']
                    self.opt_states_pi[phase][player] = state_dict['opt_state']
        else:
            print("Starting from scratch")

        q_value_losses_by_phase = collections.defaultdict(lambda: collections.defaultdict(list))
        avg_policy_losses_by_phase = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for i in tqdm(range(starting_iter, self._num_iterations + 1), desc="Overall Iteration"):
            self._iteration = i  # Only used for weighting in non-vanilla mode
            train_record_pi = i % self._num_iterations_q_per_pi == 0

            revelation_start, revelation_finish = self._revelation_intensity
            if self._revelation_decay_mode == 'exponential':
                decay_rate = (revelation_finish / revelation_start) ** (1 / self._num_iterations)
                revelation_rate = revelation_start * (decay_rate ** i)
            else:  # 'linear'
                revelation_rate = revelation_start + (revelation_finish - revelation_start) * (i / self._num_iterations)

            players_iter_tqdm = tqdm(range(self._num_players), desc="Player", leave=False) if not self._uniform else range(1)
            for player in players_iter_tqdm:
                # if self._reinit_q_nets:
                #     for phase in range(self._num_phases):
                #         self._reinitialize_q_value_network(player, phase)
                
                # Parallel traversals
                traversals_per_worker = self._num_traversals_per_player // self._num_workers
                remaining_traversals = self._num_traversals_per_player % self._num_workers
                
                # Prepare worker arguments
                worker_args = []
                for worker_id in range(self._num_workers):
                    worker_traversals = traversals_per_worker + (1 if worker_id < remaining_traversals else 0)
                    if worker_traversals > 0:  # Only create workers that have work to do
                        
                        # Collect parameters for the traversing player's Q-networks
                        q_params_all_phases = [self.params_q[phase][player] for phase in range(self._num_phases)]
                        
                        # Collect parameters for all other players' average policy networks
                        players_iter = range(self._num_players if not self._uniform else 1)
                        pi_params_all_phases_all_players = [
                            [self.params_pi[phase][p] for p in players_iter]
                            for phase in range(self._num_phases)
                        ]
                        worker_rng_keys = jax.random.split(self._next_rng_key(), self._num_workers)
                        worker_arg = {
                            'worker_id': worker_id,
                            'num_traversals': worker_traversals,
                            # 'branching_rate': self.branching_rate,
                            # 'branching_depth_limit': self.branching_depth_limit,
                            'player': player,
                            'uniform': self._uniform,
                            'game_name': self._game.get_type().short_name,
                            'q_params_all_phases': q_params_all_phases,
                            'pi_params_all_phases_all_players': pi_params_all_phases_all_players,
                            'q_models': [self.models_q[phase] for phase in range(self._num_phases)],
                            'pi_models': [self.models_pi[phase] for phase in range(self._num_phases)],
                            'record_pi': train_record_pi,
                            'info_state_tensor_transformers_q': self._info_state_tensor_transformers_q,
                            'info_state_tensor_transformers_pi': self._info_state_tensor_transformers_pi,
                            'action_transformers': self._action_transformers,
                            'phase_classifier_fn': self._phase_classifier_fn,
                            'revelation_transformer': self._revelation_transformer,
                            'revelation_intensity': revelation_rate,
                            'num_players': self._num_players,
                            'global_num_actions': self._global_num_actions,
                            'iteration': self._iteration,
                            'average_weighting_mode': self._average_weighting_mode,
                            'save_dir_buffers': self._save_dir_buffers,
                            'rng_key': worker_rng_keys[worker_id]
                        }
                        worker_args.append(worker_arg)

                # Record existing files before launching workers
                existing_files = {}
                for phase in range(self._num_phases):
                    existing_files[phase] = {}
                    for buffer_type in ["q", "pi"]:
                        buffer_dir = get_dir(self._save_dir_buffers, player, phase, buffer_type)
                        if os.path.exists(buffer_dir):
                            files = [f for f in os.listdir(buffer_dir) if f.endswith('.pkl')]
                            # Delete backup files if their main counterpart exists
                            filtered_files = []
                            for f in files:
                                if f.endswith('_backup.pkl'):
                                    main_file = f.replace('_backup.pkl', '.pkl')
                                    if main_file in files:
                                        # Delete the backup file
                                        try:
                                            os.remove(os.path.join(buffer_dir, f))
                                        except OSError:
                                            print(f"Warning: Failed to remove backup file {f}")
                                    else:
                                        filtered_files.append(f)
                                else:
                                    filtered_files.append(f)
                            existing_files[phase][buffer_type] = filtered_files
                        else:
                            existing_files[phase][buffer_type] = []

                # Launch workers asynchronously
                pool = None
                if worker_args:
                    pool = Pool(processes=len(worker_args))
                    async_result = pool.map_async(run_worker_traversals, worker_args)
                
                # Neural network training
                for phase in tqdm(range(self._num_phases), desc="Phase", leave=False):
                    self._existing_files_filter = existing_files[phase]

                    cur_q_loss = self._learn_q_value_network(player, phase) 
                    if cur_q_loss is not None:
                        q_value_losses_by_phase[phase][player].append(float(cur_q_loss))
                    if train_record_pi:
                        cur_avg_policy_loss = self._learn_average_policy_network(player, phase)
                        if cur_avg_policy_loss is not None:
                            avg_policy_losses_by_phase[phase][player].append(float(cur_avg_policy_loss))

                    # Save networks after each iteration
                    # Save Q-value networks
                    q_net_dir = get_dir(self._save_dir_nets, player, phase, "q")
                    os.makedirs(q_net_dir, exist_ok=True)
                    q_params_path = os.path.join(q_net_dir, f"q_params_iter{i}.msgpack")
                    q_state_dict = {
                        'params': self.params_q[phase][player],
                        'opt_state': self.opt_states_q[phase][player]
                    }
                    with open(q_params_path, 'wb') as f:
                        f.write(flax.serialization.to_bytes(q_state_dict))
                    # Save average policy networks
                    pi_net_dir = get_dir(self._save_dir_nets, player, phase, "pi")
                    os.makedirs(pi_net_dir, exist_ok=True)
                    pi_params_path = os.path.join(pi_net_dir, f"pi_params_iter{i}.msgpack")
                    pi_state_dict = {
                        'params': self.params_pi[phase][player],
                        'opt_state': self.opt_states_pi[phase][player]
                    }
                    with open(pi_params_path, 'wb') as f:
                        f.write(flax.serialization.to_bytes(pi_state_dict))

                # Wait for workers to finish before next iteration
                if pool is not None:
                    async_result.wait()
                    pool.close()
                    pool.join()
        
        final_avg_policy_params_by_phase = [
            [self.params_pi[phase][player] for player in range(self._num_players if not self._uniform else 1)]
            for phase in range(self._num_phases)
        ]
        return final_avg_policy_params_by_phase, dict(q_value_losses_by_phase), dict(avg_policy_losses_by_phase)

    # def _reinitialize_q_value_network(self, player: int, phase: int):
    #     rng_key = self._next_rng_key()
    #     dummy_full_info_state = jnp.ones([1, self._game.information_state_tensor_shape()[0]], dtype=jnp.float32)
    #     dummy_info_state_for_phase = self._info_state_tensor_transformers_q[phase](dummy_full_info_state)
    #     self.params_q[phase][player] = self.models_q[phase].init(rng_key, dummy_info_state_for_phase)['params']
    #     self.opt_states_q[phase][player] = self.optimizers_q[phase][player].init(self.params_q[phase][player])

    def _loss_q_value_batch_factory(self, q_model_instance_for_phase, action_transformer_fn_for_phase):
        def _loss_q_value_batch(params_q_value: Any, info_states_sliced: jax.Array, 
                                actions_taken_global: jax.Array, target_q_values: jax.Array):
            # q_model_instance_for_phase takes sliced info_states
            # and outputs Q-values over GLOBAL action space
            predicted_q_values_net_output = q_model_instance_for_phase.apply(
                {'params': params_q_value}, info_states_sliced
            )
            # actions_taken_global are indices for the global action space
            predicted_q_values_all_global_actions = action_transformer_fn_for_phase(predicted_q_values_net_output)
            predicted_q_values_for_action_taken = jnp.squeeze(
                jnp.take_along_axis(predicted_q_values_all_global_actions, actions_taken_global, axis=-1),
                axis=-1
            )
            loss = optax.squared_error(predicted_q_values_for_action_taken, jnp.squeeze(target_q_values, axis=-1))
            return jnp.mean(loss)
        return _loss_q_value_batch

    def _loss_avg_policy_batch_factory(self, pi_model_instance_for_phase, action_transformer_fn_for_phase):
        def _loss_avg_policy_batch(params_avg_policy: Any, info_states_sliced: jax.Array, 
                                   best_response_probs_global: jax.Array, masks_global: jax.Array, 
                                   sample_iterations: jax.Array = None):
            # pi_model_instance_for_phase takes sliced info_states and GLOBAL mask
            # and outputs logits over GLOBAL action space
            predicted_logits_net_output = pi_model_instance_for_phase.apply(
                {'params': params_avg_policy}, info_states_sliced, masks_global
            )
            # best_response_probs_global are one-hot vectors over GLOBAL action space
            predicted_logits_global = action_transformer_fn_for_phase(predicted_logits_net_output)
            loss_values = optax.softmax_cross_entropy(logits=predicted_logits_global, labels=best_response_probs_global) 
            
            if self._average_weighting_mode == 'vanilla' or sample_iterations is None:
                return jnp.mean(loss_values)
            else:
                if self._average_weighting_mode == 'linear':
                    iteration_weights = sample_iterations[:, 0].astype(jnp.float32)
                elif self._average_weighting_mode == 'square':
                    iteration_weights = sample_iterations[:, 0].astype(jnp.float32)**2
                elif self._average_weighting_mode == 'log':
                    safe_iterations = jnp.maximum(1.0, sample_iterations[:, 0].astype(jnp.float32))
                    iteration_weights = jnp.log(safe_iterations) / jnp.log(jnp.array(10.0, dtype=jnp.float32)) 
                else: 
                    iteration_weights = jnp.ones_like(loss_values)
                weighted_loss_values = loss_values * iteration_weights
                sum_weights = jnp.sum(iteration_weights)
                return jnp.sum(weighted_loss_values) / sum_weights if sum_weights > 1e-6 else jnp.mean(loss_values)
        return _loss_avg_policy_batch

    def _learn_q_value_network(self, player: int, phase: int) -> float:
        all_experiences_for_training = self._load_combine_prune_save_experiences(player, phase, "q")
        if not all_experiences_for_training or len(all_experiences_for_training['info_state']) < self._batch_size_q_value[phase]:
            # print("something's off")
            return None
        
        avg_loss = 0.0
        num_batches_processed = 0

        # Initialize batch_iterator outside the loop for the first pass
        batch_iterator = self._get_q_value_dataloader_from_arrays(all_experiences_for_training, phase)
        num_steps = self._q_value_network_train_steps_iter[phase]
        for _ in tqdm(range(num_steps), desc="QNet Train", leave=False):
            try:
                info_states, actions_taken_global, target_q_values = next(batch_iterator)
            except StopIteration:
                # Re-initialize iterator for a new epoch by re-shuffling within the dataloader
                batch_iterator = self._get_q_value_dataloader_from_arrays(all_experiences_for_training, phase)
                try: info_states, actions_taken_global, target_q_values = next(batch_iterator)
                except StopIteration: break 
            
            (self.params_q[phase][player],
            self.opt_states_q[phase][player],
            loss_val) = self.jitted_updates_q[phase][player](
                self.params_q[phase][player], self.opt_states_q[phase][player],
                info_states, actions_taken_global, target_q_values
            )
            avg_loss += loss_val
            num_batches_processed += 1

        del all_experiences_for_training 
        gc.collect()
        return avg_loss / num_batches_processed if num_batches_processed > 0 else None

    def _learn_average_policy_network(self, player: int, phase: int) -> float:
        all_experiences_for_training = self._load_combine_prune_save_experiences(player, phase, "pi")
        if not all_experiences_for_training or len(all_experiences_for_training['info_state']) < self._batch_size_avg_policy[phase]:
            return None
        
        avg_loss = 0.0
        num_batches_processed = 0

        # Initialize batch_iterator outside the loop for the first pass
        batch_iterator = self._get_average_policy_dataloader_from_arrays(all_experiences_for_training, phase)
        num_steps = self._avg_policy_network_train_steps_iter[phase]
        for _ in tqdm(range(num_steps), desc="AvgPolNet Train", leave=False):
            try: batch_data = next(batch_iterator)
            except StopIteration:
                # Re-initialize iterator for a new epoch by re-shuffling within the dataloader
                batch_iterator = self._get_average_policy_dataloader_from_arrays(all_experiences_for_training, phase)
                try: batch_data = next(batch_iterator)
                except StopIteration: break
                
            update_kwargs = {'params_avg_policy': self.params_pi[phase][player], 
                            'opt_state': self.opt_states_pi[phase][player]}
            if self._average_weighting_mode == 'vanilla':
                info_states_sliced, best_response_probs_global, masks_global = batch_data
                update_kwargs.update({'info_states': info_states_sliced, 
                                    'best_response_probs': best_response_probs_global, 
                                    'masks': masks_global})
            else:
                info_states_sliced, best_response_probs_global, masks_global, sample_iterations = batch_data
                update_kwargs.update({'info_states': info_states_sliced, 
                                    'best_response_probs': best_response_probs_global, 
                                    'masks': masks_global,
                                    'sample_iterations': sample_iterations})
            (self.params_pi[phase][player],
            self.opt_states_pi[phase][player],
            loss_val) = self.jitted_updates_pi[phase][player](**update_kwargs)
            
            avg_loss += loss_val
            num_batches_processed += 1
        
        del all_experiences_for_training 
        gc.collect()
        return avg_loss / num_batches_processed if num_batches_processed > 0 else None

    def _get_q_value_dataloader_from_arrays(self, all_samples: Dict[str, np.ndarray], phase):
        batch_size = self._batch_size_q_value[phase]
        num_total_samples = len(all_samples['info_state'])
        if num_total_samples < batch_size: # Not enough samples to form even one batch
            return iter([]) 
        
        rng_key = self._next_rng_key()
        shuffled_indices = jax.random.permutation(rng_key, num_total_samples)
        for i in range(0, num_total_samples - batch_size + 1, batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]
            
            info_states = all_samples['info_state'][batch_indices].astype(jnp.float32) # Phase-specific info_state
            if self._data_augmentors[phase] is not None:
                rng_key = self._next_rng_key()
                info_states = self._data_augmentors[phase](info_states, rng_key)

            yield (info_states,
                   all_samples['action_taken'][batch_indices], # Global action ID
                   all_samples['target_q_value'][batch_indices])
        
    def _get_average_policy_dataloader_from_arrays(self, all_samples: Dict[str, np.ndarray], phase):
        batch_size = self._batch_size_avg_policy[phase]
        num_total_samples = len(all_samples['info_state'])
        if num_total_samples < batch_size:
            return iter([])
        
        shuffled_indices = np.random.permutation(num_total_samples)
        for i in range(0, num_total_samples - batch_size + 1, batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]

            info_states = all_samples['info_state'][batch_indices].astype(jnp.float32) # Phase-specific info_state
            if self._data_augmentors[phase] is not None:
                rng_key = self._next_rng_key()
                info_states = self._data_augmentors[phase](info_states, rng_key)

            best_response_batch = all_samples['best_response'][batch_indices]
            # Construct one-hot tensors for the loss function
            best_response_one_hot = jax.nn.one_hot(
                jnp.squeeze(best_response_batch.astype(jnp.int32)), # Ensure indices are int32 and 1D for one_hot
                num_classes=self._global_num_actions, 
                dtype=jnp.float32 # Output float32 for loss
            )

            legal_actions_masks = all_samples['legal_action_mask'][batch_indices]
            if self._average_weighting_mode != 'vanilla':
                sample_iterations = all_samples['iterations'][batch_indices]
                yield (info_states, best_response_one_hot, legal_actions_masks,
                       sample_iterations.astype(jnp.float32))
            else:
                yield (info_states, best_response_one_hot, legal_actions_masks)

    def _get_saved_iterations(self, player, phase, net_type):
        dir_path = get_dir(self._save_dir_nets, player, phase, net_type)
        if not os.path.exists(dir_path):
            return set()
        files = [f for f in os.listdir(dir_path) if f.endswith('.msgpack')]
        iterations = set()
        for f in files:
            try:
                iter_num = int(f.split('_iter')[1].split('.')[0])
                iterations.add(iter_num)
            except (IndexError, ValueError):
                continue
        return iterations

    def _load_combine_prune_save_experiences(self, player: int, phase: int, buffer_type: str) -> Dict[str, np.ndarray]:
        buffer_dir = get_dir(self._save_dir_buffers, player, phase, buffer_type)
        capacity = self._q_value_memory_capacity[phase] if buffer_type == "q" else self._avg_policy_memory_capacity[phase]
        
        # Get all .npz files in directory
        if not os.path.exists(buffer_dir):
            return {}

        all_files = self._existing_files_filter[buffer_type]

        if not all_files:
            return {}
        
        # Load and concatenate all experiences
        accumulated_data_parts = collections.defaultdict(list)
        for fname in all_files:
            file_path = os.path.join(buffer_dir, fname)
            try:
                with open(file_path, 'rb') as f:
                    file = pickle.load(f)
                    for dict_key in file.keys():
                        accumulated_data_parts[dict_key].append(file[dict_key])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not accumulated_data_parts:
            return {}
        
        # Concatenate all data
        final_data = {}
        for dict_key, list_of_arrays in accumulated_data_parts.items():
            if list_of_arrays:
                final_data[dict_key] = jnp.concatenate(list_of_arrays, axis=0)
        
        total_experiences = len(final_data[next(iter(final_data.keys()))])
        
        # Apply reservoir sampling if over capacity
        if total_experiences > capacity:
            rng_key = self._next_rng_key()
            indices = jax.random.choice(rng_key, total_experiences, shape=(capacity,), replace=False)
            for dict_key in final_data:
                final_data[dict_key] = final_data[dict_key][indices]
            total_experiences = capacity
        
        # Save consolidated file and remove originals
        if all_files:
            consolidated_filename = f"consolidated_iter{self._iteration}.pkl"
            consolidated_path = os.path.join(buffer_dir, consolidated_filename)
            # If consolidated file already exists, rename it with _backup suffix
            if os.path.exists(consolidated_path):
                backup_path = consolidated_path.replace('.pkl', '_backup.pkl')
                os.rename(consolidated_path, backup_path)
            with open(consolidated_path, 'wb') as f:
                pickle.dump(final_data, f)
            
            # Remove original files
            for fname in all_files:
                if fname != consolidated_filename: # Don't delete the file just saved!
                    try:
                        os.remove(os.path.join(buffer_dir, fname))
                    except OSError:
                        print(f"Warning: Failed to remove file {fname}")
        
        return final_data
    
    def _next_rng_key(self) -> jax.Array:
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey