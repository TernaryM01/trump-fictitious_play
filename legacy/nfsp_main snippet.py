# . . . CODE OMITTED . . .

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

                 data_augmentor: Optional[Callable[[jax.Array], jax.Array]] = None,

                 revelation_transformer: Optional[Callable[[List[jax.Array], int, float], jax.Array]] = None,
                 revelation_intensity: Sequence[float] = [1.0, 0.0],
                 revelation_decay_mode: str = 'linear',

                 uniform: bool = False,

                 # Training parameters
                 num_iterations: int = 100,
                 num_traversals_per_player: int = 100,
                 learning_rate: float = 1e-3,
                 
                 reinitialize_q_value_networks: bool = False,
                 average_weighting_mode: str = 'vanilla',

                 save_dir_buffers: str = "cfvfp_buffers",
                 save_dir_nets: str = "cfvfp_nets",
                 num_workers = None,
                 seed: int = 42):
        
        # . . . CODE OMITTED . . .

        # Buffer Management

        self._q_value_target_memories_p: List[List[List[Dict[str, np.ndarray]]]] = \
            [[[] for _ in range(self._num_players)] for _ in range(self._num_phases)]
        self._best_response_memories_p: List[List[List[Dict[str, np.ndarray]]]] = \
            [[[] for _ in range(self._num_players)] for _ in range(self._num_phases)]
        
        self._save_dir_buffers = save_dir_buffers
        os.makedirs(self._save_dir_buffers, exist_ok=True)
        self._save_dir_nets = save_dir_nets
        os.makedirs(self._save_dir_nets, exist_ok=True)
        
    # . . . CODE OMITTED . . .

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

            revelation_start, revelation_finish = self._revelation_intensity
            if self._revelation_decay_mode == 'exponential':
                decay_rate = (revelation_finish / revelation_start) ** (1 / self._num_iterations)
                revelation_rate = revelation_start * (decay_rate ** i)
            else:  # 'linear'
                revelation_rate = revelation_start + (revelation_finish - revelation_start) * (i / self._num_iterations)

            players_iter_tqdm = tqdm(range(self._num_players), desc="Player", leave=False) if not self._uniform else range(1)
            for player in players_iter_tqdm:
                if self._reinitialize_q_value_networks:
                    for phase in range(self._num_phases):
                        self._reinitialize_q_value_network(player, phase)
                
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
                            'player': player,
                            'uniform': self._uniform,
                            'game_name': self._game.get_type().short_name,
                            'q_params_all_phases': q_params_all_phases,
                            'pi_params_all_phases_all_players': pi_params_all_phases_all_players,
                            'q_models': [self.models_q[phase] for phase in range(self._num_phases)],
                            'pi_models': [self.models_pi[phase] for phase in range(self._num_phases)],
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
                            existing_files[phase][buffer_type] = set([f for f in os.listdir(buffer_dir) if f.endswith('.pkl')])
                        else:
                            existing_files[phase][buffer_type] = set()

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

    # . . . CODE OMITTED . . .