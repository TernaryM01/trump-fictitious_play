# . . . CODE OMITTED . . .

class GameTraversalWorker:
    def __init__(self, args: Dict):
        jax.config.update('jax_platform_name', 'cpu')

        self.worker_id = args['worker_id']
        self.num_traversals = args['num_traversals']
        self.player = args['player']
        self.uniform = args['uniform']
        self.game_name = args['game_name']
        self.params_q = args['q_params_all_phases']
        self.params_pi = args['pi_params_all_phases_all_players']
        self.q_models = args['q_models']
        self.pi_models = args['pi_models']
        self.info_state_tensor_transformers_q = args['info_state_tensor_transformers_q']
        self.info_state_tensor_transformers_pi = args['info_state_tensor_transformers_pi']
        self.action_transformers = args['action_transformers']
        self.phase_classifier_fn = args['phase_classifier_fn']
        self.revelation_transformer = args['revelation_transformer']
        self.revelation_intensity = args['revelation_intensity']
        self.num_players = args['num_players']
        self.global_num_actions = args['global_num_actions']
        self.iteration = args['iteration']
        self.average_weighting_mode = args['average_weighting_mode']
        self.save_dir_buffers = args['save_dir_buffers']
        self.rng_key = args['rng_key']
        
        # . . . CODE OMITTED . . .

    def _traverse_game_tree(self, state: pyspiel.State, traverser_player: int) -> float:
        if state.is_terminal():
            return state.returns()[traverser_player]
        elif state.is_chance_node():
            chance_outcome_actions, chance_outcome_probs = zip(*state.chance_outcomes())
            chance_outcome_probs = jnp.array(chance_outcome_probs, dtype=jnp.float32)
            chance_outcome_probs /= jnp.sum(chance_outcome_probs)

            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
            sampled_action = jax.random.choice(rng_subkey, jnp.array(chance_outcome_actions),
                                               p=jnp.array(chance_outcome_probs, dtype=jnp.float32))
            sampled_action = sampled_action.item()

            state.apply_action(sampled_action)
            return self._traverse_game_tree(state, traverser_player)

        active_player = state.current_player()
        
        if self.revelation_transformer is None or active_player != traverser_player:
            full_info_state_jax = jnp.array(state.information_state_tensor(active_player), dtype=jnp.float32)
        else:
            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
            full_info_state_jax = jnp.array(
                self.revelation_transformer([state.information_state_tensor(p) for p in range(self.num_players)],
                                            active_player, self.revelation_intensity, rng_subkey),
                dtype=jnp.float32)
        
        phase = self.phase_classifier_fn(full_info_state_jax)
        # Use Q transformer for traversing player, policy transformer for opponents
        if active_player == traverser_player:
            info_state_transformed_jax = self.info_state_tensor_transformers_q[phase](full_info_state_jax)
        else:
            info_state_transformed_jax = self.info_state_tensor_transformers_pi[phase](full_info_state_jax)
        legal_actions_mask_global_jax = jnp.array(state.legal_actions_mask(active_player), dtype=bool)
        # Add batch dimension for network inference
        info_state_transformed_jax = jnp.expand_dims(info_state_transformed_jax, axis=0)
        legal_actions_mask_global_jax = jnp.expand_dims(legal_actions_mask_global_jax, axis=0)
        
        if active_player == traverser_player:
            best_response = self.jitted_inference_q[phase](
                self.params_q[phase], 
                info_state_transformed_jax, 
                legal_actions_mask_global_jax
            )
            info_state_transformed_jax = info_state_transformed_jax.squeeze(axis=0).astype(jnp.float16)

            data_to_add_pi = {
                'info_state': info_state_transformed_jax,
                'best_response': best_response.astype(jnp.uint8),
                'legal_action_mask': legal_actions_mask_global_jax.squeeze(axis=0)
            }
            if self.average_weighting_mode != 'vanilla':
                data_to_add_pi['iteration'] = jnp.array([self.iteration], dtype=jnp.uint16)
            self.best_response_memories_p[phase].append(data_to_add_pi)
            
            state.apply_action(best_response.item())
            achieved_value = self._traverse_game_tree(state, traverser_player)
            
            self.q_value_target_memories_p[phase].append({
                'info_state': info_state_transformed_jax,
                'action_taken': jnp.array([best_response], dtype=jnp.uint8), 
                'target_q_value': jnp.array([achieved_value], dtype=jnp.float16),
            })
            return achieved_value
        else: 
            effective_player = active_player if not self.uniform else 0
            opponent_avg_policy_jax_global = self.jitted_inference_pi[phase][effective_player](
                self.params_pi[phase][effective_player], 
                info_state_transformed_jax, 
                legal_actions_mask_global_jax
            )
            opponent_avg_policy_jax_global /= jnp.sum(opponent_avg_policy_jax_global)
            # Sample an action according to the opponent's average strategy
            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
            sampled_action_opp = jax.random.choice(rng_subkey, self.global_num_actions, p=opponent_avg_policy_jax_global)

            state.apply_action(sampled_action_opp.item())
            return self._traverse_game_tree(state, traverser_player)

    # . . . CODE OMITTED . . .

    def run(self) -> Optional[Tuple[List[int], List[int]]]:
        """Run traversals and save experiences."""
        # Perform traversals
        root_state = self.game.new_initial_state()
        
        for traversal in range(self.num_traversals):
            try:
                self._traverse_game_tree(root_state.clone(), self.player)
            except Exception as e:
                import traceback
                print(f"Worker {self.worker_id}: Traversal {traversal} failed with error: {e}")
                print(f"Worker {self.worker_id}: Full traceback:")
                traceback.print_exc()
                break
        
        for phase in range(self.num_phases):
            self.save_in_memory_buffer_to_disk(
                self.player, phase, "q", self.q_value_target_memories_p[phase]
            )
            self.save_in_memory_buffer_to_disk(
                self.player, phase, "pi", self.best_response_memories_p[phase]
            )
        
        return None