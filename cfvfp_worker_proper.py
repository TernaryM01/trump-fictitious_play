import os

import numpy as np
import pickle

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pyspiel

from typing import Dict, List, Any, Tuple, Optional

def get_dir(save_dir_buffers: str, player_idx: int, phase_idx: int, buffer_type: str) -> str:
    # buffer_type is "q" or "pi"
    path = os.path.join(save_dir_buffers, 
                        f"player{player_idx}",
                        f"phase{phase_idx}",
                        f"{buffer_type}_data")
    os.makedirs(path, exist_ok=True)
    return path

def run_worker_traversals(worker_args: Dict) -> Optional[Tuple[List[int], List[int]]]:
    """
    Main worker function that performs traversals and saves experiences.
    
    Returns: None
    """
    jax.config.update('jax_platform_name', 'cpu')
    worker = GameTraversalWorker(worker_args)
    return worker.run()

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
        self.record_pi = args['record_pi']
        self.infostate_transformers_q = args['info_state_tensor_transformers_q']
        self.infostate_transformers_pi = args['info_state_tensor_transformers_pi']
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
        
        # Initialize game
        self.game = pyspiel.load_game(self.game_name)
        self.num_phases = len(self.q_models)
        
        # Initialize experience buffers
        self.q_value_target_memories_p = [[] for _ in range(self.num_phases)]
        self.best_response_memories_p = [[] for _ in range(self.num_phases)]
        
        # Setup JIT compiled functions
        self._setup_jitted_functions()
    
    def _setup_jitted_functions(self):
        """Setup JIT compiled inference functions for all phases and players."""
        self.jitted_inference_q = []
        self.jitted_inference_pi = []
        
        for phase in range(self.num_phases):
            # Q-value inference for traversing player
            q_inference_fn = self._get_jitted_best_response(
                self.q_models[phase], 
                self.action_transformers[phase]
            )
            self.jitted_inference_q.append(q_inference_fn)
            
            # Policy inference for all players
            phase_pi_inferences = []
            for _ in range(self.num_players if not self.uniform else 1):
                pi_inference_fn = self._get_jitted_avg_policy(
                    self.pi_models[phase],
                    self.action_transformers[phase]
                )
                phase_pi_inferences.append(pi_inference_fn)
            self.jitted_inference_pi.append(phase_pi_inferences)
    
    def _get_jitted_best_response(self, q_model_instance, action_transformer_fn):
        @jax.jit
        def get_q_br(params_q_value: Any, info_state_transformed: jax.Array, legal_actions_mask_global: jax.Array):
            q_values_net_output = q_model_instance.apply({'params': params_q_value}, info_state_transformed)[0]
            q_values_global = action_transformer_fn(q_values_net_output)
            
            masked_q_values = jnp.where(legal_actions_mask_global, q_values_global, jnp.finfo(q_values_global.dtype).min)
            return jnp.argmax(masked_q_values) # Global action ID
        return get_q_br

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

    def _traverse_game_tree(self, state: pyspiel.State, traverser_player: int, main: bool = True) -> float:
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
            return self._traverse_game_tree(state, traverser_player, main)

        active_player = state.current_player()
        
        if self.revelation_transformer is None or active_player != traverser_player:
            infostate_raw = jnp.array(state.information_state_tensor(active_player), dtype=jnp.float32)
        else:
            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
            infostate_raw = jnp.array(
                self.revelation_transformer([state.information_state_tensor(p) for p in range(self.num_players)],
                                            active_player, self.revelation_intensity, rng_subkey),
                dtype=jnp.float32)
        
        phase = self.phase_classifier_fn(infostate_raw)
        infostate_pi = self.infostate_transformers_pi[phase](infostate_raw)
        if active_player == traverser_player:
            infostate_q = self.infostate_transformers_q[phase](infostate_raw)
        legal_actions_mask = jnp.array(state.legal_actions_mask(active_player), dtype=bool)
        
        if active_player == traverser_player:
            best_response = self.jitted_inference_q[phase](
                self.params_q[phase],
                jnp.expand_dims(infostate_q, axis=0),
                jnp.expand_dims(legal_actions_mask, axis=0)
            )

            # Record best response for policy training
            if self.record_pi:
                data_to_add_pi = {
                    'info_state': infostate_pi.astype(jnp.float16),
                    'best_response': best_response.astype(jnp.uint8),
                    'legal_action_mask': legal_actions_mask
                }
                if self.average_weighting_mode != 'vanilla':
                    data_to_add_pi['iteration'] = jnp.array([self.iteration], dtype=jnp.uint16)
                self.best_response_memories_p[phase].append(data_to_add_pi)

            best_response_item = best_response.item()

            # Exploration branching
            if main:
                legal_actions = state.legal_actions(active_player)
                for action in legal_actions:
                    if action != best_response_item:
                        achieved_value_exploratory = self._traverse_game_tree(state.child(action), traverser_player, main=False)
                        # Record Q-value target for exploratory action, but don't return it to parent
                        self.q_value_target_memories_p[phase].append({
                            'info_state': infostate_q.astype(jnp.float16),
                            'action_taken': jnp.array([action], dtype=jnp.uint8),
                            'target_q_value': jnp.array([achieved_value_exploratory], dtype=jnp.float16),
                        })

            # Main branch with best action
            state.apply_action(best_response_item)
            achieved_value = self._traverse_game_tree(state, traverser_player, main)
            self.q_value_target_memories_p[phase].append({
                'info_state': infostate_q.astype(jnp.float16),
                'action_taken': jnp.array([best_response], dtype=jnp.uint8), 
                'target_q_value': jnp.array([achieved_value], dtype=jnp.float16),
            })
            
            return achieved_value
        else: 
            effective_player = active_player if not self.uniform else 0
            opponent_avg_policy = self.jitted_inference_pi[phase][effective_player](
                self.params_pi[phase][effective_player],
                jnp.expand_dims(infostate_pi, axis=0),
                jnp.expand_dims(legal_actions_mask, axis=0)
            )
            opponent_avg_policy /= jnp.sum(opponent_avg_policy)
            # Sample an action according to the opponent's average strategy
            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
            sampled_action_opp = jax.random.choice(rng_subkey, self.global_num_actions, p=opponent_avg_policy)

            state.apply_action(sampled_action_opp.item())
            return self._traverse_game_tree(state, traverser_player, main)

    def save_in_memory_buffer_to_disk(self, player: int, phase: int, buffer_type: str,
                                      in_memory_buffer: List[Dict[str, jnp.ndarray]]):
        """Save in-memory buffer to disk with worker ID in filename."""
        if in_memory_buffer:
            keys = in_memory_buffer[0].keys()
            data_to_save_on_disk = {key: jnp.stack([exp[key] for exp in in_memory_buffer]) for key in keys}
        
            buffer_dir = get_dir(self.save_dir_buffers, player, phase, buffer_type)
            new_filename = f"exp_iter{self.iteration}_w{self.worker_id}.pkl"
            new_filepath = os.path.join(buffer_dir, new_filename)
            
            try:
                with open(new_filepath, 'wb') as f:
                    pickle.dump(data_to_save_on_disk, f)
            except Exception as e:
                print(f"Worker {self.worker_id}: Error saving batch to {new_filepath}: {e}")

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
            if self.record_pi:
                self.save_in_memory_buffer_to_disk(
                    self.player, phase, "pi", self.best_response_memories_p[phase]
                )
        
        return None