from typing import Sequence, Dict, Tuple, List, Any, Generator, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

# --- Tensor Specification, Slicing Helper, and Phase Detection Indices ---
TENSOR_COMPONENT_SPEC = [
    ("Hand", 52), 
    ("BidCards", 4 * 5),       # 20
    ("TrumpSuit", 4),          
    ("RoundBidStatus", 1),     
    ("History", 13 * (4 + 4 * 5)),   # 312
    ("OpponentGraveyard", 3 * 52), # 156
    ("ANTC", 4),               
    ("BreakOccurred", 1),      
    ("CurrentTrickCards", 4 * 5), # 20
    ("CurrentTrickLeader", 4), 
    ("CurrentTrickTrumpUncertainty", 13),    # NEW ITEM #11
    ("CurrentTrickNumber", 1)  
]

GLOBAL_NUM_ACTIONS = 52

_tensor_component_slices: Dict[str, slice] = {}
_current_offset = 0
for _name, _size in TENSOR_COMPONENT_SPEC:
    _tensor_component_slices[_name] = slice(_current_offset, _current_offset + _size)
    _current_offset += _size

# Define indices for phase classification using the slices
INDEX_ROUND_BID_STATUS = _tensor_component_slices["RoundBidStatus"].start
INDEX_FIRST_BID_CARD_FEATURE = _tensor_component_slices["BidCards"].start
INDEX_HAND_START = _tensor_component_slices["Hand"].start
INDEX_HAND_END = _tensor_component_slices["Hand"].stop
INDEX_BID_CARDS_END = _tensor_component_slices["BidCards"].stop

dummy_infostate = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 12.0, 1.0, 0.0, 0.0, 0.0, 13.0, 1.0, 0.0, 0.0, 0.0, 11.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, -1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0]

def print_graveyards(infostate_tensor):
    """
    Prints the graveyards section of the Trump game information state tensor
    in a human-readable format.
    
    Args:
        infostate_tensor: The complete information state tensor (588 dimensions)
    
    The graveyard section starts at offset 356 and contains 156 values:
    - 3 opponents × 52 cards each
    - Values: -1 (has), 0 (unknown), 1 (had), 2 (never had)
    """
    
    # Constants from the game
    NUM_PLAYERS = 4
    NUM_CARDS = 52
    NUM_SUITS = 4
    NUM_CARDS_PER_SUIT = 13
    
    # Calculate offset to graveyard section (section 6)
    offset = 0
    offset += 52    # 1. Hand
    offset += 20    # 2. Bid Cards (4 * 5)
    offset += 4     # 3. Trump Suit
    offset += 1     # 4. Round Bid Status
    offset += 312   # 5. History (13 * (4 + 4 * 5))
    # Now at graveyard section (156 values)
    
    # Graveyard section: 3 opponents * 52 cards = 156 dimensions
    graveyard_start = offset
    
    # Suit and rank names for display
    suit_names = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades
    rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    
    def card_index_to_name(card_idx):
        """Convert card index (0-51) to name like 'CA', 'D2', etc."""
        suit_idx = card_idx // NUM_CARDS_PER_SUIT
        rank_idx = card_idx % NUM_CARDS_PER_SUIT
        # Game uses rank 1-13 where 1=Ace, but we display in standard order
        return f"{suit_names[suit_idx]}{rank_names[rank_idx]}"
    
    def format_value(val):
        """Format the graveyard values for display"""
        if val == -1.0:
            return "-1"
        elif val == 0.0:
            return " 0"
        elif val == 1.0:
            return " 1"
        elif val == 2.0:
            return " 2"
        else:
            return f"{val:2.0f}"
    
    print("Graveyards (Opponent Card Knowledge):")
    print("Values: -1=has, 0=unknown, 1=had, 2=never had")
    print()
    
    # Process each opponent's graveyard
    for opponent_idx in range(NUM_PLAYERS - 1):  # 3 opponents
        print(f"Opponent {opponent_idx + 1}:")
        
        opponent_offset = graveyard_start + opponent_idx * NUM_CARDS
        
        # Display by suit for better readability
        for suit_idx in range(NUM_SUITS):
            suit_line = f"{suit_names[suit_idx]}: "
            
            for rank_idx in [1,2,3,4,5,6,7,8,9,10,11,12,0]:
                card_idx = suit_idx * NUM_CARDS_PER_SUIT + rank_idx
                tensor_idx = opponent_offset + card_idx
                
                if tensor_idx < len(infostate_tensor):
                    card_name = card_index_to_name(card_idx)
                    value = infostate_tensor[tensor_idx]
                    suit_line += f"{card_name}:{format_value(value)} "
                else:
                    suit_line += f"?:?? "
            
            print(suit_line)
        
        print()  # Empty line between opponents
    
    # Summary statistics
    print("Summary:")
    for opponent_idx in range(NUM_PLAYERS - 1):
        opponent_offset = graveyard_start + opponent_idx * NUM_CARDS
        
        counts = {-1: 0, 0: 0, 1: 0, 2: 0}
        for card_idx in range(NUM_CARDS):
            tensor_idx = opponent_offset + card_idx
            if tensor_idx < len(infostate_tensor):
                val = infostate_tensor[tensor_idx]
                # Round to nearest integer for counting
                rounded_val = round(val)
                if rounded_val in counts:
                    counts[rounded_val] += 1
        
        print(f"Opponent {opponent_idx + 1}: "
              f"Has={counts[-1]} Unknown={counts[0]}, Had={counts[1]}, "
              f"Never Had={counts[2]}")

PHASES = ["bid", "ad", "play"]

# --- Phase-Specific Neural Network Definitions for Trump ---

# --- BIDDING PHASE NETWORKS ---
class TrumpBiddingBackbone(nn.Module):
    mlp_features: Sequence[int] = (128, 64)
    backbone_features: Sequence[int] = (64, 32)

    @nn.compact
    def __call__(self, x_hand: jax.Array) -> jax.Array:
        net = x_hand
        for feat in self.mlp_features:
            net = nn.relu(nn.Dense(features=feat)(net))
        backbone_output = net
        for i, feat_size in enumerate(self.backbone_features):
            current_layer = nn.Dense(features=feat_size, name=f"dense_{i}")(backbone_output)
            current_layer = nn.relu(current_layer)
            if i < len(self.backbone_features) - 1:
                current_layer = nn.LayerNorm(use_bias=True, use_scale=True, name=f"layernorm_{i}")(current_layer)
            backbone_output = current_layer
        return backbone_output

class TrumpBiddingQNet(nn.Module):
    # Outputs Q-values for the GLOBAL_GAME_ACTIONS space
    @nn.compact
    def __call__(self, x_hand: jax.Array) -> jax.Array:
        backbone = TrumpBiddingBackbone(name="q_backbone")
        shared_output = backbone(x_hand)
        q_values = nn.Dense(features=GLOBAL_NUM_ACTIONS, name="q_head")(shared_output)
        return q_values

class TrumpBiddingPolicyNet(nn.Module):
    # Outputs policy logits for the GLOBAL_GAME_ACTIONS space
    @nn.compact
    def __call__(self, x_hand: jax.Array, legal_actions_mask_global: jax.Array) -> jax.Array:
        backbone = TrumpBiddingBackbone(name="pi_backbone")
        shared_output = backbone(x_hand)
        policy_logits = nn.Dense(features=GLOBAL_NUM_ACTIONS, name="policy_head")(shared_output)
        masked_policy_logits = jnp.where(legal_actions_mask_global, policy_logits, jnp.finfo(policy_logits.dtype).min)
        return masked_policy_logits

# --- ASCEND/DESCEND PHASE NETWORKS ---
class TrumpAD_Backbone(nn.Module):
    # Input: Hand + 4 Revealed Bid Cards (72 features)
    mlp_features: Sequence[int] = (64, 32)
    backbone_features: Sequence[int] = (32, 16)
    @nn.compact
    def __call__(self, x_hand_and_bids: jax.Array) -> jax.Array:
        net = x_hand_and_bids
        for feat in self.mlp_features:
            net = nn.relu(nn.Dense(features=feat)(net))
        backbone_output = net
        for i, feat_size in enumerate(self.backbone_features):
            current_layer = nn.Dense(features=feat_size, name=f"ad_backbone_dense_{i}")(backbone_output)
            current_layer = nn.relu(current_layer)
            if i < len(self.backbone_features) - 1:
                current_layer = nn.LayerNorm(use_bias=True, use_scale=True, name=f"ad_backbone_layernorm_{i}")(current_layer)
            backbone_output = current_layer
        return backbone_output

class TrumpAD_QNet(nn.Module):
    # Outputs 2 Q-values (for Ascend, Descend local actions)
    @nn.compact
    def __call__(self, x_hand_and_bids: jax.Array) -> jax.Array:
        backbone = TrumpAD_Backbone(name="ad_q_backbone")
        shared_output = backbone(x_hand_and_bids)
        q_values_local = nn.Dense(features=2, name="ad_q_head")(shared_output)
        return q_values_local

class TrumpAD_PolicyNet(nn.Module):
    # Outputs 2 logits (for Ascend, Descend local actions)
    @nn.compact
    def __call__(self, x_hand_and_bids: jax.Array, legal_actions_mask_for_ad_phase: jax.Array) -> jax.Array:
        # legal_actions_mask_for_ad_phase is ignored
        backbone = TrumpAD_Backbone(name="ad_pi_backbone")
        shared_output = backbone(x_hand_and_bids)
        policy_logits_local = nn.Dense(features=2, name="ad_policy_head")(shared_output)
        return policy_logits_local

# --- TRICK-TAKING PLAY PHASE NETWORKS ---
class TrumpPlayBackbone(nn.Module):
    # Processes full 523-dim tensor
    hand_mlp_features: Sequence[int] = (64, 32)
    bid_cards_mlp_features: Sequence[int] = (32, 64)
    history_gru_features: int = 128
    opponent_graveyard_mlp_features: Sequence[int] = (128, 64)
    current_trick_mlp_features: Sequence[int] = (64, 64)
    current_trick_trump_uncertainty_mlp_features: Sequence[int] = (32, 32) # NEW HYPERPARAMETER

    shared_backbone_features: Sequence[int] = (256, 128)
    
    _num_players_const: int = 4
    _features_per_card_const: int = 5
    _num_max_history_tricks_const: int = 13
    _num_cards_const: int = 52
    _trump_uncertainty_size_const: int = 13 # For the new component

    # Define GRUCell as a submodule in setup for use with nn.scan
    def setup(self):
        self.history_gru_cell = nn.GRUCell(
            features=self.history_gru_features, 
            name="play_history_gru_cell"
        )
        self.scan_gru = nn.scan(
            partial(nn.GRUCell, features=self.history_gru_features),
            variable_broadcast="params",
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1
        )

    @nn.compact
    def __call__(self, x_full_tensor: jax.Array) -> jax.Array:
        processed_features = []
        
        # 1. Hand
        hand_f = x_full_tensor[..., _tensor_component_slices["Hand"]]
        for feat in self.hand_mlp_features:
            hand_f = nn.relu(nn.Dense(features=feat)(hand_f))
        processed_features.append(hand_f)

        # 2. Bid Cards 
        bid_cards_f = x_full_tensor[..., _tensor_component_slices["BidCards"]]
        for feat in self.bid_cards_mlp_features:
            bid_cards_f = nn.relu(nn.Dense(features=feat)(bid_cards_f))
        processed_features.append(bid_cards_f)

        # 3. Trump Suit 
        processed_features.append(x_full_tensor[..., _tensor_component_slices["TrumpSuit"]])

        # 4. Round Bid Status
        processed_features.append(x_full_tensor[..., _tensor_component_slices["RoundBidStatus"]])

        # 5. History (GRU)
        history_input_flat = x_full_tensor[..., _tensor_component_slices["History"]]
        history_input_reshaped = history_input_flat.reshape(
            history_input_flat.shape[:-1] + 
            (self._num_max_history_tricks_const,
             self._num_players_const + self._num_players_const * self._features_per_card_const)
        )
        
        # Initialize carry for the GRUCell
        # The input_shape for initialize_carry is the shape of a single time step's input
        batch_dims_for_gru_carry = history_input_reshaped.shape[:-2] # e.g., (batch_size,)
        features_per_step = history_input_reshaped.shape[-1]
        input_shape_for_carry_init = batch_dims_for_gru_carry + (features_per_step,)
        
        # Use a fixed key for carry init if it's just a zero state
        # Or self.make_rng('params') if this was during model.init() for learnable initial carry (not typical for GRU)
        key_for_gru_init = jax.random.key(0) 
        
        initial_carry = self.history_gru_cell.initialize_carry( # Use the cell instance from setup
            key_for_gru_init, 
            input_shape_for_carry_init
        )
        
        _final_carry, history_outputs_sequence = self.scan_gru(name="history_scan_gru")(
            initial_carry,
            history_input_reshaped
        )
        history_f = _final_carry  # Use final carry instead of slicing sequence; Shape: (batch_size, history_gru_features)
        processed_features.append(history_f)

        # 6. Opponent Graveyard
        opp_graveyard_flat = x_full_tensor[..., _tensor_component_slices["OpponentGraveyard"]]
        opp_graveyard_reshaped = opp_graveyard_flat.reshape(
            opp_graveyard_flat.shape[:-1] + (self._num_players_const - 1, self._num_cards_const)
        )
        
        class OpponentGraveyardMLP(nn.Module): # Defined nested or outside
            features: Sequence[int]
            @nn.compact
            def __call__(self, x):
                for i, feat in enumerate(self.features):
                    x = nn.Dense(features=feat, name=f"dense_{i}")(x)
                    if i < len(self.features) - 1:
                        x = nn.relu(x)
                return x
        
        VmapOpponentMLP = nn.vmap(
            OpponentGraveyardMLP,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=-2,
            out_axes=-2
        )
        vmap_module = VmapOpponentMLP(features=self.opponent_graveyard_mlp_features, name="play_opp_grave_mlp_module")
        processed_opp_graveyards = vmap_module(opp_graveyard_reshaped)

        opp_graveyard_f = processed_opp_graveyards.reshape(
            processed_opp_graveyards.shape[:-2] + (-1,) 
        )
        processed_features.append(opp_graveyard_f)

        # 7. ANTC 
        processed_features.append(x_full_tensor[..., _tensor_component_slices["ANTC"]])

        # 8. Break Occurred
        processed_features.append(x_full_tensor[..., _tensor_component_slices["BreakOccurred"]])

        # 9. Current Trick Cards
        curr_trick_f = x_full_tensor[..., _tensor_component_slices["CurrentTrickCards"]]
        for feat in self.current_trick_mlp_features:
            curr_trick_f = nn.relu(nn.Dense(features=feat)(curr_trick_f))
        processed_features.append(curr_trick_f)
        
        # 10. Current Trick Leader
        processed_features.append(x_full_tensor[..., _tensor_component_slices["CurrentTrickLeader"]])

        # --- NEW Item #11: Current Trick Trump Uncertainty ---
        trump_uncertainty_f = x_full_tensor[..., _tensor_component_slices["CurrentTrickTrumpUncertainty"]]
        for feat in self.current_trick_trump_uncertainty_mlp_features:
            trump_uncertainty_f = nn.relu(nn.Dense(features=feat)(trump_uncertainty_f))
        processed_features.append(trump_uncertainty_f)

        # 12. Current Trick Number
        processed_features.append(x_full_tensor[..., _tensor_component_slices["CurrentTrickNumber"]])

        combined_features = jnp.concatenate(processed_features, axis=-1)
        
        backbone_output = combined_features
        for i, feat_size in enumerate(self.shared_backbone_features):
            current_layer = nn.Dense(features=feat_size, name=f"play_shared_backbone_dense_{i}")(backbone_output)
            current_layer = nn.relu(current_layer)
            if i < len(self.shared_backbone_features) - 1:
                 current_layer = nn.LayerNorm(use_bias=True, use_scale=True, name=f"play_shared_backbone_layernorm_{i}")(current_layer)
            backbone_output = current_layer
            
        return backbone_output

class TrumpPlayQNet(nn.Module):
    @nn.compact
    def __call__(self, x_full_tensor: jax.Array) -> jax.Array:
        backbone = TrumpPlayBackbone(name="play_q_backbone")
        shared_output = backbone(x_full_tensor)
        q_values = nn.Dense(features=GLOBAL_NUM_ACTIONS, name="play_q_head")(shared_output)
        return q_values

class TrumpPlayPolicyNet(nn.Module):
    @nn.compact
    def __call__(self, x_full_tensor: jax.Array, legal_actions_mask_global: jax.Array) -> jax.Array:
        backbone = TrumpPlayBackbone(name="play_pi_backbone")
        shared_output = backbone(x_full_tensor)
        policy_logits = nn.Dense(features=GLOBAL_NUM_ACTIONS, name="play_policy_head")(shared_output)
        masked_policy_logits = jnp.where(legal_actions_mask_global, policy_logits, jnp.finfo(policy_logits.dtype).min)
        return masked_policy_logits
    
# --- Wrapper Network Models ---

class QValueNetworkWrapper(nn.Module):
    # This wrapper directly calls the phase-specific Q-network (e.g., TrumpBiddingQNet)
    phase_net: nn.Module 
    @nn.compact
    def __call__(self, x_phase_specific_input: jax.Array) -> jax.Array:
        q_values = self.phase_net(x_phase_specific_input)
        return q_values

class PolicyNetworkWrapper(nn.Module):
    phase_net: nn.Module
    @nn.compact
    def __call__(self, x_phase_specific_input: jax.Array, legal_actions_mask_for_phase: jax.Array) -> jax.Array:
        masked_policy_logits = self.phase_net(x_phase_specific_input, legal_actions_mask_for_phase)
        return masked_policy_logits
    
# --- Information State Tensor Transformers ---

def bid_transformer(infostate_tensor: jax.Array) -> jax.Array:
    return infostate_tensor[..., INDEX_HAND_START:INDEX_HAND_END]

def ad_transformer(infostate_tensor: jax.Array) -> jax.Array:
    return infostate_tensor[..., INDEX_HAND_START:INDEX_BID_CARDS_END]

def play_transformer_pi(infostate_tensor: jax.Array) -> jax.Array:
    return infostate_tensor

from trump_z3_bool_current_hand import analyze_hand_cards
def play_transformer_q(infostate_tensor: jax.Array) -> jax.Array:
    # Convert to numpy for processing
    infostate_tensor = np.array(infostate_tensor)
    original_shape = infostate_tensor.shape
    
    # Handle batch dimensions by flattening and processing each sample
    if infostate_tensor.ndim > 1:
        # Flatten batch dimensions
        flattened = infostate_tensor.reshape(-1, original_shape[-1])
        
        # Process each sample in the batch
        processed_samples = []
        for i in range(flattened.shape[0]):
            sample = flattened[i]
            hand_analysis = analyze_hand_cards(sample)
            hand_analysis = np.array(hand_analysis)
            
            # Replace first 52 slots with hand analysis results
            sample[:52] = hand_analysis[:52]
            processed_samples.append(sample)
        
        # Reshape back to original batch shape
        result = np.array(processed_samples).reshape(original_shape)
    else:
        # No batch dimension - process directly
        hand_analysis = analyze_hand_cards(infostate_tensor)
        hand_analysis = np.array(hand_analysis)
        infostate_tensor[:52] = hand_analysis[:52]
        result = infostate_tensor
    
    # Convert back to JAX array
    return jax.numpy.array(result)

# --- Action Transformers ---

ASCEND_GAME_ACTION_ID = 0 
DESCEND_GAME_ACTION_ID = 1

def identity_action_transformer(net_output_global_space: jax.Array) -> jax.Array:
    return net_output_global_space

def ad_local_to_global_action_transformer(ad_net_output_local_2actions: jax.Array) -> jax.Array:
    batch_size = ad_net_output_local_2actions.shape[0]
    # Initialize with a value indicating "not a valid action" or "very unlikely"
    output_global = jnp.full((batch_size, GLOBAL_NUM_ACTIONS), jnp.finfo(ad_net_output_local_2actions.dtype).min) 
    
    output_global = output_global.at[..., ASCEND_GAME_ACTION_ID].set(ad_net_output_local_2actions[..., 0])
    output_global = output_global.at[..., DESCEND_GAME_ACTION_ID].set(ad_net_output_local_2actions[..., 1])
    return output_global

# --- Phase Classifier Function ---
def trump_phase_classifier(infostate_tensor: np.ndarray) -> int:
    """Returns phase index: 0 for 'bid', 1 for 'ad', 2 for 'play'."""
    if infostate_tensor[INDEX_ROUND_BID_STATUS] != 0: 
        return 2
    else: 
        return 0 if infostate_tensor[INDEX_FIRST_BID_CARD_FEATURE] == 0.0 else 1
    
# --- Revelation Transformer ---
def reveal_p0(player_tensors):
    """Modify player 0's tensor to reveal all hidden information
    This function is a miniature of Numpy elegance & mastery."""    
    tensor_p0_rev = player_tensors[0]
    
    # Step 1: Modify OpponentGraveyard (indices 389 to 544)
    # Extract hands from each player's tensor (indices 0 to 51)

    # Vectorized hand extraction for all players at once
    hands_mask = player_tensors[:, :52] == 1.0
    # Update graveyard for opponents (players 1, 2, 3)
    opponent_hands = hands_mask[1:4, :]  # Boolean mask for opponent's hand
    # Vectorized: update all 3 graveyards at once
    graveyard_section = tensor_p0_rev[389:545].reshape(3, 52)
    # Unconditionally set '-1' where opponents hold cards
    graveyard_section[opponent_hands] = -1
    # Set '2' where current value is '0' (avoid overwriting '1's)
    graveyard_section[graveyard_section == 0] = 2
    # Write back the modified graveyard section
    tensor_p0_rev[389:545] = graveyard_section.flatten()
    
    # Step 2: Modify CurrentTrickCards (indices 550 to 569)  # DISABLED
    # Copy actual card encoding from the tensor of the player who played it.

    revealed_trick_cards = player_tensors[1:4, 550:555]
    tensor_p0_rev[555:570] = revealed_trick_cards.flatten()

    # Step 3: Revisit graveyards by putting missing info from current trick.
    # Mark cards played in current trick by an opponent with -1.
    # This step is necessary because graveyards are supposed to lag from current trick.
    
    played_mask = np.sum(revealed_trick_cards, axis=1) > 0
    suits = np.argmax(revealed_trick_cards[:, 1:5], axis=1)
    ranks = revealed_trick_cards[:, 0].astype(int)
    ranks = np.where(ranks != 13, ranks, 0)  # Fix the rank of Ace for card index calculation.
    card_idxs = suits * 13 + ranks
    graveyard_indices = 389 + np.arange(3) * 52 + card_idxs
    tensor_p0_rev[graveyard_indices[played_mask]] = -1
    
    # Step 4: Modify CurrentTrickTrumpUncertainty (indices 574 to 586)
    # Determine trump suit from TrumpSuit section (indices 72 to 75)

    # Vectorized trump suit determination
    trump_suit_index = np.argmax(tensor_p0_rev[72:76])
    # Vectorized trump ranks collection
    card_slots = tensor_p0_rev[550:570].reshape(4, 5)
    suit_onehots = card_slots[:, :4]
    ranks = card_slots[:, 4]
    # Determine which cards were played
    played_mask = np.sum(suit_onehots, axis=1) > 0
    if np.any(played_mask):
        played_suits = np.argmax(suit_onehots[played_mask], axis=1)
        played_ranks = ranks[played_mask]
        # Find trump cards that were played
        trump_cards_mask = (played_suits == trump_suit_index) & (played_ranks != 0)
        trump_ranks_played = set(played_ranks[trump_cards_mask].astype(int))
    else:
        trump_ranks_played = set()
    # Vectorized trump uncertainty update
    ranks_array = np.arange(1, 14)
    trump_uncertainty_values = np.where(np.isin(ranks_array, list(trump_ranks_played)), 1, -1)
    tensor_p0_rev[574:587] = trump_uncertainty_values
    
    return tensor_p0_rev

import jax, time
def revelation_transformer(player_tensors, player, intensity=1, rng_key=None):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(hash(str(time.time())) % 2**32)
        
    def partially_reveal_p0(player_tensors, rate, rng_key):
        """
        Partially reveals hidden information in player 0's information state tensor based on the given proportion.
        
        Args:
            tensors (list of lists): List of information state tensors for all players.
            proportion (float): The proportion of positions to reveal (between 0 and 1).
        
        Returns:
            np.array: The modified tensor for player 0 with partial revelation.
        """
        original_tensor = np.array(player_tensors[0])
        revealed_tensor = reveal_p0(np.array(player_tensors))
        
        # Define the indices for the sections to be partially revealed
        graveyard_indices = np.arange(389, 545)
        # rank_indices = np.array([550, 555, 560, 565])  # Disabled for now.
        trump_uncertainty_indices = np.arange(574, 587)
        # all_indices = np.concatenate([graveyard_indices, rank_indices, trump_uncertainty_indices])
        all_indices = np.concatenate([graveyard_indices, trump_uncertainty_indices])
        
        if intensity < 1:
            reveal_mask = jax.random.uniform(rng_key, (len(all_indices),)) < rate
            reveal_indices = all_indices[reveal_mask]
            
            # Replace the selected indices with revealed values
            original_tensor[reveal_indices] = revealed_tensor[reveal_indices]
            
            return original_tensor
        return revealed_tensor
    
    return partially_reveal_p0(player_tensors[player:] + player_tensors[:player], intensity, rng_key)

# --- Data Augmentor ---
import jax
import jax.numpy as jnp
from typing import List, Tuple

# def permute_trump_infostate_suits(infostate_tensor: np.ndarray, 
#                                   suit_permutation: List[int]) -> np.ndarray:
#     """
#     Permute suits in a Trump game information state tensor.
#     WARNING: This function is full of advanced Numpy jujitsu. You're expected to be confused.
#     If you think something's wrong, you're wrong.
    
#     Args:
#         infostate_tensor: JAX array of shape (588,) containing the information state
#         suit_permutation: List of 4 integers representing the permutation mapping
#                          [clubs_to, diamonds_to, hearts_to, spades_to]
#                          where each value is in [0,1,2,3]
    
#     Returns:
#         JAX array with suits permuted according to the permutation scheme
        
#     Raises:
#         ValueError: If the permutation would change the highest bidder
#     """
#     # Constants from the game
#     NUM_PLAYERS = 4
#     NUM_CARDS = 52
#     NUM_SUITS = 4
#     NUM_CARDS_PER_SUIT = 13
#     FEATURES_PER_CARD = 5  # 1 rank + 4 suit features
#     NUM_TRICKS = 13
    
#     # Validate permutation
#     if len(suit_permutation) != NUM_SUITS or set(suit_permutation) != {0, 1, 2, 3}:
#         raise ValueError("suit_permutation must be a permutation of [0,1,2,3]")
    
#     # Helper function to permute 52-card sections efficiently
#     def permute_52_card_sections(section: np.ndarray) -> None:
#         """Permute a 52-card section by rearranging 4 blocks of 13 cards each.
#         Can handle single section (52,) or batch of sections (..., 52)."""
#         # Reshape to (..., 4, 13) for suit-wise operations
#         cards_by_suit = section.reshape(*section.shape[:-1], NUM_SUITS, NUM_CARDS_PER_SUIT)
#         temp = cards_by_suit.copy()
#         for orig_suit in range(NUM_SUITS):
#             new_suit = suit_permutation[orig_suit]
#             cards_by_suit[..., new_suit, :] = temp[..., orig_suit, :]
    
#     # Helper function to permute card features (rank + 4 suit features)
#     def permute_card_features(features: np.ndarray) -> None:
#         """Permute the 4 suit features in a 5-feature card representation."""        
#         # Extract rank (first feature) and suit features (last 4)
#         suit_features = features[..., 1:]  # Shape: (..., 4)
#         temp = suit_features.copy()
#         for orig_suit in range(NUM_SUITS):
#             new_suit = suit_permutation[orig_suit]
#             suit_features[..., new_suit] = temp[..., orig_suit]
    
#     # 1. Permute hand section (52-dim multi-hot)
#     start = 0
#     end = NUM_CARDS
#     hand_section = infostate_tensor[start:end]
#     permute_52_card_sections(hand_section)
    
#     # 2. Permute bid cards section (4 players * 5 features)
#     start = end  
#     end = start + NUM_PLAYERS * FEATURES_PER_CARD
#     bid_cards_section = infostate_tensor[start:end]
#     bid_cards_reshaped = bid_cards_section.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
    
#     # Check if any bidding occurred (non-zero features)
#     if bid_cards_section[0] == 0:  # Rank of this player's bidding card
#         raise ValueError("Permutation is allowed only in trick-taking-phase state.")
    
#     # Validate that permutation won't change highest bidder before applying it
#     bid_values = bid_cards_reshaped[:, 0]
#     max_bid_value = np.max(bid_values)
#     tied_bidders = np.where(bid_values == max_bid_value)[0]
    
#     if len(tied_bidders) > 1:
#         # In case of tie, winner is determined by suit strength (spades > hearts > diamonds > clubs)
#         tied_suits = bid_cards_reshaped[tied_bidders, 1:]  # Shape: (n_tied, 4)
#         orig_suits = np.argmax(tied_suits, axis=1)
#         new_suits = np.array([suit_permutation[orig_suit] for orig_suit in orig_suits])
        
#         # Check if the ordering changes (highest suit index wins)
#         if np.argmax(orig_suits) != np.argmax(new_suits):
#             raise ValueError("Permutation would change the highest bidder due to suit tie-breaking")
    
#     # Now apply the permutation
#     permute_card_features(bid_cards_reshaped)
    
#     # 3. Permute trump suit section (4-dim one-hot)
#     start = end
#     end = start + NUM_SUITS
#     trump_suit_section = infostate_tensor[start:end]
#     infostate_tensor[start:end] = trump_suit_section[suit_permutation]  # Advanced Numpy indexing
    
#     # 5. Permute history section
#     trick_len = NUM_PLAYERS + NUM_PLAYERS * FEATURES_PER_CARD
#     start = end + 1
#     end = start + NUM_TRICKS * trick_len
    
#     # Reshape to separate leaders from cards, then process all cards at once
#     history_section = infostate_tensor[start:end].reshape(NUM_TRICKS, trick_len)
#     # Extract just the card features (skip the 4 leader features per trick)
#     all_history_cards = history_section[:, NUM_PLAYERS:].reshape(NUM_TRICKS * NUM_PLAYERS, FEATURES_PER_CARD)
#     permute_card_features(all_history_cards)
    
#     # 6. Permute graveyard sections (3 opponents * 52 cards each)
#     start = end
#     end = start + (NUM_PLAYERS - 1) * NUM_CARDS
#     graveyards_section = infostate_tensor[start:end].reshape(NUM_PLAYERS - 1, NUM_CARDS)
#     permute_52_card_sections(graveyards_section)
    
#     # 9. Permute current trick cards section (4 players * 5 features)
#     start = end + NUM_PLAYERS + 1
#     end = start + NUM_PLAYERS * FEATURES_PER_CARD
#     current_trick_section = infostate_tensor[start:end].reshape(NUM_PLAYERS, FEATURES_PER_CARD)
#     permute_card_features(current_trick_section)
    
#     return infostate_tensor

def permute_trump_infostate_suits(infostate_tensor: jax.Array, 
                                  suit_permutation: jax.Array) -> jax.Array:
    """
    Permute suits in a Trump game information state tensor (JAX-compatible).
    
    Args:
        infostate_tensor: JAX array of shape (588,) containing the information state
        suit_permutation: JAX array of shape (4,) representing the permutation mapping
                         [clubs_to, diamonds_to, hearts_to, spades_to]
                         where each value is in [0,1,2,3]
    
    Returns:
        JAX array with suits permuted according to the permutation scheme,
        or the original tensor if the permutation would change the highest bidder
    """    
    # Constants from the game
    NUM_PLAYERS = 4
    NUM_CARDS = 52
    NUM_SUITS = 4
    NUM_CARDS_PER_SUIT = 13
    FEATURES_PER_CARD = 5  # 1 rank + 4 suit features
    NUM_TRICKS = 13
    
    # Helper function to permute 52-card sections
    def permute_52_card_sections(section: jax.Array) -> jax.Array:
        """Permute a 52-card section by rearranging 4 blocks of 13 cards each."""
        # Reshape to (..., 4, 13) for suit-wise operations
        cards_by_suit = section.reshape(*section.shape[:-1], NUM_SUITS, NUM_CARDS_PER_SUIT)
        # Use advanced indexing to permute suits
        return cards_by_suit[..., suit_permutation, :].reshape(section.shape)
    
    # Helper function to permute card features (rank + 4 suit features)
    def permute_card_features(features: jax.Array) -> jax.Array:
        """Permute the 4 suit features in a 5-feature card representation."""
        # Extract rank (first feature) and suit features (last 4)
        rank_features = features[..., :1]  # Shape: (..., 1)
        suit_features = features[..., 1:]  # Shape: (..., 4)
        # Permute suit features
        permuted_suit_features = suit_features[..., suit_permutation]
        return jnp.concatenate([rank_features, permuted_suit_features], axis=-1)
    
    # Check if permutation is valid before applying it
    def is_valid_permutation() -> jax.Array:
        """Check if the permutation would change the highest bidder."""
        # Extract bid cards section (4 players * 5 features)
        bid_cards_section = infostate_tensor[NUM_CARDS:NUM_CARDS + NUM_PLAYERS * FEATURES_PER_CARD]
        bid_cards_reshaped = bid_cards_section.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        
        # # Check if any bidding occurred (non-zero rank for first player)
        # bidding_occurred = bid_cards_reshaped[0, 0] > 0
        
        # Check if permutation would change highest bidder
        bid_values = bid_cards_reshaped[:, 0]
        max_bid_value = jnp.max(bid_values)
        
        # Find tied highest bidders
        is_tied = bid_values == max_bid_value
        
        # If multiple tied highest bidders, check if the tie-breaking winner changes
        # Permutation is valid if single winner OR tie-breaking doesn't change winner

        # For each player, get their suit (0-3) if they're tied, -1 otherwise
        orig_tied_suits = jnp.where(
            is_tied,
            jnp.argmax(bid_cards_reshaped[:, 1:], axis=1),  # Original suits for tied players
            -1  # Not tied
        )
        
        # Apply permutation to get new suits
        new_tied_suits = jnp.where(
            orig_tied_suits >= 0,  # Only for tied players
            suit_permutation[orig_tied_suits],  # Map original to new suits
            -1  # Keep -1 for non-tied players
        )
        
        # Find which player wins the tie-breaking (highest suit index)
        orig_winner_idx = jnp.argmax(jnp.where(is_tied, orig_tied_suits, -1))
        new_winner_idx = jnp.argmax(jnp.where(is_tied, new_tied_suits, -1))
        
        return orig_winner_idx == new_winner_idx
    
    # Apply permutation only if valid
    def apply_permutation() -> jax.Array:
        """Apply the suit permutation to all relevant sections."""
        
        # Pre-calculate all section boundaries
        hand_start = 0
        hand_end = NUM_CARDS
        
        bid_start = hand_end
        bid_end = bid_start + NUM_PLAYERS * FEATURES_PER_CARD
        
        trump_start = bid_end
        trump_end = trump_start + NUM_SUITS
        
        bid_status_start = trump_end
        bid_status_end = bid_status_start + 1
        
        history_start = bid_status_end
        trick_len = NUM_PLAYERS + NUM_PLAYERS * FEATURES_PER_CARD
        history_end = history_start + NUM_TRICKS * trick_len
        
        graveyard_start = history_end
        graveyard_end = graveyard_start + (NUM_PLAYERS - 1) * NUM_CARDS
        
        antc_break_start = graveyard_end
        antc_break_end = antc_break_start + NUM_PLAYERS + 1
        
        current_trick_start = antc_break_end
        current_trick_end = current_trick_start + NUM_PLAYERS * FEATURES_PER_CARD
        
        # Process sections efficiently
        # 1. Hand section
        hand_section = infostate_tensor[hand_start:hand_end]
        permuted_hand = permute_52_card_sections(hand_section)
        
        # 2. Bid cards section
        bid_cards_section = infostate_tensor[bid_start:bid_end].reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_bid_cards = permute_card_features(bid_cards_section).flatten()
        
        # 3. Trump suit section
        trump_suit_section = infostate_tensor[trump_start:trump_end]
        permuted_trump = trump_suit_section[suit_permutation]
        
        # 4. Bid status (unchanged)
        bid_status_section = infostate_tensor[bid_status_start:bid_status_end]
        
        # 5. History section
        history_section = infostate_tensor[history_start:history_end].reshape(NUM_TRICKS, trick_len)
        all_history_cards = history_section[:, NUM_PLAYERS:].reshape(NUM_TRICKS * NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_history_cards = permute_card_features(all_history_cards)
        
        new_history_section = jnp.concatenate([
            history_section[:, :NUM_PLAYERS],
            permuted_history_cards.reshape(NUM_TRICKS, NUM_PLAYERS * FEATURES_PER_CARD)
        ], axis=1).flatten()
        
        # 6. Graveyard sections
        graveyards_section = infostate_tensor[graveyard_start:graveyard_end].reshape(NUM_PLAYERS - 1, NUM_CARDS)
        permuted_graveyards = permute_52_card_sections(graveyards_section).flatten()
        
        # 7&8. ANTC & break occurred (unchanged)
        antc_break_section = infostate_tensor[antc_break_start:antc_break_end]
        
        # 9. Current trick cards
        current_trick_section = infostate_tensor[current_trick_start:current_trick_end].reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_current_trick = permute_card_features(current_trick_section).flatten()
        
        # Remaining sections (unchanged)
        remaining_section = infostate_tensor[current_trick_end:]
        
        # Single concatenation with fixed number of arrays (JIT-friendly)
        return jnp.concatenate([
            permuted_hand,
            permuted_bid_cards,
            permuted_trump,
            bid_status_section,
            new_history_section,
            permuted_graveyards,
            antc_break_section,
            permuted_current_trick,
            remaining_section
        ])
    
    # Return permuted tensor if valid, otherwise return original
    return jax.lax.cond(
        is_valid_permutation(),
        apply_permutation,
        lambda: infostate_tensor
    )

# import jax
# import jax.numpy as jnp
# def data_augmentor(infostate_tensor: jax.Array, rng_key: jax.Array) -> jax.Array:
#     suit_permutation = jax.random.permutation(rng_key, jnp.array([0, 1, 2, 3]))
#     try:  # Permutation might fail due to changing the highest bidder
#         return jnp.array(permute_trump_infostate_suits(
#             np.array(infostate_tensor), suit_permutation.tolist()
#         ))
#     except:
#         return infostate_tensor

@jax.jit
def data_augmentor(infostate_tensor: jax.Array, rng_key: jax.Array) -> jax.Array:
    """
    JAX-compatible data augmentor that randomly permutes suits.
    
    Args:
        infostate_tensor: JAX array of shape (588,) containing the information state
        rng_key: JAX random key for generating permutation
        
    Returns:
        JAX array with suits permuted if valid, otherwise original tensor
    """
    # suit_permutation = jax.random.permutation(rng_key, jnp.arange(4))
    # return permute_trump_infostate_suits(infostate_tensor, suit_permutation)    
    batched_tensor = jnp.atleast_2d(infostate_tensor)
    batch_size = batched_tensor.shape[0]
    
    # Generate different permutations for each batch element
    rng_keys = jax.random.split(rng_key, batch_size)
    suit_permutations = jax.vmap(
        lambda rng_key: jax.random.permutation(rng_key, jnp.arange(4))
    )(rng_keys)
    
    # Apply permutations to each batch element
    result = jax.vmap(permute_trump_infostate_suits)(batched_tensor, suit_permutations)
    
    # Preserve original shape
    return result.reshape(infostate_tensor.shape)
    

# 1. Instantiate Phase-Specific Network Models
q_bid_model_core = TrumpBiddingQNet()
pi_bid_model_core = TrumpBiddingPolicyNet()
q_ad_model_core = TrumpAD_QNet() 
pi_ad_model_core = TrumpAD_PolicyNet()
q_play_model_core = TrumpPlayQNet()
pi_play_model_core = TrumpPlayPolicyNet()

# 2. Wrap them for the solver
q_value_network_models = [
    QValueNetworkWrapper(phase_net=q_bid_model_core),
    QValueNetworkWrapper(phase_net=q_ad_model_core),
    QValueNetworkWrapper(phase_net=q_play_model_core)
]
avg_policy_network_models = [
    PolicyNetworkWrapper(phase_net=pi_bid_model_core),
    PolicyNetworkWrapper(phase_net=pi_ad_model_core),
    PolicyNetworkWrapper(phase_net=pi_play_model_core)
]

# 3. Define Transformers and Classifier
info_state_tensor_transformers = [bid_transformer, ad_transformer, [play_transformer_q, play_transformer_pi]]
# info_state_tensor_transformers = [bid_transformer, ad_transformer, play_transformer_pi]
data_augmentors = [None, None, data_augmentor]
action_transformers = [
    identity_action_transformer, 
    ad_local_to_global_action_transformer, 
    identity_action_transformer
]