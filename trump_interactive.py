import random
import sys
import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
from trump_utils import (
    TrumpBiddingPolicyNet, TrumpAD_PolicyNet, TrumpPlayPolicyNet,
    PolicyNetworkWrapper,
    bid_transformer, ad_transformer, play_transformer_pi,
    identity_action_transformer, ad_local_to_global_action_transformer,
    revelation_transformer,
    TENSOR_COMPONENT_SPEC
)
from trump_z3_bool_current_hand import analyze_hand_cards

# --- Card Definitions ---
SUIT_NAMES = ["C", "D", "H", "S"]  # Clubs, Diamonds, Hearts, Spades
NUM_SUITS = len(SUIT_NAMES)
CARDS_PER_SUIT = 13

RANK_CHAR_MAP_TO_VAL = { # Maps display char to 1-13 (Ace=1)
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13
}
# Create an inverse map for easy conversion back to characters for display
RANK_VAL_TO_CHAR_MAP = {v: k for k, v in RANK_CHAR_MAP_TO_VAL.items()}

def get_suit_from_card_name(card_name_str):
    """Extracts the suit character from a card name string."""
    if not card_name_str or len(card_name_str) < 1:
        raise ValueError("Invalid card name string for suit extraction")
    return card_name_str[0].upper()

def generate_constrained_random_hands(num_players=4, cards_per_hand=13):
    """
    Generates a random deal (4x13 cards) ensuring each player
    has at least one card of each suit.
    Returns a list of 4 lists, where each inner list contains
    13 card name strings for a player.
    """
    total_cards = num_players * cards_per_hand
    if total_cards != 52:
        raise ValueError("This generator is configured for a 52-card deck and 4 players.")

    deck_indices = list(range(52)) # Card indices 0-51
    
    valid_deal_found = False
    generated_hands_names = []
    generation_attempts = 0

    while not valid_deal_found:
        generation_attempts += 1
        if generation_attempts > 10000: # Safety break for extremely unlikely scenarios
            raise RuntimeError("Could not generate a valid constrained deal after 10000 attempts. Check constraints or logic.")

        random.shuffle(deck_indices)
        
        current_player_hands_indices = [[] for _ in range(num_players)]
        current_player_hands_names = [[] for _ in range(num_players)]
        
        # Deal cards
        for i in range(num_players):
            start_index = i * cards_per_hand
            end_index = start_index + cards_per_hand
            # Store indices sorted for consistent hand representation later, if desired,
            # but shuffling happens before this, so order of dealing is random.
            current_player_hands_indices[i] = sorted(deck_indices[start_index:end_index])
            current_player_hands_names[i] = sorted([card_index_to_name(card_idx) for card_idx in current_player_hands_indices[i]])

        # Verify this deal
        all_cards_dealt_check = []
        for hand_indices in current_player_hands_indices:
            all_cards_dealt_check.extend(hand_indices)
        
        # Assert basic deck validity (52 unique cards)
        assert len(all_cards_dealt_check) == 52, f"Dealt {len(all_cards_dealt_check)} cards, expected 52"
        assert len(set(all_cards_dealt_check)) == 52, "Dealt cards are not unique"

        # Check suit constraint for each player
        all_players_have_all_suits = True
        for p_idx in range(num_players):
            player_suits_present = set()
            for card_name in current_player_hands_names[p_idx]:
                player_suits_present.add(get_suit_from_card_name(card_name))
            if len(player_suits_present) < NUM_SUITS:
                all_players_have_all_suits = False
                break 
        
        if all_players_have_all_suits:
            valid_deal_found = True
            generated_hands_names = current_player_hands_names
            print(f"Valid deal found after {generation_attempts} attempt(s).")

    return generated_hands_names

def card_name_to_index(card_name_str):
    if len(card_name_str) < 2:
        raise ValueError(f"Invalid card name format: {card_name_str}")
    suit_char = card_name_str[0].upper()
    rank_char = card_name_str[1:].upper()
    try:
        suit = SUIT_NAMES.index(suit_char)
    except ValueError:
        raise ValueError(f"Invalid suit character: {suit_char} in {card_name_str}")
    rank_val_1_to_13 = RANK_CHAR_MAP_TO_VAL.get(rank_char)
    if rank_val_1_to_13 is None:
        raise ValueError(f"Invalid rank character: {rank_char} in {card_name_str}")
    return suit * 13 + (rank_val_1_to_13 - 1)

def card_index_to_name(card_idx):
    if card_idx < 0 or card_idx >= 52:
        return f"InvalidCard({card_idx})"
    suit_val = card_idx // 13
    rank_val_0_to_12 = card_idx % 13
    suit_char = SUIT_NAMES[suit_val]
    rank_val_1_to_13 = rank_val_0_to_12 + 1
    rank_char_display = RANK_VAL_TO_CHAR_MAP.get(rank_val_1_to_13, str(rank_val_1_to_13))
    return suit_char + rank_char_display

def get_player_hand_from_state(state, player_id):
    info_str = state.information_state_string(player_id)
    hand_line_prefix = "Hand: "
    for line in info_str.split('\n'):
        if line.startswith(hand_line_prefix):
            card_names = line[len(hand_line_prefix):].strip().split(' ')
            return sorted([card_name_to_index(name) for name in card_names if name])
    return []

def deal_specific_hands(game_obj, target_hands_as_names_list_of_lists):
    state = game_obj.new_initial_state()
    target_hands_indices = []
    for p_hand_names in target_hands_as_names_list_of_lists:
        target_hands_indices.append(sorted([card_name_to_index(name) for name in p_hand_names]))

    dealing_action_sequence = []
    # Assumes C++ deals all cards to P0, then P1, etc.
    for player_idx in range(game_obj.num_players()):
        dealing_action_sequence.extend(target_hands_indices[player_idx])
        
    for card_idx_action in dealing_action_sequence:
        if not state.is_chance_node():
            raise Exception(f"Controlled dealing phase error: Expected chance node. State:\n{state}")
        legal_chance_actions = [outcome[0] for outcome in state.chance_outcomes()]
        if card_idx_action not in legal_chance_actions:
            # This error can happen if target_hands_as_names_list_of_lists doesn't form a valid deck
            # or if the dealing sequence assumption is wrong.
            raise ValueError(
                f"Card {card_index_to_name(card_idx_action)} (idx {card_idx_action}) is not a legal chance action. "
                f"Legal: {[card_index_to_name(c) for c in legal_chance_actions]}")
        state.apply_action(card_idx_action)
    
    # Verification that hands were dealt as expected
    for p in range(game_obj.num_players()):
        current_hand_in_state = get_player_hand_from_state(state, p) # Get hand after dealing
        if current_hand_in_state != target_hands_indices[p]: # Both are sorted lists of indices
            raise Exception(
                f"Hand mismatch for Player {p} after controlled deal.\n"
                f"  Expected: {[card_index_to_name(c) for c in target_hands_indices[p]]}\n"
                f"  Got:      {[card_index_to_name(c) for c in current_hand_in_state]}")
    return state

expected_total_size = sum(size for _, size in TENSOR_COMPONENT_SPEC) # Used by print_game_info

def get_tensor_value_by_name(tensor, shape_names_and_sizes, name):
    current_offset = 0
    for item_name, item_size in shape_names_and_sizes:
        if item_name == name:
            return tensor[current_offset : current_offset + item_size]
        current_offset += item_size
    raise ValueError(f"Tensor section '{name}' not found in shape definition.")

def get_trump_phase_from_state(state, player_for_infostate_tensor=0):
    if state.is_terminal(): return "GameOver"
    
    # Check for dealing phase by history length; assumes dealing is first 52 actions
    # and history() provides all actions including chance nodes.
    if state.is_chance_node() and len(state.history()) < 52: return "Deal" 
    
    # For subsequent phases, use tensor an_d other state properties
    current_player_for_tensor = state.current_player()
    if state.is_chance_node(): # Should not happen after full deal if logic is correct
        current_player_for_tensor = player_for_infostate_tensor 
        
    info_tensor = state.information_state_tensor(current_player_for_tensor)
    round_bid_status_val = get_tensor_value_by_name(info_tensor, TENSOR_COMPONENT_SPEC, "RoundBidStatus")[0]
    current_trick_num_val = get_tensor_value_by_name(info_tensor, TENSOR_COMPONENT_SPEC, "CurrentTrickNumber")[0]

    if current_trick_num_val == 0: # Before any tricks played (Bidding or HighLowDecision)
        if round_bid_status_val == 0: # C++ round_bid_status_ is 0 during Bidding AND before H/L decision
            if not state.is_chance_node(): # Ensure it's a player's turn
                legal_actions = state.legal_actions()
                # Check if legal actions match HighLowDecision phase (Ascend=0, Descend=1)
                # These action values (0,1) are specific to Trump game logic.
                if len(legal_actions) == 2 and 0 in legal_actions and 1 in legal_actions:
                    return "HighLowDecision"
                return "Bidding" # Otherwise, it's bidding
    elif current_trick_num_val > 0 and current_trick_num_val <= 13: # kNumTricks
        return "Play"
    
    # Fallback using state string parsing if tensor logic isn't definitive
    # This is less ideal but can catch states if tensor logic is still being refined.
    state_str = str(state)
    if "Phase: Deal" in state_str: return "Deal" 
    if "Phase: Bidding" in state_str: return "Bidding"
    if "Phase: HighLowDecision" in state_str: return "HighLowDecision"
    if "Phase: Play" in state_str: return "Play"
        
    return "UnknownPhaseFromState" # Should ideally not be reached

def parse_info_from_string(state_string, key_phrase_with_colon):
    """
    Helper function to find a line starting with key_phrase (which includes a colon)
    in a multi-line string and return the value part after the colon.
    Returns "Not Found" if the key_phrase is not found.
    """
    for line in state_string.split('\n'):
        if line.strip().startswith(key_phrase_with_colon):
            try:
                return line.split(":", 1)[1].strip()
            except IndexError:
                return "Value not found after colon" # Should not happen if key_phrase includes colon
    return "Not Found"

def load_latest_params(save_dir, player, phase, net_type="pi"):
    dir_path = os.path.join(save_dir, "player" + str(player), "phase" + str(phase), net_type + "_data")
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory {dir_path} does not exist.")
    files = [f for f in os.listdir(dir_path) if f.startswith(f"{net_type}_params_iter") and f.endswith(".msgpack")]
    if not files:
        raise ValueError(f"No parameter files found in {dir_path}.")
    latest_file = max(files, key=lambda f: int(f.split("_iter")[1].split(".")[0]))
    file_path = os.path.join(dir_path, latest_file)
    with open(file_path, "rb") as f:
        state_dict = flax.serialization.from_bytes(None, f.read())
    return state_dict['params']

def interactive_trump_game(game, save_dir_nets="cfvfp_nets"):
    print("=== STARTING INTERACTIVE TRUMP GAME ===")
    state = game.new_initial_state()
    rng = random.Random()  # Used for dealing random hands

    num_players = game.num_players()
    num_cards_total = 52 
    num_tricks = num_cards_total // num_players

    # Load trained parameters
    player = 0  # Use player 0's parameters for all players due to symmetry
    params_pi = []
    for phase in range(3):
        try:
            params = load_latest_params(save_dir_nets, player, phase, "pi")
            params_pi.append(params)
        except ValueError as e:
            print(f"Error loading parameters for phase {phase}: {e}")
            return

    # Initialize policy networks
    pi_bid = PolicyNetworkWrapper(phase_net=TrumpBiddingPolicyNet())
    pi_ad = PolicyNetworkWrapper(phase_net=TrumpAD_PolicyNet())
    pi_play = PolicyNetworkWrapper(phase_net=TrumpPlayPolicyNet())

    # Define phase mapping and transformers
    phase_map = {"Bidding": 0, "HighLowDecision": 1, "Play": 2}
    info_state_transformers = [bid_transformer, ad_transformer, play_transformer_pi]
    action_transformers = [identity_action_transformer, ad_local_to_global_action_transformer, identity_action_transformer]

    # 1. Initial Dealing (played out randomly by the game's chance nodes)
    print("\n--- 1. Dealing Cards (Randomly) ---")
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        if not outcomes:
            print(f"ERROR: Chance node with no outcomes at deal move {i+1}. State:\n{state}")
            return
        action = rng.choice(outcomes)[0] 
        state.apply_action(action)
    
    print("\n--- Initial Hands Dealt (Player 0's perspective for their hand) ---")
    for p in range(num_players):
        hand_indices = get_player_hand_from_state(state, p)
        hand_names_to_print = sorted([card_index_to_name(idx) for idx in hand_indices])
        print(f"Player {p} Hand: {', '.join(hand_names_to_print)}")

    # 2. Bidding Phase - Modified to be interactive per player
    print("\n--- 2. Bidding Phase ---")
    print(f"Current state before bids:\n{str(state)}")
    
    for p in range(num_players):
        if state.current_player() != p:
            print(f"ERROR: Expected Player {p} to bid, but game is at Player {state.current_player()}'s turn.")
            print(str(state))
            return
        
        print(f"\nPlayer {p}'s turn to bid.")
        phase_str = get_trump_phase_from_state(state, p)
        if phase_str != "Bidding":
            print(f"ERROR: Expected Bidding phase, got {phase_str}")
            return
        phase_idx = phase_map[phase_str]
        full_info_state = np.array(state.information_state_tensor(p), dtype=np.float32)
        info_state_transformed = info_state_transformers[phase_idx](full_info_state)
        legal_actions_mask = np.array(state.legal_actions_mask(p), dtype=bool)
        info_state_jax = jnp.array(info_state_transformed)[None, ...]
        mask_jax = jnp.array(legal_actions_mask)[None, ...]
        net = pi_bid
        masked_logits = net.apply({'params': params_pi[phase_idx]}, info_state_jax, mask_jax)
        probs = jax.nn.softmax(masked_logits, axis=-1)[0]
        legal_actions = state.legal_actions(p)
        legal_action_names = [card_index_to_name(act) for act in legal_actions]
        probs_str = " ".join([f"{name}:{float(probs[act]):.2f}" for name, act in zip(legal_action_names, legal_actions)])
        print(f"  Legal moves for P{p}: {probs_str}")
        legal_probs = [float(probs[act]) for act in legal_actions]
        legal_probs = np.array(legal_probs) / np.sum(legal_probs)  # Renormalize
        random_choice_idx = np.random.choice(legal_actions, p=legal_probs)
        random_choice_name = card_index_to_name(random_choice_idx)
        print(f"  A random choice: {random_choice_name}")
        
        sys.stdout.flush()
        bid_card_name = input(f"Enter bid for P{p} (e.g., 'H8'): ").strip().upper()
        if bid_card_name == "QUIT":
            print("Game quit by user during bidding.")
            return state
        try:
            bid_action = card_name_to_index(bid_card_name)
        except ValueError as e:
            print(f"ERROR: Invalid card name '{bid_card_name}' for Player {p}. {e}")
            return
        if bid_action not in legal_actions:
            print(f"ERROR: Card {bid_card_name} (action {bid_action}) is not a legal bid for Player {p}.")
            return
        print(f"Player {p} bids with: {bid_card_name}")
        state.apply_action(bid_action)

    print("\n--- Bid Reveal and Round Determination ---")
    full_state_string_after_bids = str(state)
    print(full_state_string_after_bids) # Show the full state string from C++

    trump_suit_str = parse_info_from_string(full_state_string_after_bids, "Trump suit:")
    print(f"3. Trump Suit: {trump_suit_str}")

    # 4. User Input for Ascend/Descend (if applicable)
    current_phase_str = parse_info_from_string(full_state_string_after_bids, "Phase:")
    round_determination_path_str = parse_info_from_string(full_state_string_after_bids, "Round Determination:")
    
    if current_phase_str == "HighLowDecision":
        decision_player = state.current_player()
        print(f"\n--- 4. High/Low Decision Phase ---")
        print(f"Player {decision_player} (Highest Bidder) must choose to Ascend or Descend.")
        
        phase_str = get_trump_phase_from_state(state, decision_player)
        if phase_str != "HighLowDecision":
            print(f"ERROR: Expected HighLowDecision phase, got {phase_str}")
            return
        phase_idx = phase_map[phase_str]
        full_info_state = np.array(state.information_state_tensor(decision_player), dtype=np.float32)
        info_state_transformed = info_state_transformers[phase_idx](full_info_state)
        legal_actions_mask = np.array(state.legal_actions_mask(decision_player), dtype=bool)
        info_state_jax = jnp.array(info_state_transformed)[None, ...]
        mask_jax = jnp.array(legal_actions_mask)[None, ...]
        net = pi_ad
        local_logits = net.apply({'params': params_pi[phase_idx]}, info_state_jax, mask_jax)  # Mask ignored
        masked_logits = action_transformers[phase_idx](local_logits)
        probs = jax.nn.softmax(masked_logits, axis=-1)[0]
        probs = probs[0:2]
        probs /= np.sum(probs)
        action_names = ["Ascend", "Descend"]
        probs_str = " ".join([f"{name}:{float(probs[act]):.2f}" for name, act in zip(action_names, [0, 1])])
        print(f"  Choices: {probs_str}")
        random_val = np.random.random()
        random_choice_idx = 0 if random_val < probs[0] else 1
        random_choice_name = action_names[random_choice_idx]
        print(f"  A random choice: {random_choice_name}")
        
        sys.stdout.flush()
        hl_choice_input = input("Enter 'Ascend' or 'Descend': ").strip().lower()
        if hl_choice_input == "quit":
            print("Game quit by user during High/Low decision.")
            return state
        if hl_choice_input == "ascend":
            hl_action = 0
        elif hl_choice_input == "descend":
            hl_action = 1
        else:
            print("Invalid input. Please type 'Ascend' or 'Descend'.")
            return
        print(f"Player {decision_player} chose: {state.action_to_string(decision_player, hl_action)}")
        state.apply_action(hl_action)
        full_state_string_after_bids = str(state) # Update state string for round type
        print(f"State after High/Low decision:\n{full_state_string_after_bids}")
        round_determination_path_str = parse_info_from_string(full_state_string_after_bids, "Round Determination:")
    else:
        print("4. Ascend/Descend: Not applicable (bids did not sum to 13 or phase skipped).")
    
    # 5. High or Low Round
    final_round_type_str = parse_info_from_string(full_state_string_after_bids, "Resulting Round Type (for scoring):")
    print(f"5. Final Round Type (for scoring): {final_round_type_str} (based on: {round_determination_path_str})")

    # 6. Tricks Play Phase
    print("\n--- 6. Trick Play Phase ---")
    current_phase_check_play = parse_info_from_string(str(state), "Phase:")
    if current_phase_check_play != "Play": 
        print(f"ERROR: Did not enter Play phase. Current phase: {current_phase_check_play}. State:\n{str(state)}")
        return

    tricks_won_counts = [0] * num_players 
    winner_of_previous_trick = -1 

    for trick_idx in range(num_tricks): 
        if state.is_terminal():
            print("ERROR: Game ended before all tricks played.")
            break
        
        if winner_of_previous_trick != -1: 
            tricks_won_counts[winner_of_previous_trick] += 1
        
        print(f"\n-- Trick {trick_idx + 1} --")
        
        trick_counts_str = ", ".join([f"P{p_idx}:{tricks_won_counts[p_idx]}" for p_idx in range(num_players)])
        print(f"  Tricks collected so far: {trick_counts_str}")
        print(f"  Bid Cards: {parse_info_from_string(str(state), 'Revealed Bid Cards:')}")

        # Get "Broken" status from the current player's tensor (leader of this trick)
        leader_for_tensor_check = state.current_player()
        if leader_for_tensor_check >= 0 and leader_for_tensor_check < num_players:
            info_tensor = state.information_state_tensor(leader_for_tensor_check)
            break_occurred_tensor_val_list = get_tensor_value_by_name(info_tensor, TENSOR_COMPONENT_SPEC, "BreakOccurred")
            is_broken = bool(break_occurred_tensor_val_list[0] > 0.5)
            print(f"  Broken: {is_broken}")
        else:
            print(f"  Broken: (Could not determine for player {leader_for_tensor_check})")
        print(f"  Trump Suit: {trump_suit_str}")
        
        print(f"  Remaining Hands AT START of Trick {trick_idx + 1}:")
        for p_hand_check in range(num_players):
            hand_indices = get_player_hand_from_state(state, p_hand_check)
            hand_names_to_print = sorted([card_index_to_name(idx) for idx in hand_indices])
            print(f"    P{p_hand_check}: {', '.join(hand_names_to_print)}")
        
        leader_of_this_trick = state.current_player()
        print(f"  Led by: P{leader_of_this_trick}")
        
        cards_played_in_fixed_player_order = ["(-)"] * num_players 
        
        for i in range(num_players): 
            player_currently_playing = state.current_player()
            if state.is_terminal(): 
                print("ERROR: Game became terminal mid-trick.")
                break 

            print(f"\n  Player {player_currently_playing}'s turn.")
            phase_str = get_trump_phase_from_state(state, player_currently_playing)
            if phase_str != "Play":
                print(f"ERROR: Expected Play phase, got {phase_str}")
                return
            phase_idx = phase_map[phase_str]
            full_info_state = np.array(state.information_state_tensor(player_currently_playing), dtype=np.float32)
            info_state_transformed = info_state_transformers[phase_idx](full_info_state)
            legal_actions_mask = np.array(state.legal_actions_mask(player_currently_playing), dtype=bool)
            info_state_jax = jnp.array(info_state_transformed)[None, ...]
            mask_jax = jnp.array(legal_actions_mask)[None, ...]
            net = pi_play
            masked_logits = net.apply({'params': params_pi[phase_idx]}, info_state_jax, mask_jax)
            probs = jax.nn.softmax(masked_logits, axis=-1)[0]
            legal_actions = state.legal_actions(player_currently_playing)
            legal_action_names = [card_index_to_name(act) for act in legal_actions]
            # Get hand analysis for current player
            hand_analysis = analyze_hand_cards(full_info_state)
            probs_str = " ".join([f"{name}:{float(probs[act]):.2f}:{hand_analysis[act]}" for name, act in zip(legal_action_names, legal_actions)])
            print(f"  Legal moves for P{player_currently_playing}: {probs_str}")
            legal_probs = [float(probs[act]) for act in legal_actions]
            legal_probs = np.array(legal_probs) / np.sum(legal_probs)  # Renormalize
            random_choice_idx = np.random.choice(legal_actions, p=legal_probs)
            random_choice_name = card_index_to_name(random_choice_idx)
            print(f"  A random choice: {random_choice_name}")
            
            if len(legal_actions) == 1:
                # Auto-play if only one legal move
                action_to_apply = legal_actions[0]
                chosen_card_name = card_index_to_name(action_to_apply)
                print(f"  Only one legal move: {chosen_card_name} (auto-played)")
            else:
                # Multiple choices - ask user
                while True: # Loop until valid input is received for the current player
                    sys.stdout.flush()
                    chosen_card_name = input(f"  P{player_currently_playing}, enter card to play (e.g., 'SA'): ").strip().upper()
                    if chosen_card_name == "QUIT":
                        print("Game quit by user during card play.")
                        return state
                    if chosen_card_name == "UNRAVEL":
                        analyze_hand_cards(full_info_state, debug=True, max_timeout_ms=None)
                        continue  # Skip to next iteration of the while loop
                    if chosen_card_name.startswith("UNRAVEL REVEALED"):
                        all_info_states = [state.information_state_tensor(p) for p in range(num_players)]
                        # Parse intensity parameter if provided
                        parts = chosen_card_name.split()
                        if len(parts) >= 3:  # "UNRAVEL REVEALED 0.8"
                            try:
                                intensity = float(parts[2])
                                revealed_info_state = revelation_transformer(all_info_states, player_currently_playing, intensity=intensity)
                            except ValueError:
                                print(f"Invalid intensity value: {parts[2]}. Using default intensity.")
                                revealed_info_state = revelation_transformer(all_info_states, player_currently_playing)
                        else:  # Just "UNRAVEL REVEALED"
                            revealed_info_state = revelation_transformer(all_info_states, player_currently_playing)
                        analyze_hand_cards(revealed_info_state, debug=True, max_timeout_ms=None)
                        continue  # Skip to next iteration of the while loop
                    
                    try:
                        action_to_apply = card_name_to_index(chosen_card_name)
                        if action_to_apply in legal_actions:
                            action_to_apply = action_to_apply
                            chosen_card_name = chosen_card_name # Store the valid name
                            break # Valid input received, exit the while loop
                        else:
                            legal_action_names_str = ", ".join(sorted([card_index_to_name(act) for act in legal_actions]))
                            print(f"    '{chosen_card_name}' is not a legal move. Options: {legal_action_names_str}")
                            # Stays in the while loop to re-prompt the same player
                    except ValueError as e:
                        print(f"    Invalid card name format: {e}. Try again.")
                        # Stays in the while loop to re-prompt the same player
            
            cards_played_in_fixed_player_order[player_currently_playing] = chosen_card_name
            print(f"   -> P{player_currently_playing} plays: {chosen_card_name}")
            state.apply_action(action_to_apply)
        
        if state.is_terminal() and trick_idx < num_tricks - 1: # If game ended mid-play abnormally
             print(f"   Mid-trick termination detected at trick {trick_idx+1}")
             break 

        cards_played_output_str = ', '.join(cards_played_in_fixed_player_order)
        print(f"  Cards played this trick (P0-P3 order): {cards_played_output_str}")
        
        winner_of_previous_trick = state.current_player()


    # 7. Final Score
    print("\n--- 7. Game Over & Final Scores ---")
    if not state.is_terminal():
        # This case should ideally be caught by the loop break if game ends early
        print(f"ERROR: Game did not terminate after {num_tricks} tricks. State:\n{str(state)}")
        return False
    
    final_scores = state.returns()
    print("Final Scores:")
    for p, score_val in enumerate(final_scores):
        print(f"  Player {p}: {score_val}")
    return True