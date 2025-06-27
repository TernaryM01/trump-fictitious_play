import numpy as np

# Constants from the game implementation
NUM_PLAYERS = 4
NUM_CARDS = 52
NUM_SUITS = 4
NUM_CARDS_PER_SUIT = 13
NUM_CARDS_PER_PLAYER = 13
FEATURES_PER_CARD = 5
CARD_NOT_YET_PLAYED = -1
HIDDEN_TRUMP_CARD = -2
HIDDEN_TRUMP_RANK = 14
PLAYER_LEADING = 0
NO_LEADING_SUIT_YET = -1

# Helper functions
def get_card(suit, rank):
    """Convert suit (0-3) and rank (1-13) to card index (0-51)"""
    return suit * NUM_CARDS_PER_SUIT + (rank - 1)

def get_card_suit(card):
    """Get suit (0-3) from card index"""
    return card // NUM_CARDS_PER_SUIT

def card_rank(card):
    """Get rank (1-13) from card index"""
    return (card % NUM_CARDS_PER_SUIT) + 1

def card_to_string(card_index):
    """Convert card index to human-readable string (e.g., "A♠", "K♥", "2♣")"""
    # if card_index < 0 or card_index >= NUM_CARDS:
    #     return f"INVALID_CARD({card_index})"
    
    suit = get_card_suit(card_index)
    rank = card_rank(card_index)
    
    # Suit symbols
    suit_symbols = ['♣', '♦', '♥', '♠']
    
    # Rank names
    if rank == 1: rank_str = 'A'
    elif rank == 10: rank_str = 'T'
    elif rank == 11: rank_str = 'J'
    elif rank == 12: rank_str = 'Q'
    elif rank == 13: rank_str = 'K'
    else: rank_str = str(rank)
    
    return f"{rank_str}{suit_symbols[suit]}"

def suit_to_string(suit_index):
    """Convert suit index to name"""
    suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    return suit_names[suit_index] if 0 <= suit_index < NUM_SUITS else f"INVALID_SUIT({suit_index})"

# def rank_strength(rank):
#     """Convert rank to strength for comparison (2=2, 3=3, ..., 10=10, J=11, Q=12, K=13, A=14)"""
#     return NUM_CARDS_PER_SUIT + 1 if rank == 1 else rank

def analyze_hand_cards(infostate_tensor, debug=False, max_timeout_ms=None):
    """
    Analyzes cards in player's hand to determine their strategic value.
    
    Returns:
    - -1: card not in hand
    - 0: card in hand but cannot be played
    - 1: card cannot win in any scenario
    - 2: uncertain outcome (involves face-down cards)
    - 3: card beats current trick cards but outcome uncertain
    - 4: card cannot lose in any scenario
    """
    def parse_infostate_tensor(infostate_tensor):
        # 1. Player's hand (52 dims)
        start = 0
        end = NUM_CARDS
        player_hand = infostate_tensor[start:end]
        
        # 3. Trump suit (4 dims)
        start = end + NUM_PLAYERS * FEATURES_PER_CARD
        end = start + NUM_SUITS
        trump_suit_encoding = infostate_tensor[start:end]
        trump_suit = trump_suit_encoding.argmax()
        
        # 6. Enhanced graveyards (3 * 52 = 156 dims)
        start = end + 1 \
                    + NUM_CARDS_PER_PLAYER * (NUM_PLAYERS + NUM_PLAYERS * FEATURES_PER_CARD)
        end = start + (NUM_PLAYERS - 1) * NUM_CARDS
        graveyards = infostate_tensor[start:end]
        graveyards = graveyards.reshape(NUM_PLAYERS - 1, NUM_CARDS)
        
        # 8. Break occurred (1 dim)
        end = end + NUM_PLAYERS + 1
        break_occurred = infostate_tensor[end - 1] == 1.0
        
        # 9. Current trick cards (4 * 5 = 20 dims)
        start = end
        end = start + NUM_PLAYERS * FEATURES_PER_CARD
        current_trick_cards = infostate_tensor[start:end]
        current_trick_cards = current_trick_cards.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        
        # 10. Current trick leader (4 dims)
        start = end
        end = start + NUM_PLAYERS
        trick_leader_encoding = infostate_tensor[start:end]
        trick_leader = trick_leader_encoding.argmax()

        # 12. Trick number (1 dim)
        trick_num = infostate_tensor[-1]

        return player_hand, trump_suit, graveyards, break_occurred, \
               current_trick_cards, trick_leader, trick_num
    player_hand, trump_suit, graveyards, break_occurred, \
               current_trick_cards, trick_leader, trick_num = parse_infostate_tensor(infostate_tensor)
    
    current_trick_opp_cards = []  # Skip player (index 0), process opponents
    for p in range(1, NUM_PLAYERS):
        card_features = current_trick_cards[p]

        rank_encoding = int(card_features[0])
        suit_encoding = card_features[1:FEATURES_PER_CARD]
        suit_nonzero = suit_encoding.nonzero()[0]
        if len(suit_nonzero) == 0:  # The opponent hasn't played his card for the trick
            current_trick_opp_cards.append(CARD_NOT_YET_PLAYED)
            continue
        suit = suit_nonzero[0]
        
        if rank_encoding == HIDDEN_TRUMP_RANK:
            # Face-down trump card
            current_trick_opp_cards.append(HIDDEN_TRUMP_CARD)
        else:
            # Convert tensor rank system from 2-A to A-K and then to card encoding 0-51.
            game_rank = 1 if rank_encoding == NUM_CARDS_PER_SUIT else rank_encoding + 1
            current_trick_opp_cards.append(get_card(suit, game_rank))
    # print([card_to_string(card) if card is not CARD_NOT_YET_PLAYED \
    #                 else "N/A" for card in current_trick_opp_cards])
    
    # Determine leading suit
    if trick_leader != PLAYER_LEADING:
        if current_trick_opp_cards[trick_leader - 1] == HIDDEN_TRUMP_CARD:
            leading_suit = trump_suit
        else:
            leading_suit = get_card_suit(current_trick_opp_cards[trick_leader - 1])
    else:
        leading_suit = NO_LEADING_SUIT_YET  # The player is the first to play in the current trick.
    
    # # Count how many cards have been played in current trick
    # cards_played_in_trick = sum(1 for card in current_trick_cards if card != INVALID_CARD)
    
    # Initialize result array
    result = [None] * NUM_CARDS

    revealed = []
    num_not_yet_played = NUM_PLAYERS - 1
    opp_trumpers = []
    for opp, opp_card in enumerate(current_trick_opp_cards):
        if opp_card != CARD_NOT_YET_PLAYED and opp_card != HIDDEN_TRUMP_CARD:
            revealed.append(opp_card)
            num_not_yet_played -= 1
        elif opp_card == HIDDEN_TRUMP_CARD:
            num_not_yet_played -= 1
            opp_trumpers.append(opp)
    len_revealed = len(revealed)
    all_revealed = len_revealed == NUM_PLAYERS - 1
    
    z3_cards = []
    for card in range(NUM_CARDS):
        if player_hand[card] == 0:
            result[card] = -1  # Not in hand
            continue
        
        if not can_play_card(card, player_hand, leading_suit, trump_suit, break_occurred):
            result[card] = 0  # Cannot be played
            continue

        card_suit = get_card_suit(card)
        if card_suit != trump_suit:
            if (leading_suit != NO_LEADING_SUIT_YET and card_suit != leading_suit) \
                    or opp_trumpers:  # No hope of winning
               result[card] = 1
               if debug:
                   if opp_trumpers:
                       print(f'{card_to_string(card)} loses to a trump already played.')
                   else:
                       print(f'{card_to_string(card)} loses due to not following suit.')
               continue

            continue_ = False
            for opp_card in revealed:
                if not player_card_wins(card, opp_card):  # Already lost to a revealed card.
                    result[card] = 1
                    if debug:
                        print(f'{card_to_string(card)} loses to {card_to_string(opp_card)}')
                    continue_ = True
                    break
            if continue_: continue

        if all_revealed:
            result[card] = 4  # All cards already played and revealed, trick winner has been decided.
            if debug: print(f'All cards are already revealed, and {card_to_string(card)} wins.')
            continue

        z3_cards.append(card)

        if not z3_cards:  # All cards have been labeled by heuristics
            return result

    # Apply heuristic cover analysis before Z3
    z3_cards, z3_skip_label_4_flags = analyze_cards_with_covers(
        z3_cards, graveyards, leading_suit, trump_suit, break_occurred, 
        num_not_yet_played, opp_trumpers, current_trick_opp_cards, result, debug
    )
    
    # Use Z3 to analyze scenarios
    z3_labels = analyze_cards_with_z3(
        z3_cards, graveyards, current_trick_opp_cards, leading_suit,
        trump_suit, break_occurred, num_not_yet_played, len_revealed, int(trick_num),
        debug, max_timeout_ms, z3_skip_label_4_flags
    )
    
    for i, card in enumerate(z3_cards):
        result[card] = z3_labels[i]
    
    return result

def analyze_cards_with_covers(z3_cards, graveyards, leading_suit, trump_suit, break_occurred,
                              num_not_yet_played, current_trumpers,
                              current_trick_opp_cards,  # Only used to update graveyards
                              result, debug=False):
    """
    Analyze card covers and apply heuristic labeling before Z3. Modifies `result` in place.
    """
    remaining_z3_cards = []
    z3_skip_label_4_flags = []
    
    # if num_not_yet_played > 0:
    # Reshape graveyards for efficient numpy operations
    graveyards = np.array(graveyards).reshape(NUM_PLAYERS - 1, NUM_SUITS, NUM_CARDS_PER_SUIT)
    # Update graveyards to rule out cards already played in the current trick
    for opp in range(NUM_PLAYERS - 1 - num_not_yet_played, NUM_PLAYERS - 1):
        trick_card = current_trick_opp_cards[opp]
        if trick_card >= 0:  # Card played and revealed
            for opp in range(num_not_yet_played):
                graveyards[opp, get_card_suit(trick_card), card_rank(trick_card) - 1] = 2
    # else: graveyards = np.array([])
    
    # Determine which suits the player can legally play
    if leading_suit == NO_LEADING_SUIT_YET:
        # Player is trick leader - analyze each suit player can play
        playable_suits = set()
        for card in z3_cards:
            playable_suits.add(get_card_suit(card))
        
        for suit in playable_suits:
            suit_cards = [card for card in z3_cards
                          if suit * NUM_CARDS_PER_SUIT <= card < (suit + 1) * NUM_CARDS_PER_SUIT]
            remaining, skip_flags = analyze_suit_covers(
                suit_cards, graveyards, suit, trump_suit, break_occurred, 
                num_not_yet_played, result, current_trumpers, debug
            )
            remaining_z3_cards.extend(remaining)
            z3_skip_label_4_flags.extend(skip_flags)
    else:
        # Analyze based on leading suit
        remaining_z3_cards, z3_skip_label_4_flags = analyze_suit_covers(
            z3_cards, graveyards, leading_suit, trump_suit, break_occurred,
            num_not_yet_played, result, current_trumpers, debug
        )
    
    return remaining_z3_cards, z3_skip_label_4_flags

def analyze_suit_covers(cards, graveyards, leading_suit, trump_suit, break_occurred,
                        num_not_yet_played, result, current_trumpers=[], debug=False):
    """Analyze covers for a specific leading suit scenario."""
    remaining_cards = []
    skip_label_4_flags = []
    
    # Separate trump and leading suit cards (others are already eliminated by caller)
    trump_suit_cards = [card for card in cards if get_card_suit(card) == trump_suit]
    leading_suit_cards = [card for card in cards if get_card_suit(card) == leading_suit]
    
    potential_trumpers, all_trumpers_might_exist = find_potential_trumpers(
        graveyards, leading_suit, trump_suit, break_occurred, num_not_yet_played) \
                         if leading_suit != trump_suit else ([], False)
    actual_trumpers, all_trumpers_exist, actual_voids = find_actual_trumpers_and_voids(
        graveyards, leading_suit, trump_suit, break_occurred, num_not_yet_played) \
                         if leading_suit != trump_suit else ([], False, [])
    
    # Handle trump cards when leading suit is not trump
    if leading_suit != trump_suit and trump_suit_cards:
        remaining_card, skip_flag = analyze_trump_suit_covers(
            trump_suit_cards, graveyards, trump_suit, num_not_yet_played,
            current_trumpers, potential_trumpers, actual_trumpers,
            result, debug
        )
        remaining_cards.extend(remaining_card)
        skip_label_4_flags.extend(skip_flag)
    
    # Handle leading suit cards where player follows suit
    if leading_suit_cards:

        if all_trumpers_exist:
            for card in leading_suit_cards:
                result[card] = 1
                if debug: print(f"{card_to_string(card)} loses to an all-trumper.")
        
        else:
            remaining_card, skip_flag = analyze_leading_suit_covers(
                leading_suit_cards, graveyards, leading_suit, num_not_yet_played,
                current_trumpers, potential_trumpers, len(actual_trumpers) > 0, all_trumpers_might_exist, actual_voids,
                result, debug
            )
            remaining_cards.extend(remaining_card)
            skip_label_4_flags.extend(skip_flag)
    
    return remaining_cards, skip_label_4_flags

def analyze_leading_suit_covers(leading_suit_cards, graveyards, leading_suit, num_not_yet_played,
                                current_trumpers, potential_trumpers, actual_trumpers_exist, all_trumpers_might_exist, actual_voids,
                                result, debug=False):
    """Analyze covers for when player plays a leading suit card."""
    remaining_cards = []
    skip_label_4_flags = []
    
    # Find potential voids and trumpers
    potential_voids = find_potential_voids(graveyards, leading_suit, num_not_yet_played)    
    covers = build_leading_covers(graveyards, leading_suit, num_not_yet_played,
                                  potential_trumpers, current_trumpers)
    
    # Flatten all covers for beats_all_covers check
    flattened_covers = []
    for opp_cover in covers:
        if opp_cover is not None:
            flattened_covers.extend(opp_cover)
    
    # Process each leading suit card
    for card in leading_suit_cards:
        card_rank_ = card_rank(card)
        
        # Check if card beats all covers
        if beats_all_covers(card_rank_, flattened_covers):
            result[card] = 4
            if debug: print(f"{card_to_string(card)} beats all covers.")
            continue
        
        # Check if card loses in a scenario where everybody plays the lowest rank
        if loses_to_enough_covers(card_rank_, covers, potential_voids, num_not_yet_played,
                                  current_trumpers):  # only non-empty if leading_suit == trump_suit
            result[card] = 1
            if debug: print(f"{card_to_string(card)} loses to enough covers.")
            continue

        if skip_flag := actual_trumpers_exist:
            if debug: print(f"Trumpers can defeat {card_to_string(card)}.")
        else:
            # Check if card can be defeated by opponent not yet played.
            reduced_covers = get_reduced_leading_covers(covers, graveyards, leading_suit, num_not_yet_played, debug)
            skip_flag = not beats_all_covers(card_rank_, reduced_covers)
            if debug and skip_flag:
                print(f"{card_to_string(card)} can be beaten by an opponent who have not yet played.")

        if skip_flag and wins_to_enough_covers(card_rank_, covers, actual_voids, num_not_yet_played) \
                and not all_trumpers_might_exist:
            result[card] = 3
            if debug: print(f"{card_to_string(card)} wins to enough covers.")
            continue
        
        remaining_cards.append(card)
        skip_label_4_flags.append(skip_flag)
    
    return remaining_cards, skip_label_4_flags

def analyze_trump_suit_covers(trump_suit_cards, graveyards, trump_suit, num_not_yet_played,
                              current_trumpers, potential_trumpers, actual_trumpers,
                              result, debug=False):
    """Analyze covers for trump cards when leading suit is not trump."""
    remaining_cards = []
    skip_label_4_flags = []
    
    # Build trump covers (no special rank 14 for trumps)
    thin_covers = build_trump_covers(graveyards, trump_suit, current_trumpers + potential_trumpers)
    thick_covers = build_trump_covers(graveyards, trump_suit, current_trumpers + actual_trumpers)
    
    # Process each trump card
    for card in trump_suit_cards:
        card_rank_ = card_rank(card)
        
        # Check if card beats all trump covers
        if not thin_covers or beats_all_covers(card_rank_, thin_covers):
            result[card] = 4
            if debug: print(f"{card_to_string(card)} beats all covers.")
            continue
        
        # Check if card can be beaten by an opponent in current trick.
        reduced_covers = get_reduced_trump_covers(thick_covers, graveyards, trump_suit, current_trumpers, num_not_yet_played)
        skip_flag = not beats_all_covers(card_rank_, reduced_covers)
        if debug and skip_flag:
            print(f"{card_to_string(card)} can be beaten by an opponent in current trick.")
        
        remaining_cards.append(card)
        skip_label_4_flags.append(skip_flag)
    
    return remaining_cards, skip_label_4_flags

def find_potential_voids(graveyards, leading_suit, num_not_yet_played):
    """Find opponents who might be void in leading suit."""
    # Check if any card of leading suit has been played (value == 1)
    if num_not_yet_played == 0: return []

    played_leading_suit = np.any(graveyards[:num_not_yet_played, leading_suit, :] == 1, axis=1)
    return np.where(played_leading_suit)[0].tolist()

def find_potential_trumpers(graveyards, leading_suit, trump_suit, break_occurred, num_not_yet_played):
    """Find opponents who might play trump cards."""
    # if leading_suit == trump_suit: return []
    if num_not_yet_played == 0: return [], False

    # Has played at least one card of each non-trump suit
    non_trump_suits = [s for s in range(NUM_SUITS) if s != trump_suit]
    played_non_trump_suits = np.all([
        np.any(graveyards[:num_not_yet_played, suit, :] == 1, axis=1) 
        for suit in non_trump_suits
    ], axis=0)
    no_non_trump_known = np.all([
        np.all(graveyards[:num_not_yet_played, suit, :] >= 0, axis=1) 
        for suit in non_trump_suits
    ], axis=0)
    potential_all_trumpers = np.where(played_non_trump_suits & no_non_trump_known)[0].tolist()

    if not break_occurred:
        return potential_all_trumpers, len(potential_all_trumpers) > 0
    else:
        # Has played at least one card of leading suit
        played_leading_suit = np.any(graveyards[:num_not_yet_played, leading_suit, :] == 1, axis=1)
        return np.where(played_leading_suit)[0].tolist(), len(potential_all_trumpers) > 0

def find_actual_trumpers_and_voids(graveyards, leading_suit, trump_suit, break_occurred, num_not_yet_played):
    """Find opponents who can definitely play trump cards."""
    # if leading_suit == trump_suit: return []
    if num_not_yet_played == 0: return [], False, []
    
    # All cards in hand are trump suit cards.
    non_trump_suits = [s for s in range(NUM_SUITS) if s != trump_suit]
    all_trumps = np.all([
        np.all(graveyards[:num_not_yet_played, suit, :] > 0, axis=1) 
        for suit in non_trump_suits
    ], axis=0)
    all_trumpers = np.where(all_trumps)[0].tolist()
    # No leading suit cards in hand
    no_leading_suit = np.all(graveyards[:num_not_yet_played, leading_suit, :] > 0, axis=1)
    # Any trumps in hand
    have_trumps = np.any(graveyards[:num_not_yet_played, trump_suit, :] > 0, axis=1)
    
    if not break_occurred: return all_trumpers, len(all_trumpers) > 0, no_leading_suit
    else:
        return np.where(no_leading_suit & have_trumps)[0].tolist(), len(all_trumpers) > 0, no_leading_suit

def build_leading_covers(graveyards, leading_suit, num_not_yet_played, potential_trumpers, current_trumpers):
    # Get mask for cards that might be available (value <= 0)
    available_mask = graveyards[:, leading_suit, :] <= 0
    
    # Convert to ranks (1-13) and build covers for each opponent
    covers = [None] * (NUM_PLAYERS - 1)
    for opp in list(range(num_not_yet_played)) + current_trumpers:
        # Get available ranks for this opponent
        available_ranks = np.where(available_mask[opp])[0] + 1  # Convert to 1-13
        opp_covers = available_ranks.tolist()
        
        # Add special rank 14 if this opponent is a potential trumper
        if opp in potential_trumpers:
            opp_covers.append(14)
        
        covers[opp] = opp_covers
    
    return covers

def build_trump_covers(graveyards, trump_suit, potential_trumpers):
    if not potential_trumpers: return []
    
    # Get available trump cards for potential trumpers
    potential_trumpers = np.array(potential_trumpers)
    available_mask = graveyards[potential_trumpers, trump_suit, :] <= 0
    
    # Get all available trump ranks across all potential trumpers
    available_ranks = np.where(np.any(available_mask, axis=0))[0] + 1  # Convert to 1-13
    return available_ranks.tolist()

def loses_to_enough_covers(card_rank, covers, potential_voids, num_not_yet_played, current_trumpers):
    """Check if card loses to the n-th lowest rank in combined covers from non-void opponents."""
    non_potential_voids = [opp for opp in range(num_not_yet_played) if opp not in potential_voids]
    if not non_potential_voids: return False
    n = len(non_potential_voids)
    
    # Convert covers to boolean masks and combine with np.any
    all_masks = []
    for opp in non_potential_voids + current_trumpers:
        if covers[opp]:
            ranks = np.array(covers[opp])
            mask = np.zeros(13, dtype=bool)
            # Positions 0-12 for ranks 2-A
            positions = np.where(ranks == 1, 12, ranks - 2)
            mask[positions] = True
            all_masks.append(mask)
    # if not all_masks: return False
    
    # Combine all masks using np.any
    combined_mask = np.any(all_masks, axis=0)
    # Get available ranks
    available_ranks = np.where(combined_mask)[0] + 2  # Ranks 2-14
    
    if len(available_ranks) < n:
        return False
    
    # Handle player card rank (Ace as 14)
    effective_card_rank = 14 if card_rank == 1 else card_rank
    nth_lowest = available_ranks[n-1]
    
    return effective_card_rank < nth_lowest

def wins_to_enough_covers(card_rank, covers, actual_voids, num_not_yet_played):
    """Try to construct a scenario where player wins."""
    non_actual_voids = [opp for opp in range(num_not_yet_played) if opp not in actual_voids]
    if not non_actual_voids: return True
    n = len(non_actual_voids)
    
    # Convert covers to boolean masks and combine with np.any
    all_masks = []
    for opp in non_actual_voids:
        if covers[opp]:
            ranks = np.array(covers[opp])
            mask = np.zeros(13, dtype=bool)
            # Positions 0-12 for ranks 2-A
            positions = np.where(ranks == 1, 12, ranks - 2)
            mask[positions] = True
            all_masks.append(mask)
    # if not all_masks: return False
    
    # Combine all masks using np.any
    combined_mask = np.any(all_masks, axis=0)
    # Get available ranks
    available_ranks = np.where(combined_mask)[0] + 2  # Ranks 2-14
    
    if len(available_ranks) < n:
        return True
    
    # Handle player card rank (Ace as 14)
    effective_card_rank = 14 if card_rank == 1 else card_rank
    nth_lowest = available_ranks[n-1]
    
    return effective_card_rank > nth_lowest

def beats_all_covers(card_rank_, all_covers):
    """Check if card rank beats all possible cards played by opponents yet to play."""
    if not all_covers:
        return True
    
    # Handle Ace as highest
    effective_rank = 14 if card_rank_ == 1 else card_rank_
    
    # Convert all covers to effective values
    covers_array = np.array(all_covers)
    effective_covers = np.where(covers_array == 1, 14, covers_array)
    
    return effective_rank > np.max(effective_covers)

def get_reduced_leading_covers(covers, graveyards, leading_suit, num_not_yet_played, debug):
    reduced_covers = []
    
    # Get mask for cards that might belong to already-played opponents
    if num_not_yet_played < NUM_PLAYERS - 1:
        played_masks = graveyards[num_not_yet_played:, leading_suit, :] <= 0
        nobody_masks = graveyards[:, leading_suit, :] > 0
        played_masks_flattened = np.any(played_masks, axis=0)
        nobody_masks_flattened = np.all(nobody_masks, axis=0)
        masks_flattened = np.any(np.stack((played_masks_flattened, nobody_masks_flattened)), axis=0)
    else:
        masks_flattened = np.zeros(NUM_CARDS_PER_SUIT, dtype=bool)
    
    for opp in range(num_not_yet_played):
        opp_cover = np.array(covers[opp])
        
        # Remove special rank 14
        non_trump_opp_cover = opp_cover[opp_cover != 14]
        
        if len(non_trump_opp_cover) > 0:
            # Convert back to 0-12 indices for checking
            rank_indices = non_trump_opp_cover - 1
            # Remove ranks that might belong to played opponents
            valid_mask = ~masks_flattened[rank_indices]
            valid_ranks = non_trump_opp_cover[valid_mask]
            reduced_covers.extend(valid_ranks.tolist())
    
    return reduced_covers

def get_reduced_trump_covers(trump_covers, graveyards, trump_suit, current_trumpers, num_not_yet_played):
    if not trump_covers or num_not_yet_played == NUM_PLAYERS - 1:
        return trump_covers
    
    # Get mask for cards that belong to opponents not playing trumps
    excluded_opps = set(range(num_not_yet_played, NUM_PLAYERS - 1)).difference(current_trumpers)
    included_opps = excluded_opps.difference(range(NUM_PLAYERS - 1))
    played_masks = graveyards[list(excluded_opps), trump_suit, :] <= 0
    ruled_out_masks = graveyards[list(included_opps), trump_suit, :] > 0
    masks_flattened = np.any(np.concatenate((played_masks, ruled_out_masks)), axis=0)
    
    # Filter out ranks available to played opponents
    trump_covers = np.array(trump_covers)
    rank_indices = trump_covers - 1  # Convert to 0-12 indices
    valid_mask = ~masks_flattened[rank_indices]
    
    return trump_covers[valid_mask].tolist()

def analyze_cards_with_z3(player_cards, graveyards, current_trick_opp_cards, leading_suit, trump_suit,
                         break_occurred, num_not_yet_played, len_revealed, trick_num,
                         debug, max_timeout_ms, skip_label_4_flags=None):
    """Use Z3 solver to analyze card scenarios"""
    import z3
    z3.set_param('parallel.enable', False)
    # z3.set_param('parallel.threads.max', 2)

    def add_current_hand_and_played_cards_domain_constraints(solver, opponent_hands, opponent_plays, trick_num):
        """
        Adds constraints for opponents' initial hands:
        1. Each opponent has exactly 13 cards.
        2. Each opponent has at least one card of each suit.
        3. Each card is not owned by more than one opponent.
        and domain constraints for the current trick cards.
        """
        for opp in range(NUM_PLAYERS - 1):
            # Two equivalent versions are both used
            num_cards_hand_constraint_sumif = z3.Sum([z3.If(opponent_hands[opp][card], 1, 0)
                                                      for card in range(NUM_CARDS)]) \
                                                     == NUM_CARDS_PER_PLAYER + 1 - trick_num
            solver.add(num_cards_hand_constraint_sumif)
            num_cards_hand_constraint_pb = z3.PbEq([(opponent_hands[opp][card], 1)
                                                    for card in range(NUM_CARDS)],
                                                   NUM_CARDS_PER_PLAYER + 1 - trick_num)
            solver.add(num_cards_hand_constraint_pb)
            
            for suit in range(NUM_SUITS):
                # If opponent has not played any card of the suit, he has one.
                if not [card for card in range(suit * NUM_CARDS_PER_SUIT, (suit + 1) * NUM_CARDS_PER_SUIT)
                        if graveyards[opp][card] == 1]:
                    suit_in_hand_constraint = z3.Or([opponent_hands[opp][card]
                                                     for card in range(suit * NUM_CARDS_PER_SUIT, (suit + 1) * NUM_CARDS_PER_SUIT)])
                    solver.add(suit_in_hand_constraint)

        for card in range(NUM_CARDS):
            # Two equivalent versions are both used
            no_two_owners_constraint_sumif = z3.Sum([z3.If(opponent_hands[opp][card], 1, 0)
                                                     for opp in range(NUM_PLAYERS - 1)]) <= 1
            solver.add(no_two_owners_constraint_sumif)
            no_two_owners_constraint_pb = z3.PbLe([(opponent_hands[opp][card], 1)
                                                   for opp in range(NUM_PLAYERS - 1)], 1)
            solver.add(no_two_owners_constraint_pb)

        # Add domain constraints for opponent plays
        for opp in range(NUM_PLAYERS - 1):
            # domain_constraint = z3.And([opponent_plays[opp] >= 0, opponent_plays[opp] < NUM_CARDS])
            domain_constraint = z3.Or([opponent_plays[opp] == card for card in range(NUM_CARDS)])
            solver.add(domain_constraint)

    def add_graveyard_and_current_trick_constraints(solver, opponent_hands, graveyards,
                                                    opponent_plays, current_trick_opp_cards, trump_suit):
        """Add constraints based on graveyard information and cards already played in current trick."""
        for opp in range(NUM_PLAYERS - 1):
            card_played = current_trick_opp_cards[opp]

            if card_played != CARD_NOT_YET_PLAYED:  # Card has been played
                if card_played == HIDDEN_TRUMP_CARD:  # Face-down trump card
                    # Must be a trump suit card
                    # trump_cards = [trump_suit * NUM_CARDS_PER_SUIT + rank for rank in range(NUM_CARDS_PER_SUIT)]
                    # solver.add(z3.Or([opponent_plays[opp] == card for card in trump_cards]))
                    solver.add(z3.And([opponent_plays[opp] >= trump_suit * NUM_CARDS_PER_SUIT,
                                       opponent_plays[opp] < (trump_suit + 1) * NUM_CARDS_PER_SUIT]))
                else:  # Specific card played
                    solver.add(opponent_plays[opp] == card_played)

            # Add constraints based on graveyard information
            for card in range(NUM_CARDS):
                graveyard_value = graveyards[opp][card]
                if graveyard_value == -1:  # Opponent has this card
                    solver.add(opponent_hands[opp][card])
                elif graveyard_value > 0:  # Opponent doesn't have this card
                    solver.add(z3.Not(opponent_hands[opp][card]))
                # For value 0, no constraint on hand or current trick card played.

                # Played card must be from opponent's hand
                hand_constraint = z3.Implies(opponent_plays[opp] == card, opponent_hands[opp][card])
                solver.add(hand_constraint)

    def add_legality_constraints(solver, opponent_hands, opponent_plays,
                                 leading_suit, trump_suit, break_occurred):
        """Add constraints for legal card play
        See also non-Z3 counterpart for player: `can_play_card(...)`.
        """
        for opp in range(NUM_PLAYERS - 1):
            # Must follow suit if possible
            leading_suit_cards = range(leading_suit * NUM_CARDS_PER_SUIT, (leading_suit + 1) * NUM_CARDS_PER_SUIT)

            has_leading_suit = z3.Or([opponent_hands[opp][card] for card in leading_suit_cards])
            
            opp_card = opponent_plays[opp]
            # follows_suit = z3.Or([opp_card == card for card in leading_suit_cards])
            follows_suit = z3.And([opp_card >= leading_suit * NUM_CARDS_PER_SUIT,
                                   opp_card < (leading_suit + 1) * NUM_CARDS_PER_SUIT])

            solver.add(z3.Implies(has_leading_suit, follows_suit))

            # "Break" rule
            if not break_occurred and leading_suit != trump_suit:
                trump_cards = range(trump_suit * NUM_CARDS_PER_SUIT, (trump_suit + 1) * NUM_CARDS_PER_SUIT)
                has_nontrump = z3.Or([opponent_hands[opp][card]
                                      for card in range(NUM_CARDS) if card not in trump_cards])
                
                # must_play_nontrump = z3.And([opp_card != card for card in trump_cards])
                must_play_nontrump = z3.Or([opp_card < trump_suit * NUM_CARDS_PER_SUIT,
                                            opp_card >= (trump_suit + 1) * NUM_CARDS_PER_SUIT])
                
                solver.add(z3.Implies(has_nontrump, must_play_nontrump))

    def check_wins_all_scenarios(solver, player_card, opponent_plays,
                                 trump_suit, leading_suit, debug):
        """Check if player card wins in all possible scenarios."""
        solver.push()  # Add a temporary extra constraint that will be removed after the function returns.
        try:
            # Add constraint that player doesn't win
            player_loses = create_player_loses_constraint(player_card, opponent_plays,
                                                          trump_suit, leading_suit)
            solver.add(player_loses)
            
            if solver.check() == z3.sat:
                if debug:
                    model = solver.model()
                    print(f"A scenario where player loses with {card_to_string(player_card)}:")
                    extract_scenario_from_model(model)
                return False
            return True
        finally:
            solver.pop()

    def check_loses_all_scenarios(solver, player_card, opponent_plays,
                                  trump_suit, leading_suit, debug):
        """Check if player card loses in all possible scenarios."""
        solver.push()  # Add a temporary extra constraint that will be removed after the function returns.
        try:
            # Add constraint that player wins
            player_wins = create_player_wins_constraint(player_card, opponent_plays,
                                                        trump_suit, leading_suit)
            solver.add(player_wins)
            
            if solver.check() == z3.sat:
                if debug:
                    model = solver.model()
                    print(f"A scenario where player wins with {card_to_string(player_card)}:")
                    extract_scenario_from_model(model)
                return False
            return True
        finally:
            solver.pop()

    def check_loses_to_not_yet_played(solver, player_card, opponent_plays,
                                    trump_suit, leading_suit, num_not_yet_played, debug):
        """Check if player card loses to an opponent who plays after,
        assuming "wins_all_scenarios" & "loses_all_scenarios" have been ruled out.
        """
        solver.push()  # Add a temporary extra constraint that will be removed after the function returns.
        try:
            # Add constraint that player loses to an opponent who played before him
            player_loses = create_player_loses_to_already_played(player_card, opponent_plays,
                                                                trump_suit, leading_suit,
                                                                NUM_PLAYERS - 1 - num_not_yet_played)
            solver.add(player_loses)
            
            try:
                if solver.check() == z3.sat:
                    if debug:
                        model = solver.model()
                        print(f"A scenario where player loses with {card_to_string(player_card)} "
                              f"to an opponent who already played:")
                        extract_scenario_from_model(model)
                    return False
                return True
            except Exception as e:
                print(e)
                return True
        finally:
            solver.pop()

    def create_player_wins_constraint(player_card, opponent_plays,
                                    trump_suit, leading_suit):
        """Create Z3 constraint for player winning the trick"""
        constraints = []
        for opp in range(NUM_PLAYERS - 1):
            # Opponent's card must be weaker
            opp_card_weaker = create_player_card_wins_constraint(
                player_card, opponent_plays[opp], trump_suit, leading_suit
            )
            constraints.append(opp_card_weaker)
        
        return z3.And(constraints)  # Player wins if his card is stronger than all opponents'

    def create_player_loses_constraint(player_card, opponent_plays,
                                    trump_suit, leading_suit):
        """Create Z3 constraint for player losing the trick"""
        constraints = []
        for opp in range(NUM_PLAYERS - 1):
            opp_card_weaker = create_player_card_wins_constraint(
                player_card, opponent_plays[opp], trump_suit, leading_suit
            )
            constraints.append(z3.Not(opp_card_weaker))
        return z3.Or(constraints)  # Player loses if any opponent card is stronger

    def create_player_loses_to_already_played(player_card, opponent_plays,
                                            trump_suit, leading_suit, num_already_played):
        """Create Z3 constraint for player losing the trick to an opponent who played before"""
        constraints = []
        for opp in range(NUM_PLAYERS - 1 - num_already_played, NUM_PLAYERS - 1):
            opp_card_weaker = create_player_card_wins_constraint(
                player_card, opponent_plays[opp], trump_suit, leading_suit
            )
            constraints.append(z3.Not(opp_card_weaker))
        return z3.Or(constraints)  # Player loses if any opponent card is stronger

    def create_player_card_wins_constraint(player_card, opp_card, trump_suit, leading_suit):
        """Create constraint comparing player card with opponent card.
        See also non-Z3 counterpart: `player_card_wins(...)`.
        """
        
        # Adjust card indices to handle Aces as highest: if card % 13 == 0, add 13
        # def ace_adjusted_strength(card):
        #     return NUM_CARDS_PER_SUIT if (card % NUM_CARDS_PER_SUIT) == 0 else (card % NUM_CARDS_PER_SUIT)
        # def ace_adjusted_strength_z3(card):
        #     return z3.If(card % NUM_CARDS_PER_SUIT == 0, NUM_CARDS_PER_SUIT, card % NUM_CARDS_PER_SUIT)
        # opp_rank_weaker = ace_adjusted_strength_z3(opp_card) < ace_adjusted_strength(player_card)
        
        if get_card_suit(player_card) == trump_suit:
            # return z3.Implies(opp_card_is_trump, opp_rank_weaker)
            # trump_cards = range(trump_suit * NUM_CARDS_PER_SUIT, (trump_suit + 1) * NUM_CARDS_PER_SUIT)
            # opp_stronger = z3.Or([opp_card == card for card in trump_cards
            #                       if ace_adjusted_strength(card) > ace_adjusted_strength(player_card)])
            if player_card == trump_suit * NUM_CARDS_PER_SUIT:  # Ace
                return z3.BoolVal(True)
            opp_stronger = z3.Or([opp_card == trump_suit * NUM_CARDS_PER_SUIT,  # Ace
                                  # Trump suit card of higher rank
                                  z3.And(opp_card < (trump_suit + 1) * NUM_CARDS_PER_SUIT,
                                         opp_card > player_card)])
            return z3.Not(opp_stronger)
        # elif player_card_suit != leading_suit:     # Already ruled out by Python caller
        #     return False
        else:
            # leading_suit_cards = range(leading_suit * NUM_CARDS_PER_SUIT, (leading_suit + 1) * NUM_CARDS_PER_SUIT)
            # opp_card_follows_suit = z3.Or([opp_card == card for card in leading_suit_cards])
            # opp_card_follows_suit = z3.And([opp_card >= leading_suit * NUM_CARDS_PER_SUIT,
            #                                 opp_card < (leading_suit + 1) * NUM_CARDS_PER_SUIT])
            # return z3.And(z3.Not(opp_card_is_trump),
            #               z3.Implies(opp_card_follows_suit, opp_rank_weaker))

            # opp_trumper = z3.Or([opp_card == card for card in trump_cards])
            opp_trumper = z3.And([opp_card >= trump_suit * NUM_CARDS_PER_SUIT,
                                  opp_card < (trump_suit + 1) * NUM_CARDS_PER_SUIT])
            # opp_stronger_leading = z3.Or([opp_card == card for card in leading_suit_cards
            #                               if ace_adjusted_strength(card) > ace_adjusted_strength(player_card)])
            if player_card == leading_suit * NUM_CARDS_PER_SUIT:  # Ace
                return z3.Not(opp_trumper)
            opp_stronger_leading = z3.Or([opp_card == leading_suit * NUM_CARDS_PER_SUIT,  # Ace
                                          # Leading suit card of higher rank
                                          z3.And(opp_card < (leading_suit + 1) * NUM_CARDS_PER_SUIT,
                                                 opp_card > player_card)])
            return z3.And([z3.Not(opp_trumper), z3.Not(opp_stronger_leading)])

    def extract_scenario_from_model(model):
        """Extract opponent hands and plays from Z3 model"""
        # Extract opponent hands
        print("Opponent Current Hands:")
        for opp in range(NUM_PLAYERS - 1):
            print(f"  P{opp + 1}: ", end="")
            
            # Collect all cards for this opponent
            cards = []
            for card in range(NUM_CARDS):
                var_name = f"opp_{opp}_has_card_{card}"
                var = model[z3.Bool(var_name)]
                if z3.is_true(var):
                    cards.append(card)
            
            # Sort cards by suit first, then by rank within suit
            cards.sort(key=lambda c: (get_card_suit(c), card_rank(c) if card_rank(c) != 1 else 14))
            
            # Convert to string and print
            cards_str = ' '.join(card_to_string(card) for card in cards)
            print(cards_str)
        
        # Extract opponent plays for current trick
        print("Current Trick:    ", end="")
        trick_plays = []
        for opp in range(NUM_PLAYERS - 1):
            var_name = f"opp_{opp}_plays"
            if var_name in [str(d) for d in model.decls()]:
                var = model[z3.Int(var_name)]
                if var is not None:
                    card_played = var.as_long()
                    trick_plays.append(f"P{opp + 1}:{card_to_string(card_played)}")
                else:
                    trick_plays.append(f"P{opp + 1}:UNKNOWN")
            else:
                trick_plays.append(f"P{opp + 1}:NOT_FOUND")
        
        print(" ".join(trick_plays))

    solver = z3.SolverFor("QF_LIA")
    solver.set("smt.arith.solver", 6)
    if max_timeout_ms: solver.set("timeout", max_timeout_ms)
    
    # Create variables for each opponent's hand
    opponent_hands = [[z3.Bool(f"opp_{opp}_has_card_{card}") for card in range(NUM_CARDS)]
                      for opp in range(NUM_PLAYERS - 1)]
    # Create variables for cards played by opponents in current trick
    opponent_plays = [z3.Int(f"opp_{opp}_plays") for opp in range(NUM_PLAYERS - 1)]
    
    add_current_hand_and_played_cards_domain_constraints(solver, opponent_hands, opponent_plays, trick_num)
    add_graveyard_and_current_trick_constraints(solver, opponent_hands, graveyards,
                                                opponent_plays, current_trick_opp_cards, trump_suit)
    if leading_suit != NO_LEADING_SUIT_YET:
        add_legality_constraints(solver, opponent_hands, opponent_plays,
                                 leading_suit, trump_suit, break_occurred)
    
    labels = []
    for i, player_card in enumerate(player_cards):
        # If the player is the trick leader, the leading suit is determined by the card he is about to play.
        if leading_suit == NO_LEADING_SUIT_YET:
            solver.push()
            leading_suit_ = get_card_suit(player_card)
            add_legality_constraints(solver, opponent_hands, opponent_plays,
                                     leading_suit_, trump_suit, break_occurred)
        else:
            leading_suit_ = leading_suit

        if debug:
            if solver.check() == z3.unsat:
                print("ERROR: No valid sequence of play found. Check constraints!")
            else:
                model = solver.model()
                print(f"A valid play where player plays {card_to_string(player_card)}:")
                extract_scenario_from_model(model)
        
        # Determine label
        skip_label_4 = skip_label_4_flags[i] if skip_label_4_flags else False
        
        if not skip_label_4 and \
            check_wins_all_scenarios(solver, player_card, opponent_plays,
                                     trump_suit, leading_suit_, debug):
            labels.append(4)
        elif check_loses_all_scenarios(solver, player_card, opponent_plays,
                                       trump_suit, leading_suit_, debug):
            labels.append(1)
        elif len_revealed + num_not_yet_played == NUM_PLAYERS - 1:
            labels.append(3)
        elif num_not_yet_played > 0:
            if check_loses_to_not_yet_played(solver, player_card, opponent_plays,
                                            trump_suit, leading_suit_, num_not_yet_played, debug):
                labels.append(3)
            else: labels.append(2)
        else: labels.append(2)

        if leading_suit == NO_LEADING_SUIT_YET:
            solver.pop()
    return labels
    
#Non-Z3 comparison for quick short-circuiting
def player_card_wins(player_card, opp_card):#trump_suit, leading_suit):
    """Determine whether player has hope of winning before launching Z3
       See also Z3 counterpart: `create_player_card_wins_constraint(...)`.
    """    
    # Adjust card indices to handle Aces as highest: if card % 13 == 0, add 13
    def ace_adjusted_strength(card):
        return card + NUM_CARDS_PER_SUIT if card % NUM_CARDS_PER_SUIT == 0 else card
    
    # if player_card_suit == trump_suit: return True    # Ruled out by caller.
    # if get_card_suit(player_card) != leading_suit:# or get_card_suit(opp_card) == trump_suit:  # Ruled out by caller.
    #     return False  # Player is guaranteed to lose, if not to this opponent.    # Ruled out by caller.
    if get_card_suit(opp_card) != get_card_suit(player_card):  # i.e. leading suit, as guaranteed by caller.
        return True
    else:
        return ace_adjusted_strength(opp_card) < ace_adjusted_strength(player_card)

def can_play_card(card_to_play, player_hand, leading_suit, trump_suit, break_occurred):
    """Check if a card can be legally played.
       See also Z3 counterpart for opponents: `add_legality_constraints(...)`.
    """
    card_suit = get_card_suit(card_to_play)
    
    # If no leading suit yet, any card can be played (with trump restrictions)
    if leading_suit == NO_LEADING_SUIT_YET:
        if card_suit == trump_suit and not break_occurred:
            # Can only play trump if no other choice
            has_nontrump = any(player_hand[card] == 1 and get_card_suit(card) != trump_suit
                               for card in range(NUM_CARDS))
            return not has_nontrump
        return True
    
    # Must follow suit if possible
    has_leading_suit = any(player_hand[card] == 1 and get_card_suit(card) == leading_suit
                           for card in range(NUM_CARDS))
    if has_leading_suit:
        return card_suit == leading_suit
    
    # Can't follow suit, can play any card (with trump restrictions)
    if card_suit == trump_suit and not break_occurred:
        # Can only play trump if no other choice
        has_nontrump = any(player_hand[card] == 1 and get_card_suit(card) != trump_suit 
                           for card in range(NUM_CARDS))
        return not has_nontrump
    
    return True