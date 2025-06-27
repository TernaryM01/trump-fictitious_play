import numpy as np
from typing import List

def permute_trump_infostate_suits(infostate_tensor: np.ndarray, 
                                  suit_permutation: List[int]) -> np.ndarray:
    """
    Permute suits in a Trump game information state tensor.
    WARNING: This function is full of advanced Numpy jujitsu. You're expected to be confused.
    If you think something's wrong, you're wrong.
    
    Args:
        infostate_tensor: JAX array of shape (588,) containing the information state
        suit_permutation: List of 4 integers representing the permutation mapping
                         [clubs_to, diamonds_to, hearts_to, spades_to]
                         where each value is in [0,1,2,3]
    
    Returns:
        JAX array with suits permuted according to the permutation scheme
        
    Raises:
        ValueError: If the permutation would change the highest bidder
    """
    # Constants from the game
    NUM_PLAYERS = 4
    NUM_CARDS = 52
    NUM_SUITS = 4
    NUM_CARDS_PER_SUIT = 13
    FEATURES_PER_CARD = 5  # 1 rank + 4 suit features
    NUM_TRICKS = 13
    
    # Validate permutation
    if len(suit_permutation) != NUM_SUITS or set(suit_permutation) != {0, 1, 2, 3}:
        raise ValueError("suit_permutation must be a permutation of [0,1,2,3]")
    
    # Helper function to permute 52-card sections efficiently
    def permute_52_card_sections(section: np.ndarray) -> None:
        """Permute a 52-card section by rearranging 4 blocks of 13 cards each.
        Can handle single section (52,) or batch of sections (..., 52)."""
        # Reshape to (..., 4, 13) for suit-wise operations
        cards_by_suit = section.reshape(*section.shape[:-1], NUM_SUITS, NUM_CARDS_PER_SUIT)
        temp = cards_by_suit.copy()
        for orig_suit in range(NUM_SUITS):
            new_suit = suit_permutation[orig_suit]
            cards_by_suit[..., new_suit, :] = temp[..., orig_suit, :]
    
    # Helper function to permute card features (rank + 4 suit features)
    def permute_card_features(features: np.ndarray) -> None:
        """Permute the 4 suit features in a 5-feature card representation."""        
        # Extract rank (first feature) and suit features (last 4)
        suit_features = features[..., 1:]  # Shape: (..., 4)
        temp = suit_features.copy()
        for orig_suit in range(NUM_SUITS):
            new_suit = suit_permutation[orig_suit]
            suit_features[..., new_suit] = temp[..., orig_suit]
    
    # 1. Permute hand section (52-dim multi-hot)
    start = 0
    end = NUM_CARDS
    hand_section = infostate_tensor[start:end]
    permute_52_card_sections(hand_section)
    
    # 2. Permute bid cards section (4 players * 5 features)
    start = end  
    end = start + NUM_PLAYERS * FEATURES_PER_CARD
    bid_cards_section = infostate_tensor[start:end]
    bid_cards_reshaped = bid_cards_section.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
    
    # Check if any bidding occurred (non-zero features)
    if bid_cards_section[0] == 0:  # Rank of this player's bidding card
        raise ValueError("Permutation is allowed only in trick-taking-phase state.")
    
    # Validate that permutation won't change highest bidder before applying it
    bid_values = bid_cards_reshaped[:, 0]
    max_bid_value = np.max(bid_values)
    tied_bidders = np.where(bid_values == max_bid_value)[0]
    
    if len(tied_bidders) > 1:
        # In case of tie, winner is determined by suit strength (spades > hearts > diamonds > clubs)
        tied_suits = bid_cards_reshaped[tied_bidders, 1:]  # Shape: (n_tied, 4)
        orig_suits = np.argmax(tied_suits, axis=1)
        new_suits = np.array([suit_permutation[orig_suit] for orig_suit in orig_suits])
        
        # Check if the ordering changes (highest suit index wins)
        if np.argmax(orig_suits) != np.argmax(new_suits):
            raise ValueError("Permutation would change the highest bidder due to suit tie-breaking")
    
    # Now apply the permutation
    permute_card_features(bid_cards_reshaped)
    
    # 3. Permute trump suit section (4-dim one-hot)
    start = end
    end = start + NUM_SUITS
    trump_suit_section = infostate_tensor[start:end]
    infostate_tensor[start:end] = trump_suit_section[suit_permutation]  # Advanced Numpy indexing
    
    # 5. Permute history section
    trick_len = NUM_PLAYERS + NUM_PLAYERS * FEATURES_PER_CARD
    start = end + 1
    end = start + NUM_TRICKS * trick_len
    
    # Reshape to separate leaders from cards, then process all cards at once
    history_section = infostate_tensor[start:end].reshape(NUM_TRICKS, trick_len)
    # Extract just the card features (skip the 4 leader features per trick)
    all_history_cards = history_section[:, NUM_PLAYERS:].reshape(NUM_TRICKS * NUM_PLAYERS, FEATURES_PER_CARD)
    permute_card_features(all_history_cards)
    
    # 6. Permute graveyard sections (3 opponents * 52 cards each)
    start = end
    end = start + (NUM_PLAYERS - 1) * NUM_CARDS
    graveyards_section = infostate_tensor[start:end].reshape(NUM_PLAYERS - 1, NUM_CARDS)
    permute_52_card_sections(graveyards_section)
    
    # 9. Permute current trick cards section (4 players * 5 features)
    start = end + NUM_PLAYERS + 1
    end = start + NUM_PLAYERS * FEATURES_PER_CARD
    current_trick_section = infostate_tensor[start:end].reshape(NUM_PLAYERS, FEATURES_PER_CARD)
    permute_card_features(current_trick_section)
    
    return infostate_tensor