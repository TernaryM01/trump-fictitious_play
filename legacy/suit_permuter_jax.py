import jax
import jax.numpy as jnp

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
        start = NUM_CARDS
        end = start + NUM_PLAYERS * FEATURES_PER_CARD
        bid_cards_section = infostate_tensor[start:end]
        bid_cards_reshaped = bid_cards_section.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        
        # Check if any bidding occurred (non-zero rank for first player)
        bidding_occurred = bid_cards_reshaped[0, 0] > 0
        # If no bidding, permutation is not allowed
        valid = bidding_occurred
        
        # Check if permutation would change highest bidder
        bid_values = bid_cards_reshaped[:, 0]
        max_bid_value = jnp.max(bid_values)
        
        # Find tied bidders
        is_tied = bid_values == max_bid_value
        # If only one bidder has max value, permutation is safe
        single_winner = jnp.sum(is_tied) == 1
        
        # If multiple tied bidders, check if the tie-breaking winner changes
        def check_suit_tie_breaking():
            # Get indices of tied bidders
            tied_bidders = jnp.where(is_tied)[0]
            # Extract suit features for tied bidders only
            tied_suits = bid_cards_reshaped[tied_bidders, 1:]  # Shape: (n_tied, 4)
            orig_suits = jnp.argmax(tied_suits, axis=1)
            new_suits = suit_permutation[orig_suits]
            
            # Check if the ordering changes
            return jnp.argmax(orig_suits) == jnp.argmax(new_suits)
        
        # Permutation is valid if single winner OR tie-breaking doesn't change
        valid = valid & (single_winner | check_suit_tie_breaking())
        
        return valid
    
    # Apply permutation only if valid
    def apply_permutation() -> jax.Array:
        """Apply the suit permutation to all relevant sections."""
        result = infostate_tensor
        
        # 1. Permute hand section (52-dim multi-hot)
        start = 0
        end = NUM_CARDS
        hand_section = result[start:end]
        permuted_hand = permute_52_card_sections(hand_section)
        result = result.at[start:end].set(permuted_hand)
        
        # 2. Permute bid cards section (4 players * 5 features)
        start = end
        end = start + NUM_PLAYERS * FEATURES_PER_CARD
        bid_cards_section = result[start:end]
        bid_cards_reshaped = bid_cards_section.reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_bid_cards = permute_card_features(bid_cards_reshaped)
        result = result.at[start:end].set(permuted_bid_cards.flatten())
        
        # 3. Permute trump suit section (4-dim one-hot)
        start = end
        end = start + NUM_SUITS
        trump_suit_section = result[start:end]
        permuted_trump = trump_suit_section[suit_permutation]
        result = result.at[start:end].set(permuted_trump)
        
        # 4. Skip round bid status section
        start = end + 1
        
        # 5. Permute history section
        trick_len = NUM_PLAYERS + NUM_PLAYERS * FEATURES_PER_CARD
        end = start + NUM_TRICKS * trick_len
    
        # Reshape to separate leaders from cards, then process all cards at once
        history_section = result[start:end].reshape(NUM_TRICKS, trick_len)
        # Extract just the card features (skip the 4 leader features per trick)
        all_history_cards = history_section[:, NUM_PLAYERS:].reshape(NUM_TRICKS * NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_history_cards = permute_card_features(all_history_cards)
        
        # Reconstruct history section
        new_history_section = jnp.concatenate([
            history_section[:, :NUM_PLAYERS],  # Leaders: (13, 4)
            permuted_history_cards.reshape(NUM_TRICKS, NUM_PLAYERS * FEATURES_PER_CARD)  # Cards: (13, 20)
        ], axis=1)  # Result: (13, 24)
        new_history = new_history_section.flatten()
        result = result.at[start:end].set(new_history)
        
        # 6. Permute graveyard sections (3 opponents * 52 cards each)
        start = end
        end = start + (NUM_PLAYERS - 1) * NUM_CARDS
        graveyards_section = result[start:end].reshape(NUM_PLAYERS - 1, NUM_CARDS)
        permuted_graveyards = permute_52_card_sections(graveyards_section)
        result = result.at[start:end].set(permuted_graveyards.flatten())
        
        # 7&8. Skip ANTC & break occurred
        start = end + NUM_PLAYERS + 1
        
        # 9. Permute current trick cards section (4 players * 5 features)
        end = start + NUM_PLAYERS * FEATURES_PER_CARD
        current_trick_section = result[start:end].reshape(NUM_PLAYERS, FEATURES_PER_CARD)
        permuted_current_trick = permute_card_features(current_trick_section)
        result = result.at[start:end].set(permuted_current_trick.flatten())
        
        return result
    
    # Return permuted tensor if valid, otherwise return original
    return jax.lax.cond(
        is_valid_permutation(),
        apply_permutation,
        lambda: infostate_tensor
    )