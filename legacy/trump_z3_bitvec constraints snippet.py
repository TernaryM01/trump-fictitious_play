    # Create variables for each opponent's initial hand
    opponent_hands = [z3.BitVec(f"opp_{opp}_hand", NUM_CARDS) for opp in range(NUM_PLAYERS - 1)]

    def add_initial_hand_and_played_cards_domain_constraints(solver, opponent_hands, opponent_plays):
        """
        Adds constraints for opponents' initial hands:
        1. Each opponent has exactly 13 cards.
        2. Each opponent has at least one card of each suit.
        3. Each card is not owned by more than one opponent.
        and domain constraints for the current trick cards.
        """
        SUIT_MASKS = []
        for suit in range(NUM_SUITS):
            mask = 0
            for rank in range(NUM_CARDS_PER_SUIT):
                card_idx = suit * NUM_CARDS_PER_SUIT + rank
                mask |= (1 << card_idx)
            SUIT_MASKS.append(mask)
            
        for opp in range(NUM_PLAYERS - 1):
            num_cards_hand_constraint = z3.Sum([z3.If(opponent_hands[opp] & (1 << card) != 0, 1, 0)
                                                    for card in range(NUM_CARDS)]) == NUM_CARDS_PER_PLAYER
            # num_cards_hand_constraint = z3.PbEq([(opponent_hands[opp] & (1 << card) != 0, 1)
            #                                         for card in range(NUM_CARDS)], NUM_CARDS_PER_PLAYER)
            solver.add(num_cards_hand_constraint)
            
            for suit in range(NUM_SUITS):
                suit_in_hand_constraint = (opponent_hands[opp] & SUIT_MASKS[suit]) != 0
                solver.add(suit_in_hand_constraint)

        for card in range(NUM_CARDS):
            no_two_owners_constraint = z3.Sum([z3.If(opponent_hands[opp] & (1 << card) != 0, 1, 0)
                                               for opp in range(NUM_PLAYERS - 1)]) <= 1
            solver.add(no_two_owners_constraint)
            # solver.add(z3.PbLe([(opponent_hands[opp] & (1 << card) != 0, 1) 
            #                     for opp in range(NUM_PLAYERS - 1)], 1))

        # Add domain constraints for opponent plays
        for opp in range(NUM_PLAYERS - 1):
            domain_constraint = z3.And([opponent_plays[opp] >= 0, opponent_plays[opp] < NUM_CARDS])
            # domain_constraint = z3.Or([opponent_plays[opp] == card for card in range(NUM_CARDS)])
            solver.add(domain_constraint)

    def add_graveyard_and_current_trick_constraints(solver, opponent_hands, graveyards,
                                                    opponent_plays, current_trick_opp_cards, trump_suit):
        """Add constraints based on graveyard information and cards already played in current trick."""
        for opp in range(NUM_PLAYERS - 1):
            card_played = current_trick_opp_cards[opp]

            if card_played != CARD_NOT_YET_PLAYED:  # Card has been played
                if card_played == HIDDEN_TRUMP_CARD:  # Face-down trump card
                    # Must be a trump suit card
                    trump_cards = [trump_suit * NUM_CARDS_PER_SUIT + rank for rank in range(NUM_CARDS_PER_SUIT)]
                    solver.add(z3.Or([opponent_plays[opp] == card for card in trump_cards]))
                    solver.add(z3.And([opponent_plays[opp] >= trump_suit * NUM_CARDS_PER_SUIT,
                                       opponent_plays[opp] < (trump_suit + 1) * NUM_CARDS_PER_SUIT]))
                else:  # Specific card played
                    solver.add(opponent_plays[opp] == card_played)

            # Add constraints based on graveyard information
            for card in range(NUM_CARDS):
                graveyard_value = graveyards[opp][card]
                if graveyard_value == -1:  # Opponent has this card (unplayed)
                    solver.add((opponent_hands[opp] & (1 << card)) != 0)
                elif graveyard_value == 1:  # Opponent has played this card before
                    solver.add((opponent_hands[opp] & (1 << card)) != 0)
                    solver.add(opponent_plays[opp] != card)  # Opponent is surely not playing a card already played.
                elif graveyard_value == 2:  # Opponent never had this card
                    solver.add((opponent_hands[opp] & (1 << card)) == 0)
                # For value 0, no constraint on initial hand or current trick card played.

                # Played card must be from opponent's initial hand
                hand_constraint = z3.Implies(opponent_plays[opp] == card, (opponent_hands[opp] & (1 << card)) != 0)
                solver.add(hand_constraint)

    def add_legality_constraints(solver, opponent_hands, opponent_plays,
                                leading_suit, trump_suit, break_occurred, graveyards):
        """Add constraints for legal card play
        See also non-Z3 counterpart for player: `can_play_card(...)`.
        """
        for opp in range(3):
            # Must follow suit if possible
            available_leading_mask = 0
            for card in range(leading_suit * NUM_CARDS_PER_SUIT, (leading_suit + 1) * NUM_CARDS_PER_SUIT):
                if graveyards[opp][card] < 1:  # Filter out cards already played
                    available_leading_mask |= (1 << card)
            has_leading_suit = (opponent_hands[opp] & available_leading_mask) != 0
            
            opp_card = opponent_plays[opp]
            # follows_suit = z3.Or([opp_card == card for card in leading_suit_cards])
            follows_suit = z3.And([opp_card >= leading_suit * NUM_CARDS_PER_SUIT,
                                   opp_card < (leading_suit + 1) * NUM_CARDS_PER_SUIT])
            solver.add(z3.Implies(has_leading_suit, follows_suit))

            # "Break" rule
            if not break_occurred and leading_suit != trump_suit:
                available_nontrump_mask = 0
                for card in range(NUM_CARDS):
                    if card < trump_suit * NUM_CARDS_PER_SUIT or card >= (trump_suit + 1) * NUM_CARDS_PER_SUIT:  # Not trump
                        if graveyards[opp][card] < 1:  # Filter out cards already played
                            available_nontrump_mask |= (1 << card)
                has_nontrump = (opponent_hands[opp] & available_nontrump_mask) != 0
                
                must_play_nontrump = z3.Or([opp_card < trump_suit * NUM_CARDS_PER_SUIT,
                                            opp_card >= (trump_suit + 1) * NUM_CARDS_PER_SUIT])
                
                solver.add(z3.Implies(has_nontrump, must_play_nontrump))

    def extract_scenario_from_solver(solver, graveyards):
        """Extract opponent hands and plays from Z3 model"""
        # Create auxiliary Boolean variables for each opponent-card combination
        aux_vars = []
        for opp in range(NUM_PLAYERS - 1):
            aux_vars.append([])
            for card in range(NUM_CARDS):
                var_name = f"opp_{opp}_has_card_{card}"
                aux_vars[opp].append(z3.Bool(var_name))
        
        solver.push()
        try:
            # Add constraints linking auxiliary variables to bit-vector variables
            for opp in range(NUM_PLAYERS - 1):
                for card in range(NUM_CARDS):
                    # aux_var is true iff the corresponding bit in opponent_hands[opp] is set
                    bit_is_set = (opponent_hands[opp] & (1 << card)) != 0
                    solver.add(aux_vars[opp][card] == bit_is_set)

            # Extract opponent hands
            solver.check()
            model = solver.model()
            print("Opponent Current Hands:")
            for opp in range(NUM_PLAYERS - 1):
                print(f"  P{opp + 1}: ", end="")
                
                # Collect all cards for this opponent
                cards = []
                for card in [card for card in range(NUM_CARDS) if graveyards[opp][card] < 1]:  # Filter out cards already played
                    var = model[aux_vars[opp][card]]
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
        finally:
            solver.pop()