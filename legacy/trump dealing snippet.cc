TrumpGame::TrumpGame(const GameParameters& params)
    : Game(kGameType, params) {
}

TrumpState::TrumpState(std::shared_ptr<const Game> game)
    : State(game),
      num_players_(game->NumPlayers()), 
      current_player_(kChancePlayerId), 
      phase_(Phase::kDeal),

      dealing_subphase_(DealingSubPhase::kDealPerSuit),
      dealing_player_(0),
      suit_index_(0),
      cards_dealt_in_subphase_(0),

      round_bid_status_(0), 
      trick_number_(0),
      current_trick_starter_(kInvalidPlayer),
      current_trick_card_count_(0),
      high_low_round_(HighLowRound::kUndecided), 
      trump_suit_(Suit::kInvalidSuit),
      highest_bidder_(kInvalidPlayer),
      trump_break_occurred_(false) { 
  SPIEL_CHECK_EQ(this->num_players_, kNumPlayers); 

  deck_.resize(kNumCards); 
  std::iota(deck_.begin(), deck_.end(), 0); 

  is_undealt_.resize(kNumCards, true);
  is_set_aside_.resize(kNumCards, false);

  hands_.resize(this->num_players_);
  for (int p = 0; p < this->num_players_; ++p) {
    hands_[p].reserve(kNumCardsPerPlayer); 
  }
  
  bids_card_index_.resize(this->num_players_, kInvalidCard);
  bid_values_.resize(this->num_players_, -1); 
  
  tricks_won_.resize(this->num_players_, 0);
  current_trick_cards_.fill(kInvalidCard);
}

Player TrumpState::CurrentPlayer() const { 
  return current_player_; 
}

std::vector<Action> TrumpState::LegalActions() const {
  std::vector<Action> actions;
  if (IsTerminal()) return actions;

  if (State::IsChanceNode()) { 
    if (phase_ == Phase::kDeal) {
      if (dealing_subphase_ == DealingSubPhase::kDealPerSuit) {
        Suit current_suit = static_cast<Suit>(suit_index_);
        for (int card_idx = 0; card_idx < kNumCards; ++card_idx) {
          if (is_undealt_[card_idx] && CardSuit(card_idx) == current_suit) {
            actions.push_back(card_idx);
          }
        }
      } else if (dealing_subphase_ == DealingSubPhase::kSetAside) {
        Suit current_suit = static_cast<Suit>(suit_index_);
        for (int card_idx = 0; card_idx < kNumCards; ++card_idx) {
          if (is_undealt_[card_idx] && CardSuit(card_idx) == current_suit) {
            actions.push_back(card_idx);
          }
        }
      } else if (dealing_subphase_ == DealingSubPhase::kDealRemaining) {
        for (int card_idx = 0; card_idx < kNumCards; ++card_idx) {
          if (is_undealt_[card_idx] && !is_set_aside_[card_idx]) {
            actions.push_back(card_idx);
          }
        }
      }
      return actions;
    }
  }
  
  if (phase_ == Phase::kBidding) {
    for (int card_idx : hands_[current_player_]) {
        actions.push_back(card_idx);
    }
  } else if (phase_ == Phase::kHighLowDecision) {
    SPIEL_CHECK_EQ(current_player_, highest_bidder_);
    actions.push_back(0); 
    actions.push_back(1); 
  } else if (phase_ == Phase::kPlay) {
    const std::vector<int>& player_hand = hands_[current_player_];
    if (IsCurrentTrickEmpty()) { 
        bool only_trumps_in_hand = true;
        for (int card_in_hand : player_hand) {
            if (CardSuit(card_in_hand) != trump_suit_) { 
                only_trumps_in_hand = false;
                break;
            }
        }
        for (int card_to_play : player_hand) {
            if (CardSuit(card_to_play) != trump_suit_ || 
                trump_break_occurred_ ||                 
                only_trumps_in_hand) {                   
                actions.push_back(card_to_play);
            }
        }
    } else { 
        SPIEL_CHECK_NE(current_trick_starter_, kInvalidPlayer);
        SPIEL_CHECK_NE(current_trick_cards_[current_trick_starter_], kInvalidCard);
        
        Suit leading_s = CardSuit(current_trick_cards_[current_trick_starter_]); 
        bool has_leading_suit_card = false;
        for (int card_in_hand : player_hand) {
            if (CardSuit(card_in_hand) == leading_s) { 
                actions.push_back(card_in_hand);
                has_leading_suit_card = true;
            }
        }

        if (!has_leading_suit_card) { 
            std::vector<int> possible_nontrump_sluffs;
            std::vector<int> possible_trump_plays;
            bool has_any_nontrump_in_hand = false;

            for (int card_in_hand : player_hand) {
                if (CardSuit(card_in_hand) != trump_suit_) { 
                    possible_nontrump_sluffs.push_back(card_in_hand);
                    has_any_nontrump_in_hand = true;
                } else {
                    possible_trump_plays.push_back(card_in_hand);
                }
            }

            for (int card : possible_nontrump_sluffs) { 
                actions.push_back(card);
            }
            if (trump_break_occurred_ || !has_any_nontrump_in_hand) {
                for (int card : possible_trump_plays) {
                    actions.push_back(card);
                }
            }
        }
    }
  }
  return actions;
}

void TrumpState::DoApplyAction(Action action) {
  if (State::IsChanceNode()) {  // phase_ == Phase::kDeal
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kNumCards);
    SPIEL_CHECK_TRUE(is_undealt_[action]);
    
    if (dealing_subphase_ == DealingSubPhase::kDealPerSuit) {
      hands_[dealing_player_].push_back(action);
      is_undealt_[action] = false;
      suit_index_++;
      if (suit_index_ == kNumSuits) {
        dealing_subphase_ = DealingSubPhase::kSetAside;
        suit_index_ = 0;
      }
    } else if (dealing_subphase_ == DealingSubPhase::kSetAside) {
      is_set_aside_[action] = true;
      suit_index_++;
      if (suit_index_ == kNumSuits) {
        dealing_subphase_ = DealingSubPhase::kDealRemaining;
        cards_dealt_in_subphase_ = 0;
      }
    } else if (dealing_subphase_ == DealingSubPhase::kDealRemaining) {
      hands_[dealing_player_].push_back(action);
      is_undealt_[action] = false;
      cards_dealt_in_subphase_++;
      if (cards_dealt_in_subphase_ == 9) {
        for (int c = 0; c < kNumCards; ++c) {
          if (is_set_aside_[c]) {
            is_set_aside_[c] = false;
          }
        }
        if (dealing_player_ == 2) {
          for (int c = 0; c < kNumCards; ++c) {
            if (is_undealt_[c]) {
              hands_[3].push_back(c);
              is_undealt_[c] = false;
            }
          }
          phase_ = Phase::kBidding;
          current_player_ = 0;
        } else {
          dealing_player_++;
          dealing_subphase_ = DealingSubPhase::kDealPerSuit;
          suit_index_ = 0;
        }
      }
    }
    return;
  }

  if (phase_ == Phase::kBidding) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(hands_[current_player_], action)); 
    bids_card_index_[current_player_] = action; 
    
    current_player_ = (current_player_ + 1) % num_players_;
    if (current_player_ == 0) {  // All players have finished bidding
      DetermineBidWinnerAndTrump(); 
    }
  } else if (phase_ == Phase::kHighLowDecision) {
    SPIEL_CHECK_EQ(current_player_, highest_bidder_);
    if (action == 0) { // Ascend
      high_low_round_ = HighLowRound::kHigh;
      round_bid_status_ = 2; 
      for (int p = 0; p < num_players_; ++p) {
        SPIEL_CHECK_GE(bid_values_[p], 0); 
        bid_values_[p]++; 
      }
    } else { // Descend (action == 1)
      high_low_round_ = HighLowRound::kLow;
      round_bid_status_ = -2; 
      for (int p = 0; p < num_players_; ++p) {
        if (bid_values_[p] > 0) { 
             bid_values_[p]--;
        }
      }
    }
    phase_ = Phase::kPlay;
  } else if (phase_ == Phase::kPlay) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(hands_[current_player_], action));

    if (IsCurrentTrickEmpty()) { 
        current_trick_starter_ = current_player_;
    }
    SPIEL_CHECK_NE(current_trick_starter_, kInvalidPlayer); 

    hands_[current_player_].erase(
        std::remove(hands_[current_player_].begin(), hands_[current_player_].end(), action),
        hands_[current_player_].end());
    
    current_trick_cards_[current_player_] = action;
    current_trick_card_count_++;

    if (IsCurrentTrickComplete()) {
      Player winner = ResolveTrick(); 
      tricks_won_[winner]++;
      
      TrickRecord completed_trick;
      completed_trick.leader = current_trick_starter_;
      completed_trick.cards = current_trick_cards_; 
      completed_trick.winner = winner;
      tricks_history_.push_back(completed_trick);
      
      trick_number_++;
      
      if (trick_number_ >= kNumTricks) {
        phase_ = Phase::kGameOver;
        current_player_ = kTerminalPlayerId;
      } else {
        // Clear trick data only when starting a new trick
        current_trick_cards_.fill(kInvalidCard);
        current_trick_card_count_ = 0;
        current_trick_starter_ = winner;
        current_player_ = winner; 
      }
    } else { 
      current_player_ = (current_player_ + 1) % num_players_;
    }
  }
}

std::vector<std::pair<Action, double>> TrumpState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(State::IsChanceNode());
  std::vector<Action> legal_actions = LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty());
  double probability = 1.0 / static_cast<double>(legal_actions.size());
  std::vector<std::pair<Action, double>> outcomes;
  for (Action action : legal_actions) {
    outcomes.push_back({action, probability});
  }
  return outcomes;
}