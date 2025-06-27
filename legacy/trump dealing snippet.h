// Game Constants
constexpr int kNumPlayers = 4;
constexpr int kNumCards = 52;
constexpr int kNumSuits = 4;
constexpr int kNumCardsPerSuit = 13;
constexpr int kNumCardsPerPlayer = kNumCards / kNumPlayers;
constexpr int kNumTricks = kNumCardsPerPlayer;

constexpr int kInvalidCard = -1; 
constexpr Player kInvalidPlayer = -1; 

// Enums for game state clarity
enum class Suit {
  kClubs = 0,
  kDiamonds = 1,
  kHearts = 2,
  kSpades = 3,
  kInvalidSuit = -1 
};

enum class Phase {
  kDeal = 0,
  kBidding = 1,
  kHighLowDecision = 2,
  kPlay = 3,
  kGameOver = 4
};

enum class DealingSubPhase {
  kDealPerSuit = 0,
  kSetAside = 1,
  kDealRemaining = 2
};

enum class HighLowRound {
  kUndecided = 0,
  kHigh = 1,
  kLow = 2
};

// Get the canonical index of a card (0-51). Rank parameter is 1-13 (1=Ace, 13=King).
inline int CardIndex(Suit suit, int rank) {
  SPIEL_CHECK_GE(rank, 1);
  SPIEL_CHECK_LE(rank, kNumCardsPerSuit);
  return static_cast<int>(suit) * kNumCardsPerSuit + (rank - 1);
}

// Convert a card's index (0-51) to its suit.
inline Suit CardSuit(int card_index) {
  return static_cast<Suit>(card_index / kNumCardsPerSuit);
}

// Convert a card's index (0-51) to its game rank (1-13, Ace=1, King=13).
inline int CardRank(int card_index) { 
  return (card_index % kNumCardsPerSuit) + 1;
}

class TrumpGame; // Forward declaration

class TrumpState : public State {
 public:
  explicit TrumpState(std::shared_ptr<const Game> game);
  TrumpState(const TrumpState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; } // Inline like HeartsState
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override { // Inline like HeartsState
    return std::unique_ptr<State>(new TrumpState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  
  // IsChanceNode() is NOT overridden; will use State::IsChanceNode().
  // ResampleFromInfostate() is NOT overridden; will use State::ResampleFromInfostate()
  // (which typically fatals if not implemented, which is fine for this game).
  // If compiler errors indicate these MUST be overridden due to being pure virtual in a base
  // for some reason, or if custom logic is needed, then declarations would be added back here.

  std::string PhaseToString() const; 

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  // Trump-specific private helper methods

  bool IsCurrentTrickEmpty() const;
  bool IsCurrentTrickComplete() const;
  void DetermineBidWinnerAndTrump();
  int ResolveTrick(); 
  bool IsCardStronger(int card1_idx, int card2_idx) const; 

  // Game state variables
  const int num_players_; 
  Player current_player_;
  Phase phase_;
  int round_bid_status_; 
  int trick_number_; 
  
  DealingSubPhase dealing_subphase_;
  int dealing_player_;
  int suit_index_;
  int cards_dealt_in_subphase_;
  std::vector<bool> is_undealt_;
  std::vector<bool> is_set_aside_;
  
  std::vector<int> deck_; 
  std::vector<std::vector<int>> hands_; 
  std::vector<int> bids_card_index_; 
  std::vector<int> bid_values_;      
  HighLowRound high_low_round_; 
  Suit trump_suit_;
  Player highest_bidder_;
  bool trump_break_occurred_; 
  Player current_trick_starter_;             
  std::array<int, kNumPlayers> current_trick_cards_;
  int current_trick_card_count_;
  std::vector<int> tricks_won_;              
  std::vector<TrickRecord> tricks_history_;  
  
  friend class TrumpGame; 
};