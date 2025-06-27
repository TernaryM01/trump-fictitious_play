#ifndef OPEN_SPIEL_GAMES_TRUMP_TRUMP_H_
#define OPEN_SPIEL_GAMES_TRUMP_TRUMP_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <functional> // Required for std::function

#include "open_spiel/spiel.h"       
#include "open_spiel/spiel_utils.h" 
#include "open_spiel/abseil-cpp/absl/types/span.h" // For absl::Span

namespace open_spiel {
namespace trump {

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

// Structure to record completed tricks for history
struct TrickRecord {
  Player leader = kInvalidPlayer;
  std::array<int, kNumPlayers> cards; 
  Player winner = kInvalidPlayer;

  TrickRecord() {
    cards.fill(kInvalidCard);
  }
};

// Card representation constants for tensor encoding logic (used by EncodeCardToFeatureVector)
constexpr int kNumRankFeatures = 1; 
constexpr int kNumSuitFeatures = 4; 
constexpr int kFeaturesPerCard = kNumRankFeatures + kNumSuitFeatures; 
constexpr int kHiddenTrumpRankEncoding = 14; 

constexpr int kRankStrengthIndices[14] = {-1, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}; // Index=rank, value=strength


// ---- Inline Helper Functions (mimicking hearts.h style) ----

// Converts a card's game rank (1-13, Ace=1) to its tensor encoding rank (1 for '2', ..., 13 for 'Ace').
inline int GetTensorRankEncoding(int card_rank_1_to_13) {
    SPIEL_CHECK_GE(card_rank_1_to_13, 1);
    SPIEL_CHECK_LE(card_rank_1_to_13, 13);
    if (card_rank_1_to_13 == 1) return 13; 
    return card_rank_1_to_13 - 1; 
}

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

// Calculate bidding value of a card for determining initial numerical bids.
// Not inline if it has more complex logic, but can be if simple.
// For consistency with other card helpers, making it inline.
inline int BiddingValue(int card_index) {
  int rank = CardRank(card_index); 
  if (rank == 1) return 1;   
  if (rank >= 11 && rank <=13) return 0; 
  SPIEL_CHECK_GE(rank, 2);
  SPIEL_CHECK_LE(rank, 10);
  return rank;
}

// Get single-character string representation of a suit.
inline std::string SuitName(Suit suit) {
  switch (suit) {
    case Suit::kClubs:    return "C";
    case Suit::kDiamonds: return "D";
    case Suit::kHearts:   return "H";
    case Suit::kSpades:   return "S";
    default:
      // Fallback or error for kInvalidSuit if it can reach here
      if (suit == Suit::kInvalidSuit) return "Inv";
      SpielFatalError(absl::StrCat("Invalid suit: ", static_cast<int>(suit)));
      return "X"; 
  }
}

// Get string representation of a game rank (1-13).
inline std::string RankName(int rank) { 
  SPIEL_CHECK_GE(rank, 1);
  SPIEL_CHECK_LE(rank, kNumCardsPerSuit);
  switch (rank) {
    case 1:  return "A";
    case 10: return "T";
    case 11: return "J";
    case 12: return "Q";
    case 13: return "K";
    default: return std::to_string(rank);
  }
}

// Get string representation of a card (e.g., "SA", "HT", "C2").
inline std::string CardName(int card_index) {
  if (card_index == kInvalidCard) return "?"; 
  return absl::StrCat(SuitName(CardSuit(card_index)), RankName(CardRank(card_index)));
}

// Encodes a single card into the 5-feature representation for tensors.
void EncodeCardToFeatureVector(int card_index, absl::Span<float> feature_vector, 
                               int rank_encoding_override = -1);


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
  int sets_to_set_aside_;
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

class TrumpGame : public Game {
 public:
  explicit TrumpGame(const GameParameters& params); 
  
  // Standard OpenSpiel Game overrides
  int NumDistinctActions() const override { return kNumCards; } // Inline like HeartsGame
  int MaxChanceOutcomes() const override { return kNumCards; } // Inline like HeartsGame
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new TrumpState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override;
  double MaxUtility() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    return (4 + 3*4 + 9) + (4 + 2*4 + 9) + (4 + 4 + 9) // Dealing phase
         +  4 + 1                                      // Bidding phase
         +  4*13;                                      // Play phase
  }
};

}  // namespace trump
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TRUMP_TRUMP_H_