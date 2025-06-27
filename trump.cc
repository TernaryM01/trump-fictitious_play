#include "open_spiel/games/trump/trump.h" // Primary header for this game

#include <algorithm> // For std::sort, std::remove, std::fill, std::iota
#include <array>     // For std::array
#include <memory>    // For std::shared_ptr, std::make_shared, std::unique_ptr
#include <numeric>   // For std::iota
#include <string>    // For std::string, std::to_string
#include <vector>    // For std::vector
#include <map>       // MIMICKING hearts.cc
#include <set>       // For std::set

#include "open_spiel/abseil-cpp/absl/algorithm/container.h" 
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h" // MIMICKING hearts.cc
#include "open_spiel/abseil-cpp/absl/types/optional.h"   // MIMICKING hearts.cc
#include "open_spiel/game_parameters.h" // For GameParameters
#include "open_spiel/spiel.h"           // For Game, State, GameType, Player, Action, etc.
#include "open_spiel/spiel_globals.h"   // For kChancePlayerId, kTerminalPlayerId, etc.
#include "open_spiel/spiel_utils.h"     // For SPIEL_CHECK_*, SpielFatalError
// tensor_observer.h is NOT explicitly included, RegisterSingleTensorObserver is assumed from spiel.h/utils.h

namespace open_spiel {
namespace trump { 

// Helper functions defined inline in trump.h are used directly by class methods.
// Constants are also defined in trump.h.

namespace { // Anonymous namespace for game registration items ONLY, as per hearts.cc

// GameType definition (uses constants from trump.h like kNumPlayers)
const GameType kGameType{ // Use a distinct name if kGameType is also in trump.h's global namespace
    /*short_name=*/"trump",
    /*long_name=*/"Trump",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers, 
    /*min_num_players=*/kNumPlayers, 
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true, 
    /*provides_observation_tensor=*/true,  
    /*parameter_specification=*/{}
};
// Note: If trump.h declares 'extern const GameType kGameType;', then the definition
// should be outside anonymous namespace: 'const GameType kGameType = {...};'
// For now, following hearts.cc which defines its kGameType in anonymous.

// Factory function
std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TrumpGame(params));
}

// Game registration
REGISTER_SPIEL_GAME(kGameType, Factory); // Use the kGameType from this scope

// Default tensor observer registration
open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace (anonymous)


// ---------------- Game Class Method Definitions -------------------

TrumpGame::TrumpGame(const GameParameters& params)
    : Game(kGameType, params) { // Use kGameType from anonymous namespace
      // Trump game does not have parameters like HeartsGame does, so constructor is simpler.
}

double TrumpGame::MinUtility() const {
  return -26.0 - 7.0;  // Everybody bids 0, you collect all 13 tricks.
}

double TrumpGame::MaxUtility() const {
  return 10.0 + 18.0;  // Everybody bids 10, you collect 10 while opponents collect 1 each.
}

std::vector<int> TrumpGame::InformationStateTensorShape() const {
  // 1. Hand: 52
  // 2. Bid Cards: 4 * 5 = 20
  // 3. Trump Suit: 4
  // 4. Round Bid Status: 1
  // 5. History: 13 * (4 + 4 * 5) = 312
  // 6. Opponent Graveyard: 3 * 52 = 156
  // 7. ANTC: 4
  // 8. Break Occurred: 1
  // 9. Current Trick Cards: 4 * 5 = 20
  // 10. Current Trick Leader: 4
  // 11. Current Trick Trumps: 13
  // 12. Current Trick Number: 1
  return { 588 };
}

std::vector<int> TrumpGame::ObservationTensorShape() const {
  return InformationStateTensorShape();
}


// ---------------- State Class Method Definitions -------------------

TrumpState::TrumpState(std::shared_ptr<const Game> game)
    : State(game),
      num_players_(game->NumPlayers()), 
      current_player_(kChancePlayerId), 
      phase_(Phase::kDeal),

      dealing_subphase_(DealingSubPhase::kDealPerSuit),
      dealing_player_(0),
      suit_index_(0),
      cards_dealt_in_subphase_(0),
      sets_to_set_aside_(3), // Player 0 needs 3 sets of 4 cards set aside

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

std::string TrumpState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal ", CardName(action));
  } else if (phase_ == Phase::kBidding) {
    return absl::StrCat("Bid with ", CardName(action));
  } else if (phase_ == Phase::kHighLowDecision) {
    return action == 0 ? "Ascend" : "Descend";
  } else if (phase_ == Phase::kPlay) {
    return absl::StrCat("Play ", CardName(action));
  }
  return "Unknown action";
}

std::string TrumpState::PhaseToString() const {
  switch (phase_) {
    case Phase::kDeal:            return "Deal";
    case Phase::kBidding:         return "Bidding";
    case Phase::kHighLowDecision: return "HighLowDecision";
    case Phase::kPlay:            return "Play";
    case Phase::kGameOver:        return "GameOver";
    default:                      return "UnknownPhase";
  }
}

std::string TrumpState::ToString() const {
  // TODO: This function has not been updated with the infostate tensor function
  // having undergone many changes. Expect it to have gone a bit out of sync.
  std::string rv = absl::StrCat("Phase: ", PhaseToString(), "\n");
  if (current_player_ == kChancePlayerId) {
      absl::StrAppend(&rv, "Current Player: Chance\n");
  } else if (current_player_ == kTerminalPlayerId) {
      absl::StrAppend(&rv, "Current Player: Terminal\n");
  } else if (current_player_ >= 0 && current_player_ < num_players_) {
      absl::StrAppend(&rv, "Current Player: P", current_player_, "\n");
  } else {
      absl::StrAppend(&rv, "Current Player: Invalid (", current_player_, ")\n");
  }

  if (phase_ != Phase::kDeal) { 
     absl::StrAppend(&rv, "Hands (visible in full state string):\n");
     for (int p = 0; p < num_players_; ++p) {
       absl::StrAppend(&rv, "P", p, ": ");
       std::vector<std::string> card_names_str;
       std::vector<int> sorted_hand_display = hands_[p];
       std::sort(sorted_hand_display.begin(), sorted_hand_display.end());
       for (int c : sorted_hand_display) {
         card_names_str.push_back(CardName(c));
       }
       absl::StrAppend(&rv, absl::StrJoin(card_names_str, " "), "\n");
     }
  }

  if (phase_ == Phase::kBidding) {
    absl::StrAppend(&rv, "Bid Cards Status: (All bids are hidden until revealed simultaneously after this phase)\n");
  } else if (phase_ > Phase::kBidding && phase_ != Phase::kDeal) { 
    absl::StrAppend(&rv, "Revealed Bid Cards: ");
    for (int p = 0; p < num_players_; ++p) {
      absl::StrAppend(&rv, "P", p, ":", CardName(bids_card_index_[p]), " ");
    }
    absl::StrAppend(&rv, "\n");
  }

  if (phase_ >= Phase::kHighLowDecision || (phase_ == Phase::kPlay && round_bid_status_ != 0) || phase_ == Phase::kGameOver) {
    std::string round_bid_status_str;
    switch(round_bid_status_){
        case -2: round_bid_status_str = "Descend Occurred"; break;
        case -1: round_bid_status_str = "Normal Low Round (Sum < 13)"; break;
        case 0: round_bid_status_str = (phase_ == Phase::kBidding || phase_ == Phase::kDeal) ? "Bidding/Undetermined" : "Error: Bid Status 0 Post-Bidding"; break;
        case 1: round_bid_status_str = "Normal High Round (Sum > 13)"; break;
        case 2: round_bid_status_str = "Ascend Occurred"; break;
        default: round_bid_status_str = "Invalid Status";
    }
    absl::StrAppend(&rv, "Round Determination: ", round_bid_status_str, "\n");
    absl::StrAppend(&rv, "Resulting Round Type (for scoring): ", (high_low_round_ == HighLowRound::kHigh ? "High" : (high_low_round_ == HighLowRound::kLow ? "Low" : "Undecided")), "\n");

    absl::StrAppend(&rv, "Trump suit: ", 
                    (trump_suit_ == Suit::kInvalidSuit) ? "Undecided" : SuitName(trump_suit_), "\n");
    absl::StrAppend(&rv, "Internal Target Bid values (for scoring logic): "); 
    for (int p = 0; p < num_players_; ++p) {
      absl::StrAppend(&rv, "P", p, ":", bid_values_[p] == -1 ? "?" : std::to_string(bid_values_[p]), " ");
    }
    absl::StrAppend(&rv, "\n");
    if(highest_bidder_ != kInvalidPlayer) 
        absl::StrAppend(&rv, "Highest bidder (determined trump): P", highest_bidder_, "\n");
  }

  if (phase_ == Phase::kPlay || phase_ == Phase::kGameOver) {
    absl::StrAppend(&rv, "Trick #", trick_number_ +1, "\n"); 
    if (current_trick_starter_ != kInvalidPlayer) {
        absl::StrAppend(&rv, "Current trick (Leader P", current_trick_starter_, "): ");
        for (int i = 0; i < num_players_; ++i) {
            Player p_in_order = (current_trick_starter_ + i) % num_players_;
            int card_played = current_trick_cards_[p_in_order];
            if (card_played != kInvalidCard) {
                bool is_trump = CardSuit(card_played) == trump_suit_;
                if (is_trump && !IsCurrentTrickComplete()) {
                    absl::StrAppend(&rv, "P", p_in_order, ":(Tr?) ");
                } else {
                    absl::StrAppend(&rv, "P", p_in_order, ":", CardName(card_played), " ");
                }
            }
        }
        absl::StrAppend(&rv, "\n");
    }

    absl::StrAppend(&rv, "Trump break occurred (from previous tricks): ", trump_break_occurred_ ? "Yes" : "No", "\n");
    absl::StrAppend(&rv, "Tricks won: ");
    for (int p = 0; p < num_players_; ++p) {
      absl::StrAppend(&rv, "P", p, ":", tricks_won_[p], " ");
    }
    absl::StrAppend(&rv, "\n");

    if (!tricks_history_.empty()) {
        absl::StrAppend(&rv, "Previous Tricks (Most Recent First):\n");
        for (int t_idx = tricks_history_.size() - 1; t_idx >= 0; --t_idx) {
            const auto& trick = tricks_history_[t_idx];
            absl::StrAppend(&rv, "  Hist.Trick ", t_idx + 1, " (Lead P", trick.leader, "): ");
            for (int i = 0; i < num_players_; ++i) {
                Player p_in_actual_play_order = (trick.leader + i) % num_players_;
                absl::StrAppend(&rv, "P", p_in_actual_play_order, ":", CardName(trick.cards[p_in_actual_play_order]), " ");
            }
            absl::StrAppend(&rv, "-> Won by P", trick.winner, "\n");
        }
    }
  }

  if (phase_ == Phase::kGameOver) {
    absl::StrAppend(&rv, "Scores: ");
    std::vector<double> returns = Returns();
    for (int p = 0; p < num_players_; ++p) {
      absl::StrAppend(&rv, "P", p, ":", returns[p], " ");
    }
    absl::StrAppend(&rv, "\n");
  }
  return rv;
}

void EncodeCardToFeatureVector(int card_index, absl::Span<float> feature_vector,
                               int rank_encoding_override) {
    SPIEL_CHECK_EQ(feature_vector.size(), kFeaturesPerCard);
    std::fill(feature_vector.begin(), feature_vector.end(), 0.0f);
    if (card_index == kInvalidCard) {
        return;
    }

    int actual_rank_for_tensor_encoding;
    if (rank_encoding_override != -1) {
        actual_rank_for_tensor_encoding = rank_encoding_override;
    } else {
        actual_rank_for_tensor_encoding = GetTensorRankEncoding(CardRank(card_index));
    }
    
    feature_vector[0] = static_cast<float>(actual_rank_for_tensor_encoding);

    Suit suit = CardSuit(card_index);
    if (static_cast<int>(suit) >= 0 && static_cast<int>(suit) < kNumSuits) {
       feature_vector[kNumRankFeatures + static_cast<int>(suit)] = 1.0f; // one-hot suit encoding
    } else {
       SpielFatalError(absl::StrCat("EncodeCardToFeatureVector: Invalid suit for encoding: ", static_cast<int>(suit)));
    }
}

std::vector<double> TrumpState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }
  std::vector<double> raw_scores(num_players_); // Step 1: Calculate raw scores first
  for (int p = 0; p < num_players_; ++p) {
    if (bid_values_[p] == -1) { 
        SpielFatalError(absl::StrCat("Player ", p, " has an unassigned bid value (-1) at game end."));
        raw_scores[p] = -999;
        continue;
    }

    if (tricks_won_[p] == bid_values_[p]) { 
      if (bid_values_[p] == 0) { 
        if (high_low_round_ == HighLowRound::kUndecided) SpielFatalError("Game ended with undecided round type for scoring 0 bid.");
        raw_scores[p] = (high_low_round_ == HighLowRound::kHigh) ? 5.0 : 7.0;
      } else { 
        raw_scores[p] = static_cast<double>(bid_values_[p]);
      }
    } else { 
      int diff_m = std::abs(tricks_won_[p] - bid_values_[p]);
      double base_penalty_score;

      if (high_low_round_ == HighLowRound::kLow) {
        base_penalty_score = (tricks_won_[p] < bid_values_[p]) ? (-1.0 * diff_m) : (-2.0 * diff_m);
      } else if (high_low_round_ == HighLowRound::kHigh) {
        base_penalty_score = (tricks_won_[p] > bid_values_[p]) ? (-1.0 * diff_m) : (-2.0 * diff_m);
      } else { 
          SpielFatalError("Scoring attempted with undecided round type for player a miss.");
          base_penalty_score = -999; 
      }

      if (bid_values_[p] == 0) { 
        SPIEL_CHECK_NE(tricks_won_[p], 0); 
        if (high_low_round_ == HighLowRound::kLow) {
            base_penalty_score -= 3.0;
        } else { 
            base_penalty_score -= 4.0;
        }
      }
      raw_scores[p] = base_penalty_score;
    }
  }

  // Step 2: Calculate zero-sum adjusted scores
  std::vector<double> adjusted_returns(num_players_);
  double sum_of_all_raw_scores = 0.0;
  for (int i = 0; i < num_players_; ++i) {
    sum_of_all_raw_scores += raw_scores[i];
  }

  double divider = static_cast<double>(num_players_ - 1.0);

  for (int i = 0; i < num_players_; ++i) {
    double average_of_others_raw_scores = (sum_of_all_raw_scores - raw_scores[i]) / divider;
    adjusted_returns[i] = raw_scores[i] - average_of_others_raw_scores;
  }
  return adjusted_returns;
}

std::string TrumpState::InformationStateString(Player player) const {
  // TODO: This function has not been updated with the infostate tensor function
  // having undergone many changes. Expect it to have gone a bit out of sync.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string rv = absl::StrCat("Player: P", player, "\n");
  absl::StrAppend(&rv, "Phase: ", PhaseToString(), "\n");
  
  absl::StrAppend(&rv, "Hand: ");
  std::vector<std::string> card_names_str_hand;
  std::vector<int> sorted_hand_display_info = hands_[player];
  std::sort(sorted_hand_display_info.begin(), sorted_hand_display_info.end());
  for (int c : sorted_hand_display_info) {
    card_names_str_hand.push_back(CardName(c));
  }
  absl::StrAppend(&rv, absl::StrJoin(card_names_str_hand, " "), "\n");

  if (phase_ == Phase::kBidding) {
    absl::StrAppend(&rv, "Bid Cards Status: (All bids are hidden until simultaneously revealed after this phase)\n");
  } else if (phase_ > Phase::kBidding && phase_ != Phase::kDeal) { 
    absl::StrAppend(&rv, "Revealed Bid Cards: ");
    for (int p_other = 0; p_other < num_players_; ++p_other) {
      absl::StrAppend(&rv, "P", p_other, ":", CardName(bids_card_index_[p_other]), " ");
    }
    absl::StrAppend(&rv, "\n");
  }

  std::string round_bid_status_str_info;
  switch(round_bid_status_){
      case -2: round_bid_status_str_info = "Descend Occurred"; break;
      case -1: round_bid_status_str_info = "Normal Low Round (Sum < 13)"; break;
      case 0: round_bid_status_str_info = "Bidding Phase / Undetermined"; break;
      case 1: round_bid_status_str_info = "Normal High Round (Sum > 13)"; break;
      case 2: round_bid_status_str_info = "Ascend Occurred"; break;
      default: round_bid_status_str_info = "Invalid Status";
  }
  if (phase_ > Phase::kBidding) { 
      absl::StrAppend(&rv, "Round Determination: ", round_bid_status_str_info, "\n");
  }

  if (phase_ >= Phase::kHighLowDecision || (phase_ == Phase::kPlay && trump_suit_ != Suit::kInvalidSuit)) { 
    absl::StrAppend(&rv, "Trump suit: ", 
                    (trump_suit_ == Suit::kInvalidSuit) ? "Undecided" : SuitName(trump_suit_), "\n");
     if(highest_bidder_ != kInvalidPlayer && phase_ > Phase::kBidding) 
        absl::StrAppend(&rv, "Highest bidder (determined trump): P", highest_bidder_, "\n");
  }

  if (phase_ == Phase::kPlay || phase_ == Phase::kGameOver) {
    if (!tricks_history_.empty()) {
        absl::StrAppend(&rv, "Previous Tricks (Most Recent First):\n");
        for (int t_idx = tricks_history_.size() - 1; t_idx >= 0; --t_idx) {
            const auto& trick = tricks_history_[t_idx];
            absl::StrAppend(&rv, "  Hist.Trick ", t_idx + 1, " (Lead P", trick.leader, "): ");
            for (int i = 0; i < num_players_; ++i) { 
                Player p_in_actual_play_order = (trick.leader + i) % num_players_; 
                absl::StrAppend(&rv, "P", p_in_actual_play_order, ":", CardName(trick.cards[p_in_actual_play_order]), " ");
            }
            absl::StrAppend(&rv, "-> Won by P", trick.winner, "\n");
        }
    }

    if (current_trick_starter_ != kInvalidPlayer) { 
        absl::StrAppend(&rv, "Current Trick #", trick_number_ + 1, " (Lead P", current_trick_starter_, "): ");
        for (int i = 0; i < num_players_; ++i) {
            Player p_in_play_order = (current_trick_starter_ + i) % num_players_; 
            int card_played = current_trick_cards_[p_in_play_order]; 
            
            absl::StrAppend(&rv, "P", p_in_play_order, ":"); 
            if (card_played != kInvalidCard) {
                bool is_trump = CardSuit(card_played) == trump_suit_;
                if (is_trump && !IsCurrentTrickComplete() && p_in_play_order != player) { 
                    absl::StrAppend(&rv, "(Tr?) ");
                } else { 
                    absl::StrAppend(&rv, CardName(card_played), " ");
                }
            } else {
                 absl::StrAppend(&rv, "(-) "); 
            }
        }
        absl::StrAppend(&rv, "\n");
    }
    absl::StrAppend(&rv, "Trump break occurred (from previous tricks): ", trump_break_occurred_ ? "Yes" : "No", "\n");
    absl::StrAppend(&rv, "Tricks won by each player: ");
    for (int p_other = 0; p_other < num_players_; ++p_other) { 
      absl::StrAppend(&rv, "P", p_other, ":", tricks_won_[p_other], " ");
    }
    absl::StrAppend(&rv, "\n");
  }
  return rv;
}

std::string TrumpState::ObservationString(Player player) const {
  return InformationStateString(player); 
}

void TrumpState::ObservationTensor(Player player, absl::Span<float> values) const {
  InformationStateTensor(player, values);
}

void TrumpState::InformationStateTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.f);

  int offset = 0;

  // 1. Set of cards currently held in one's hand (multi-hot 52-dim)
  // MOVED TO STEP 4 OF #6 FOR MAXIMUM EFFICIENCY
  // for (int card_idx : hands_[player]) { 
  //   values[offset + card_idx] = 1.0;
  // }
  offset += kNumCards; 

  // 2. Bidding card of each player (4 players * 5 features/card = 20 dim)
  if (phase_ > Phase::kBidding && phase_ != Phase::kDeal) {
    for (int i = 0; i < num_players_; ++i) {
        Player actual_player = (player + i) % num_players_; 
        EncodeCardToFeatureVector(bids_card_index_[actual_player], 
                                  values.subspan(offset + i * kFeaturesPerCard, kFeaturesPerCard));
    }
  }
  offset += num_players_ * kFeaturesPerCard; 

  // 3. Trump suit (1-hot 4-dim)
  if (phase_ > Phase::kBidding && trump_suit_ != Suit::kInvalidSuit) {
    values[offset + static_cast<int>(trump_suit_)] = 1.0;
  }
  offset += kNumSuits; 

  // 4. Ascend/Descend/Normal/Bidding Status (1-dim: -2 to 2)
  values[offset] = static_cast<float>(round_bid_status_);
  offset += 1; 

  // 5. History of previous tricks (13 tricks * (4 for leader + 4 players/trick * 5 features/card) = 312 dim)
  for (int i = 0; i < kNumTricks; ++i) { 
    int history_vector_idx = tricks_history_.size() - 1 - i; 
    int base_trick_slot_offset = offset + i * (num_players_ + num_players_ * kFeaturesPerCard);

    if (history_vector_idx >= 0) {
      const auto& trick_record = tricks_history_[history_vector_idx];

      // A. Encode trick leader (player-centric one-hot)
      Player trick_leader_absolute = trick_record.leader;
      for (int j = 0; j < num_players_; ++j) {
        if (trick_leader_absolute == (player + j) % num_players_) {
          values[base_trick_slot_offset + j] = 1.0f;
          break;
        }
      }
      int card_data_start_in_trick_block = base_trick_slot_offset + num_players_;

      // B. Encode cards played in the trick (player-centric order)
      for (int j = 0; j < num_players_; ++j) { 
        Player actual_player_id_for_slot = (player + j) % num_players_; 
        int card_played = trick_record.cards[actual_player_id_for_slot]; 
        
        EncodeCardToFeatureVector(card_played, 
                                  values.subspan(card_data_start_in_trick_block + j * kFeaturesPerCard, 
                                                 kFeaturesPerCard));
      }
    } else {
      // for (int j = 0; j < num_players_; ++j) {
      //     EncodeCardToFeatureVector(kInvalidCard,
      //                               values.subspan(base_trick_slot_offset + j * kFeaturesPerCard,
      //                                              kFeaturesPerCard));
      // }
      break;
    }
  }
  offset += kNumTricks * (num_players_ + num_players_ * kFeaturesPerCard);

  // 6. Enhanced Graveyard (3 opponents * 52 cards = 156 dim)
  // Step 1: Mark bidding cards
  for (int i = 0; i < num_players_-1; ++i) {
    Player opponent_absolute_id = (player + 1 + i) % num_players_;
    int bid_card_i = bids_card_index_[opponent_absolute_id];
    if (bid_card_i != kInvalidCard) {
      values[offset + i * kNumCards + bid_card_i] = -1.0f;
      for (int j = 0; j < num_players_-1; ++j) {
        if (j != i) {
          values[offset + j * kNumCards + bid_card_i] = 2.0f;
        }
      }
    }
  }

  // Step 2: Mark suits not followed
  for (const auto& trick : tricks_history_) {
    Suit leading_suit = CardSuit(trick.cards[trick.leader]);
    if (trick.cards[trick.leader] == kInvalidCard) continue;
    
    for (int i = 0; i < num_players_-1; ++i) {
      Player opponent_absolute_id = (player + 1 + i) % num_players_;
      int card_played = trick.cards[opponent_absolute_id];
      
      if (card_played != kInvalidCard && CardSuit(card_played) != leading_suit) {
        // This opponent doesn't have cards of the leading suit
        int graveyard_start = offset + i * kNumCards;
        // for (int card_rank = 0; card_rank < kNumCardsPerSuit; ++card_rank) {
        //   int card = CardIndex(leading_suit, card_rank + 1);
        //   values[graveyard_start + card] = 2.0f;
        // }    // REPLACED BY FILLING BELOW, FOR MAXIMUM EFFICIENCY
        int first_card_of_leading_suit = CardIndex(leading_suit, 1);
        auto fill_start_iter = values.begin() + graveyard_start + first_card_of_leading_suit;
        std::fill(fill_start_iter, fill_start_iter + kNumCardsPerSuit, 2.0f);
      }
    }
  }

  // Step 3: If two opponents have 2 for a card, set -1 for the third
  for (int c = 0; c < kNumCards; ++c) {
    int count_2 = 0;
    std::vector<int> indices_with_2;
    for (int i = 0; i < num_players_-1; ++i) {
      if (values[offset + i * kNumCards + c] == 2.0) {
        count_2++;
        indices_with_2.push_back(i);
      }
    }
    if (count_2 == 2) {
      int sum = indices_with_2[0] + indices_with_2[1];
      int remaining_i = num_players_-1 - sum;
      values[offset + remaining_i * kNumCards + c] = -1.0;
    }
  }

  // Step 4: Mark your hand cards with 2 in all graveyards
  for (int card_idx : hands_[player]) {
    values[card_idx] = 1.0;  // MOVED FROM #1, WHERE offset=0

    for (int i = 0; i < num_players_-1; ++i) {
      values[offset + i * kNumCards + card_idx] = 2.0;
    }
  }

  // Step 5: Mark played cards from trick history
  for (const auto& trick : tricks_history_) {
    for (int p = 0; p < num_players_; ++p) {
      int card_played = trick.cards[p];
      if (card_played != kInvalidCard) {
        if (p == player) {
          // Player's card - mark as 2 in all graveyards
          for (int i = 0; i < num_players_-1; ++i) {
            values[offset + i * kNumCards + card_played] = 2.0f;
          }
        } else {
          // Opponent's card - mark as 1 in their graveyard, 2 in others
          for (int i = 0; i < num_players_-1; ++i) {
            Player opponent_id = (player + 1 + i) % num_players_;
            if (opponent_id == p) {
              values[offset + i * kNumCards + card_played] = 1.0f;
            } else {
              values[offset + i * kNumCards + card_played] = 2.0f;
            }
          }
        }
      }
    }
  }

  offset += (num_players_-1) * kNumCards; 

  // 7. Adjusted Number of Tricks to Collect (ANTC) for each player (4-dim)
  for (int i = 0; i < num_players_; ++i) {
    Player p_score_actual = (player + i) % num_players_; 
    
    float antc_p = 0.0f;
    // ANTC is meaningful primarily during the Play phase,
    // once bids and round type are fully determined.
    if (phase_ >= Phase::kPlay &&
       bid_values_[p_score_actual] != -1 && 
        high_low_round_ != HighLowRound::kUndecided) {
        
        int final_bid = bid_values_[p_score_actual];
        int tricks_won = tricks_won_[p_score_actual];
        int difference = final_bid - tricks_won; // How many more tricks needed to meet bid
        
        if (difference > 0) { // Still needs to win tricks
            if (high_low_round_ == HighLowRound::kHigh) {
                antc_p = static_cast<float>(difference * 2);
            } else { // Low round
                antc_p = static_cast<float>(difference);
            }
       } else if (difference < 0) { // Has won more tricks than bid
            if (high_low_round_ == HighLowRound::kLow) {
                antc_p = static_cast<float>(difference * 2);
            } else { // High round
                antc_p = static_cast<float>(difference);
            }
        } else { // difference == 0 (exactly met bid so far)
           antc_p = 0.0f;
        }
    } else { 
          antc_p = 0.0f; 
    }
    values[offset + i] = antc_p;
  }
  offset += num_players_; 

  // 8. Boolean: break has occurred in a previous trick (1-dim)
  values[offset] = trump_break_occurred_ ? 1.0f : 0.0f;
  offset += 1; 

  // 9. Current running trick (4 players * 5 features/card = 20 dim)
  for (int i = 0; i < num_players_; ++i) { 
    Player actual_player = (player + i) % num_players_; 
    int card_played = current_trick_cards_[actual_player]; 
    
    int rank_override = -1;
    if (card_played != kInvalidCard && 
        actual_player != player &&  
        CardSuit(card_played) == trump_suit_
        // && !IsCurrentTrickComplete()
       ) {
        rank_override = kHiddenTrumpRankEncoding; 
    }
    
    EncodeCardToFeatureVector(card_played, 
                              values.subspan(offset + i * kFeaturesPerCard, kFeaturesPerCard),
                              rank_override);
  }
  offset += num_players_ * kFeaturesPerCard; 

  // 10. Who played the leading card (current_trick_starter_) (1-hot 4-dim, player-centric)
  if (phase_ == Phase::kPlay && current_trick_starter_ != kInvalidPlayer) {
    for (int i = 0; i < num_players_; ++i) { 
        if (current_trick_starter_ == (player + i) % num_players_) {
            values[offset + i] = 1.0f; 
            break;
        }
    }
  }
  offset += num_players_;

  // 11. Trump uncertainty in current trick (13 trump ranks)
  std::fill(values.begin() + offset, values.begin() + offset + kNumCardsPerSuit, -1.0f);

  if (phase_ == Phase::kPlay && trump_suit_ != Suit::kInvalidSuit && 
      current_trick_starter_ != kInvalidPlayer) {
    
    // Check if any trump cards are being played in current trick
    bool trump_in_current_trick = false;
    std::set<int> trump_ranks_in_trick;
    
    for (int p = 0; p < num_players_; ++p) {
      int card = current_trick_cards_[p];
      if (card != kInvalidCard && CardSuit(card) == trump_suit_) {
        trump_in_current_trick = true;
        if (p == player || IsCurrentTrickComplete()) {
          trump_ranks_in_trick.insert(CardRank(card));
        }
      }
    }
    
    if (trump_in_current_trick) {
      // Count unknown trump cards
      std::set<int> unknown_trump_ranks;
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        int trump_card = rank * kNumSuits + static_cast<int>(trump_suit_);
        bool is_unknown = true;
        
        // Check if in player's hand (reuse hand encoding)
        if (values[trump_card] == 1.0f) {
          is_unknown = false;
        }
        
        // Check if player is playing this card
        if (current_trick_cards_[player] == trump_card) {
          is_unknown = false;
        }
        
        // Check if in previous tricks
        if (is_unknown) {
          for (const auto& trick : tricks_history_) {
            for (int p = 0; p < num_players_; ++p) {
              if (trick.cards[p] == trump_card) {
                is_unknown = false;
                break;
              }
            }
            if (!is_unknown) break;
          }
        }
        
        if (is_unknown) {
          unknown_trump_ranks.insert(rank);
        }
      }
      
      // Count opponents' trump cards in current trick
      int opponent_trump_count = 0;
      for (int p = 0; p < num_players_; ++p) {
        if (p != player && current_trick_cards_[p] != kInvalidCard && 
            CardSuit(current_trick_cards_[p]) == trump_suit_) {
          opponent_trump_count++;
        }
      }
      
      if (opponent_trump_count < unknown_trump_ranks.size()) {
        // More unknown trumps than being played
        // Known trumps and player's trump: -1, others: 0, player's current: 1
        for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
          if (unknown_trump_ranks.count(rank)) {
            values[offset + rank] = 0.0f; // Unknown
          }
          // else stays -1 (known/played/in hand)
        }
      } else {
        // Can deduce which trumps are being played
        for (int rank : unknown_trump_ranks) {
          values[offset + rank] = 1.0f; // Being played
        }
        // Others stay -1
      }
      
      // Mark player's trump card as 1 if playing one
      if (current_trick_cards_[player] != kInvalidCard && 
          CardSuit(current_trick_cards_[player]) == trump_suit_) {
        int player_trump_rank = CardRank(current_trick_cards_[player]);
        values[offset + player_trump_rank] = 1.0f;
      }
    }
  }
  offset += kNumCardsPerSuit;

  // 12. Current trick number (1-13), 0 in bidding phase (1-dim)
  if (phase_ >= Phase::kPlay) {
      values[offset] = static_cast<float>(trick_number_ + 1); 
  } else {
      values[offset] = 0.0f; 
  }
  offset += 1; 

  SPIEL_CHECK_EQ(offset, values.size());
}


// Definition of IsChanceNode relies on base State::IsChanceNode() if not overridden in trump.h
// Definition of ResampleFromInfostate relies on base State::ResampleFromInfostate() if not overridden in trump.h

void TrumpState::DetermineBidWinnerAndTrump() {
  int total_bid_sum = 0;
  int max_bid_value_for_tiebreak = -1;
  
  for (int p = 0; p < num_players_; ++p) {
    SPIEL_CHECK_NE(bids_card_index_[p], kInvalidCard);
    bid_values_[p] = BiddingValue(bids_card_index_[p]);
    total_bid_sum += bid_values_[p];
    if (bid_values_[p] > max_bid_value_for_tiebreak) {
      max_bid_value_for_tiebreak = bid_values_[p];
    }
  }

  std::vector<Player> potential_highest_bidders;
  for (int p = 0; p < num_players_; ++p) {
    if (bid_values_[p] == max_bid_value_for_tiebreak) {
      potential_highest_bidders.push_back(p);
    }
  }
  
  SPIEL_CHECK_FALSE(potential_highest_bidders.empty());

  if (potential_highest_bidders.size() == 1) {
    highest_bidder_ = potential_highest_bidders[0];
  } else { 
    Player current_strongest_player = kInvalidPlayer;
    Suit current_strongest_suit = Suit::kInvalidSuit; 
    int current_strongest_suit_ = static_cast<int>(current_strongest_suit);

    for (Player p : potential_highest_bidders) {
      Suit p_suit = CardSuit(bids_card_index_[p]);
      int p_suit_ = static_cast<int>(p_suit);

      if (p_suit_ > current_strongest_suit_) {
          current_strongest_player = p;
          current_strongest_suit_ = p_suit_;
      }
    }
    highest_bidder_ = current_strongest_player;
  }
  current_player_ = highest_bidder_;
  current_trick_starter_ = highest_bidder_;
  trump_suit_ = CardSuit(bids_card_index_[highest_bidder_]);

  if (total_bid_sum > 13) {
    high_low_round_ = HighLowRound::kHigh; 
    round_bid_status_ = 1; 
    phase_ = Phase::kPlay;
  } else if (total_bid_sum < 13) {
    high_low_round_ = HighLowRound::kLow; 
    round_bid_status_ = -1; 
    phase_ = Phase::kPlay;
  } else { 
    phase_ = Phase::kHighLowDecision;
  }
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
      // Deal one card of each suit to current player
      hands_[dealing_player_].push_back(action);
      is_undealt_[action] = false;
      suit_index_++;
      
      if (suit_index_ == kNumSuits) {
        dealing_subphase_ = DealingSubPhase::kSetAside;
        suit_index_ = 0;
        cards_dealt_in_subphase_ = 0; // Reset counter for set-aside phase
      }
    } 
    else if (dealing_subphase_ == DealingSubPhase::kSetAside) {
      // Set aside cards - we need multiple sets of 4 cards
      is_set_aside_[action] = true;
      is_undealt_[action] = false; // Mark as temporarily unavailable
      cards_dealt_in_subphase_++;
      
      // Check if we've completed a set of 4 cards (one per suit)
      if (cards_dealt_in_subphase_ % kNumSuits == 0) {
        int completed_sets = cards_dealt_in_subphase_ / kNumSuits;
        if (completed_sets >= sets_to_set_aside_) {
          // Done setting aside cards for this player
          dealing_subphase_ = DealingSubPhase::kDealRemaining;
          cards_dealt_in_subphase_ = 0;
        }
        // Reset suit_index for next set or move to dealing remaining
        suit_index_ = 0;
      } else {
        suit_index_++;
      }
    } 
    else if (dealing_subphase_ == DealingSubPhase::kDealRemaining) {
      // Deal remaining 9 cards to current player (excluding set-aside cards)
      hands_[dealing_player_].push_back(action);
      is_undealt_[action] = false;
      cards_dealt_in_subphase_++;
      
      if (cards_dealt_in_subphase_ == 9) {
        // Restore set-aside cards to available pool
        for (int c = 0; c < kNumCards; ++c) {
          if (is_set_aside_[c]) {
            is_set_aside_[c] = false;
            is_undealt_[c] = true; // Make available again
          }
        }
        
        // Move to next player or finish dealing
        if (dealing_player_ == 2) {
          // Player 3 gets all remaining cards
          for (int c = 0; c < kNumCards; ++c) {
            if (is_undealt_[c]) {
              hands_[3].push_back(c);
              is_undealt_[c] = false;
            }
          }
          phase_ = Phase::kBidding;
          current_player_ = 0;
        } else {
          // Move to next player
          dealing_player_++;
          dealing_subphase_ = DealingSubPhase::kDealPerSuit;
          suit_index_ = 0;
          cards_dealt_in_subphase_ = 0;
          
          // Set the number of sets to set aside for the next player
          if (dealing_player_ == 1) {
            sets_to_set_aside_ = 2; // Player 1 needs 2 sets
          } else if (dealing_player_ == 2) {
            sets_to_set_aside_ = 1; // Player 2 needs 1 set
          }
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

int TrumpState::ResolveTrick() {
  Suit leading_suit_of_trick = CardSuit(current_trick_cards_[current_trick_starter_]); 
  
  Player trick_winner = current_trick_starter_;
  int winning_card_in_trick = current_trick_cards_[current_trick_starter_];
  
  for (int i = 0; i < num_players_; ++i) { 
    Player player_in_sequence = (current_trick_starter_ + i) % num_players_;
    int card_played = current_trick_cards_[player_in_sequence];

    if (CardSuit(card_played) == trump_suit_) {
        this->trump_break_occurred_ = true; 
    } else if (player_in_sequence != current_trick_starter_ && CardSuit(card_played) != leading_suit_of_trick) {
        this->trump_break_occurred_ = true;
    }

    if (player_in_sequence != current_trick_starter_) { 
        Suit suit_of_card_played = CardSuit(card_played);
        bool current_card_beats_winning_card = false;

        if (suit_of_card_played == trump_suit_) { 
          if (CardSuit(winning_card_in_trick) == trump_suit_) { 
            if (IsCardStronger(card_played, winning_card_in_trick)) {
              current_card_beats_winning_card = true;
            }
          } else { 
            current_card_beats_winning_card = true;
          }
        } else { 
          if (CardSuit(winning_card_in_trick) != trump_suit_ &&  
              suit_of_card_played == leading_suit_of_trick) { 
            if (IsCardStronger(card_played, winning_card_in_trick)) {
              current_card_beats_winning_card = true;
            }
          }
        }
        
        if (current_card_beats_winning_card) {
          trick_winner = player_in_sequence;
          winning_card_in_trick = card_played;
        }
    }
  }
  
  return trick_winner;
}

bool TrumpState::IsCardStronger(int card1_idx, int card2_idx) const {
  int rank1 = CardRank(card1_idx); 
  int rank2 = CardRank(card2_idx);
  
  int strength1 = kRankStrengthIndices[rank1];
  int strength2 = kRankStrengthIndices[rank2];
  return strength1 > strength2;
}

bool TrumpState::IsCurrentTrickEmpty() const {
  return current_trick_card_count_ == 0;
}

bool TrumpState::IsCurrentTrickComplete() const {
  return current_trick_card_count_ == num_players_;
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


}  // namespace trump
}  // namespace open_spiel