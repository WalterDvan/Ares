/***************************************************************************
 *  bhv_unmark_move.h - Unmark Move Behavior (CYRUS-style)
 *
 *  Based on CYRUS 2021 RoboCup Champion team description:
 *  - Unmark Decisioning: Build a pass prediction tree (BFS) to find
 *    the future passer who will eventually pass to self.
 *  - Unmark Positioning: Generate 24 candidate positions (3-layer ring,
 *    radii 2/4/8m), select the one where self can arrive before opponents
 *    and is closest to the opponent's goal line.
 *
 *  Since we lack trained DNN data, we use heuristic pass prediction
 *  (distance, angle, forward preference, pass lane occlusion, openness,
 *  distance to opponent goal, offside check) instead of neural network.
 *
 *  Expected improvement: +6.6% win rate (as reported by CYRUS)
 ****************************************************************************/

#ifndef BHV_UNMARK_MOVE_H
#define BHV_UNMARK_MOVE_H

#include <rcsc/player/player_agent.h>
#include <rcsc/geom/vector_2d.h>
#include <rcsc/geom/angle_deg.h>

class Bhv_UnmarkMove {
public:
    /*!
     * \brief Main entry point. Attempts unmark positioning.
     * \return true if unmark positioning was executed, false if fallback needed
     */
    static bool execute( rcsc::PlayerAgent * agent );

private:
    /*!
     * \brief Check if the player should attempt unmark move.
     */
    static bool shouldAttemptUnmark( const rcsc::PlayerAgent * agent );

    /*!
     * \brief Heuristic pass prediction - score how likely passer will pass to receiver.
     * \param agent The agent (for world model access)
     * \param passer_pos Position of the passer
     * \param receiver_pos Position of the potential receiver
     * \param receiver_unum Uniform number of the potential receiver
     * \return Score between 0.0 and 1.0
     */
    static double predictPassScore( const rcsc::PlayerAgent * agent,
                                    const rcsc::Vector2D & passer_pos,
                                    const rcsc::Vector2D & receiver_pos,
                                    int receiver_unum );

    /*!
     * \brief Build the pass prediction tree using BFS.
     * Finds the teammate who is most likely to eventually pass to self.
     * \param agent The agent
     * \param current_holder_unum Uniform number of current ball holder
     * \param[out] best_passer_pos Output: position of the best future passer
     * \param[out] confidence Output: confidence score of the prediction (0-1)
     * \return true if a valid future passer was found
     */
    static bool buildPassPredictionTree( const rcsc::PlayerAgent * agent,
                                         int current_holder_unum,
                                         rcsc::Vector2D & best_passer_pos,
                                         double & confidence );

    /*!
     * \brief Calculate the best unmark position for the agent.
     * Generates 24 candidate positions (3-layer ring: 2/4/8m radii, 8 angles).
     * Selects the best based on: earliest arrival time and closeness to opponent goal.
     * \param agent The agent
     * \param passer_pos Position of the expected future passer
     * \param[out] target_pos Output: the best position to move to
     * \return true if a valid position was found
     */
    static bool calculateUnmarkPosition( const rcsc::PlayerAgent * agent,
                                          const rcsc::Vector2D & passer_pos,
                                          rcsc::Vector2D & target_pos );

    /*!
     * \brief Calculate a player's time to reach a point (simplified: cycles).
     * Uses stamina-aware dash estimation.
     */
    static int cyclesToReach( const rcsc::PlayerAgent * agent,
                              const rcsc::Vector2D & start_pos,
                              const rcsc::Vector2D & target_pos,
                              double player_speed_max );

    /*!
     * \brief Check if pass lane is relatively clear of opponents.
     * \param agent The agent
     * \param from Pass origin
     * \param to Pass destination
     * \return score 0.0-1.0, 1.0 = completely clear
     */
    static double checkPassLaneClearance( const rcsc::PlayerAgent * agent,
                                           const rcsc::Vector2D & from,
                                           const rcsc::Vector2D & to );

    /*!
     * \brief Check if a point is offside.
     */
    static bool isOffside( const rcsc::PlayerAgent * agent,
                           const rcsc::Vector2D & pos );

    /*!
     * \brief Calculate how open a receiver is (distance to nearest opponent).
     * \return score 0.0-1.0, 1.0 = very open
     */
    static double calculateOpenness( const rcsc::PlayerAgent * agent,
                                      const rcsc::Vector2D & pos );

    // Constants
    static const int MAX_TREE_DEPTH;
    static const int TOP_N_CANDIDATES;
    static const double MIN_CONFIDENCE_THRESHOLD;
    static const double OFFSIDE_MARGIN;
    static const double LANE_BASE_WIDTH;

    // Candidate position ring parameters (from CYRUS paper: 3 layers)
    static const double RING_RADII[3];
    static const int RING_ANGLES;
};

#endif // BHV_UNMARK_MOVE_H
