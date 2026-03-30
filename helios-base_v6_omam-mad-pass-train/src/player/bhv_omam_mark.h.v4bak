/***************************************************************************
 *  bhv_omam_mark.h - OMAM (Optimized Multi-Agent Matching) Marking Defense
 *
 *  Based on CYRUS 2021 paper: "Improving Dribbling, Passing, and Marking
 *  Actions in Soccer Simulation 2D Games Using Machine Learning"
 *  (arXiv:2401.03406)
 *
 *  Algorithm Overview:
 *  1. Classify our defensive players into Back / Middle / Forward groups
 *  2. Classify opponents into Attacker / Normal based on danger level
 *  3. Layer matching:
 *     - Layer 1: Back players -> most dangerous Attackers
 *     - Layer 2: Middle players -> unmarked Attackers
 *     - Layer 3: Idle players -> remaining opponents
 *  4. K-top optimization: each player keeps only top-3 candidate assignments
 *  5. Solution comparison: prefer covering higher-threat opponents
 *
 *  Expected improvement: Goals conceded reduced by 63% (0.9 -> 0.33 avg)
 ****************************************************************************/

#ifndef BHV_OMAM_MARK_H
#define BHV_OMAM_MARK_H

#include <rcsc/player/player_agent.h>
#include <rcsc/player/player_object.h>
#include <rcsc/geom/vector_2d.h>
#include <vector>
#include <utility>

class Bhv_OmamMark {
public:

    /*!
     * \brief Main entry point for OMAM marking defense.
     * Called in doMove() of defensive roles to override formation-based
     * positioning with intelligent opponent marking.
     * \param agent The player agent
     * \return true if OMAM marking position was executed, false if fallback
     */
    static bool execute( rcsc::PlayerAgent * agent );

private:

    /*!
     * \brief Check if conditions are suitable for OMAM marking.
     * Only active when opponent has the ball or ball is in our half.
     */
    static bool shouldMark( const rcsc::PlayerAgent * agent );

    /*!
     * \brief Calculate danger score for an opponent (0-1, higher = more dangerous).
     * Considers: distance to our goal, distance to ball, angle to goal, speed.
     */
    static double calculateDangerScore( const rcsc::PlayerAgent * agent,
                                         const rcsc::AbstractPlayerObject * opponent );

    /*!
     * \brief Classify our players into Back/Middle/Forward groups.
     * Back: defenders + defensive half (unums 2-6 typically)
     * Middle: midfielders (unums 7-8)
     * Forward: forwards (unums 9-11)
     */
    static void classifyOurPlayers( const rcsc::PlayerAgent * agent,
                                     std::vector< int > & back_players,
                                     std::vector< int > & middle_players,
                                     std::vector< int > & forward_players );

    /*!
     * \brief Classify opponents into Attacker/Normal groups.
     * Attacker: danger score > threshold AND in our half
     * Normal: all others
     */
    static void classifyOpponents( const rcsc::PlayerAgent * agent,
                                    std::vector< std::pair< double, int > > & attackers,
                                    std::vector< std::pair< double, int > > & normal_opps );

    /*!
     * \brief Run the OMAM matching algorithm.
     * Returns a vector of (our_unum, target_opponent_unum) pairs.
     */
    static void runOmamMatching( const rcsc::PlayerAgent * agent,
                                  const std::vector< int > & back_players,
                                  const std::vector< int > & middle_players,
                                  const std::vector< int > & forward_players,
                                  const std::vector< std::pair< double, int > > & attackers,
                                  const std::vector< std::pair< double, int > > & normal_opps,
                                  std::vector< std::pair< int, int > > & assignments );

    /*!
     * \brief Calculate marking position for a player.
     * Positions between opponent and our goal, at a controlled distance.
     */
    static rcsc::Vector2D calculateMarkingPosition( const rcsc::PlayerAgent * agent,
                                                     const rcsc::AbstractPlayerObject * target_opponent );

    /*!
     * \brief Simple greedy matching: assign closest available player to opponent.
     * Used as fallback within OMAM when layered matching leaves gaps.
     */
    static void greedyAssign( const rcsc::PlayerAgent * agent,
                               const std::vector< int > & available_players,
                               const std::vector< std::pair< double, int > > & opponents,
                               std::vector< std::pair< int, int > > & assignments,
                               std::vector< bool > & player_used,
                               std::vector< bool > & opp_used );

    /*!
     * \brief Defensive recovery: when shouldMark is true but no assignment,
     * actively recover towards ball/goal instead of falling to BasicMove.
     */
    static void defensiveRecovery( rcsc::PlayerAgent * agent );

    // Constants
    static const double ATTACKER_DANGER_THRESHOLD;  // Min danger to be "Attacker"
    static const double MARKING_DISTANCE;            // Distance to mark (from opponent)
    static const double MARKING_BALL_SIDE_BIAS;      // Bias towards ball side
    static const int MAX_ASSIGNMENT_DIST;             // Max distance for assignment
    static const int K_TOP;                           // K-top candidates per player
};

#endif // BHV_OMAM_MARK_H
