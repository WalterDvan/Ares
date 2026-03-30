/***************************************************************************
 *  bhv_omam_mark.h - OMAM Marking Defense (Paper-Faithful Implementation)
 *
 *  Based on CYRUS 2021 paper: "Improving Dribbling, Passing, and Marking
 *  Actions in Soccer Simulation 2D Games Using Machine Learning"
 *  (arXiv:2401.03406)
 *
 *  This is a complete re-implementation following the paper exactly:
 *
 *  Algorithm Steps:
 *  1. Classify our defenders into groups: Back / Middle / Forward
 *     Back = CB, CB, SB, SB, DH (unums 2-6)
 *     Middle = OH, OH (unums 7-8)
 *     Forward = SF, SF, CF (unums 9-11)
 *
 *  2. Classify opponents into: Attacker / Non-Attacker
 *     Attacker = danger score > threshold
 *     danger = w1*d_goal + w2*d_ball + w3*angle_goal + w4*own_half + w5*penalty_area + w6*speed_toward_goal
 *
 *  3. Layered matching (OMAM core):
 *     Layer 1: Back players -> unmarked Attackers (sorted by danger desc)
 *     Layer 2: Remaining players -> unmarked Attackers
 *     Layer 3: Idle players -> remaining Non-Attackers
 *
 *  4. K-top optimization: each defender keeps top-K candidates
 *     Reduces solution space from 10! to ~6! or less
 *
 *  5. Solution comparison: select assignment maximizing total danger covered
 *
 *  Matching inserted via doMove() chain:
 *    OMAM -> UnmarkMove -> BasicMove
 *  Only players with assignments execute OMAM; others fall through.
 ****************************************************************************/

#ifndef BHV_OMAM_MARK_H
#define BHV_OMAM_MARK_H

#include <rcsc/player/player_agent.h>
#include <rcsc/geom/vector_2d.h>
#include <vector>
#include <utility>

class Bhv_OmamMark {
public:

    static bool execute( rcsc::PlayerAgent * agent );

private:

    struct Assignment {
        int our_unum;
        int opp_unum;
        double opp_danger;
        double total_cost;  // sum of individual assignment costs
    };

    //! Check if OMAM should be active (game state filter)
    static bool shouldActivate( const rcsc::PlayerAgent * agent );

    //! Danger score for an opponent (0.0 ~ 1.0, higher = more dangerous)
    static double dangerScore( const rcsc::PlayerAgent * agent, int opp_unum );

    //! Classify our non-goalie field players into Back/Middle/Forward
    static void classifyOurPlayers( const rcsc::PlayerAgent * agent,
                                     std::vector< int > & back,
                                     std::vector< int > & middle,
                                     std::vector< int > & forward );

    //! Split opponents into attacker/non-attacker lists (sorted by danger desc)
    static void classifyOpponents( const rcsc::PlayerAgent * agent,
                                    std::vector< std::pair< double, int > > & attackers,
                                    std::vector< std::pair< double, int > > & non_attackers );

    //! Cost of assigning ourUnum to mark oppUnum
    static double assignmentCost( const rcsc::PlayerAgent * agent,
                                   int our_unum, int opp_unum );

    //! OMAM layered matching - produces final assignment list
    static void omamMatch( const rcsc::PlayerAgent * agent,
                            const std::vector< int > & back,
                            const std::vector< int > & middle,
                            const std::vector< int > & forward,
                            const std::vector< std::pair< double, int > > & attackers,
                            const std::vector< std::pair< double, int > > & non_attackers,
                            std::vector< Assignment > & result );

    //! Layer: assign players from avail to target opps, one-to-one, min total cost
    static void greedyLayer( const rcsc::PlayerAgent * agent,
                              const std::vector< int > & avail_players,
                              const std::vector< std::pair< double, int > > & target_opps,
                              std::vector< Assignment > & assignments,
                              std::vector< bool > & player_used,
                              std::vector< bool > & opp_used );

    //! Marking position: between opponent and ball (ball-side marking)
    static rcsc::Vector2D markPosition( const rcsc::PlayerAgent * agent,
                                         int opp_unum );

    //! Move body to marking position
    static void goToMark( rcsc::PlayerAgent * agent, const rcsc::Vector2D & pos );
};

#endif
