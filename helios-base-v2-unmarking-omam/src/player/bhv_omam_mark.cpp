/***************************************************************************
 *  bhv_omam_mark.cpp - OMAM (Optimized Multi-Agent Matching) Marking Defense
 *
 *  Implementation of CYRUS 2021 OMAM algorithm for intelligent marking.
 *
 *  Key differences from simple proximity-based marking:
 *  - Layered player/opponent classification reduces solution space
 *  - Back defenders prioritize the most dangerous attackers
 *  - K-top optimization limits search combinatorics
 *  - Marking position calculated between opponent and own goal
 ****************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_omam_mark.h"
#include "strategy.h"

#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_turn_to_ball.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/player_object.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include <algorithm>
#include <set>
#include <cmath>

using namespace rcsc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const double Bhv_OmamMark::ATTACKER_DANGER_THRESHOLD = 0.35;
const double Bhv_OmamMark::MARKING_DISTANCE = 1.5;       // meters from opponent
const double Bhv_OmamMark::MARKING_BALL_SIDE_BIAS = 0.3; // bias towards ball side
const int    Bhv_OmamMark::MAX_ASSIGNMENT_DIST = 30;      // max assignment dist (m)
const int    Bhv_OmamMark::K_TOP = 3;                     // top-k candidates

// ---------------------------------------------------------------------------
// Main Entry
// ---------------------------------------------------------------------------

bool
Bhv_OmamMark::execute( PlayerAgent * agent )
{
    if ( ! shouldMark( agent ) )
    {
        return false;
    }

    const WorldModel & wm = agent->world();
    int self_unum = wm.self().unum();

    // Step 1: Classify our players
    std::vector< int > back_players, middle_players, forward_players;
    classifyOurPlayers( agent, back_players, middle_players, forward_players );

    // Step 2: Classify opponents
    std::vector< std::pair< double, int > > attackers;    // (danger, unum)
    std::vector< std::pair< double, int > > normal_opps;   // (danger, unum)
    classifyOpponents( agent, attackers, normal_opps );

    // Sort by danger descending
    std::sort( attackers.begin(), attackers.end(),
               []( const std::pair< double, int > & a,
                   const std::pair< double, int > & b )
               { return a.first > b.first; } );
    std::sort( normal_opps.begin(), normal_opps.end(),
               []( const std::pair< double, int > & a,
                   const std::pair< double, int > & b )
               { return a.first > b.first; } );

    // Step 3: Run OMAM matching
    std::vector< std::pair< int, int > > assignments; // (our_unum, opp_unum)
    runOmamMatching( agent, back_players, middle_players, forward_players,
                     attackers, normal_opps, assignments );

    // Step 4: Find this player's assignment
    const AbstractPlayerObject * target_opponent = nullptr;
    for ( const auto & assign : assignments )
    {
        if ( assign.first == self_unum )
        {
            target_opponent = wm.theirPlayer( assign.second );
            break;
        }
    }

    if ( ! target_opponent )
    {
        return false;
    }

    // Step 5: Calculate and execute marking position
    Vector2D mark_pos = calculateMarkingPosition( agent, target_opponent );

    double dist_thr = wm.ball().distFromSelf() * 0.1;
    if ( dist_thr < 1.0 ) dist_thr = 1.0;

    dlog.addText( Logger::TEAM,
                  __FILE__": OMAM marking opp(%d) at (%.1f, %.1f), mark_pos=(%.1f, %.1f)",
                  target_opponent->unum(),
                  target_opponent->pos().x, target_opponent->pos().y,
                  mark_pos.x, mark_pos.y );

    agent->debugClient().addMessage( "OMAM_Mark%d", target_opponent->unum() );
    agent->debugClient().setTarget( mark_pos );
    agent->debugClient().addCircle( mark_pos, dist_thr );
    agent->debugClient().addLine( wm.self().pos(), mark_pos );

    double dash_power = Strategy::get_normal_dash_power( wm );

    if ( ! Body_GoToPoint( mark_pos, dist_thr, dash_power
                           ).execute( agent ) )
    {
        Body_TurnToBall().execute( agent );
    }

    agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );

    return true;
}

// ---------------------------------------------------------------------------
// Condition Check
// ---------------------------------------------------------------------------

bool
Bhv_OmamMark::shouldMark( const PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    // Don't mark if we have the ball
    if ( wm.self().isKickable() )
    {
        return false;
    }

    // Don't mark during setplay - our set plays use setplay roles
    if ( wm.gameMode().isOurSetPlay( wm.ourSide() ) )
    {
        return false;
    }

    // Mark only if opponent potentially has ball or ball is in contested area
    const int opp_min = wm.interceptTable().opponentStep();
    const int mate_min = wm.interceptTable().teammateStep();
    const int self_min = wm.interceptTable().selfStep();

    // If we can intercept the ball, don't mark - go for it
    if ( self_min <= mate_min && self_min <= opp_min + 3 )
    {
        return false;
    }

    // If teammate has ball, mark when ball is in our half or middle
    if ( mate_min <= opp_min && mate_min <= self_min )
    {
        // Teammate has/taking ball - still useful to mark in our defensive zone
        if ( wm.ball().pos().x > 10.0 )
        {
            return false;
        }
    }

    // Only mark when ball is in certain areas (opponent threat zone)
    double ball_x = wm.ball().pos().x;
    if ( ball_x < -30.0 )
    {
        // Ball very close to our goal - always mark
        return true;
    }
    if ( ball_x > 20.0 )
    {
        // Ball far in opponent's half - no need for tight marking
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Danger Score Calculation
// ---------------------------------------------------------------------------

double
Bhv_OmamMark::calculateDangerScore( const PlayerAgent * agent,
                                     const AbstractPlayerObject * opponent )
{
    if ( ! opponent || opponent->unum() == 0 )
    {
        return 0.0;
    }

    const WorldModel & wm = agent->world();
    const ServerParam & sp = ServerParam::i();

    Vector2D opp_pos = opponent->pos();
    Vector2D ball_pos = wm.ball().pos();
    Vector2D our_goal_pos( -sp.pitchHalfLength(), 0.0 );

    double score = 0.0;

    // Factor 1: Distance to our goal (closer = more dangerous)
    double dist_to_goal = opp_pos.dist( our_goal_pos );
    double max_goal_dist = sp.pitchHalfLength(); // ~52.5
    double goal_dist_score = 1.0 - std::min( dist_to_goal / max_goal_dist, 1.0 );
    score += goal_dist_score * 0.35;

    // Factor 2: Distance to ball (closer = more dangerous - can receive pass)
    double dist_to_ball = opp_pos.dist( ball_pos );
    double ball_dist_score = 1.0 - std::min( dist_to_ball / 40.0, 1.0 );
    score += ball_dist_score * 0.30;

    // Factor 3: Angle to our goal (center = more dangerous)
    AngleDeg angle_to_goal = ( our_goal_pos - opp_pos ).th();
    double angle_score = 1.0 - std::abs( angle_to_goal.degree() ) / 180.0;
    score += angle_score * 0.15;

    // Factor 4: Opponent in our half (bonus)
    if ( opp_pos.x < 0.0 )
    {
        score += 0.10;
    }
    // Opponent in our penalty area
    if ( opp_pos.x < -36.0 && std::abs( opp_pos.y ) < 20.0 )
    {
        score += 0.15;
    }

    // Factor 5: Opponent speed (moving towards our goal = more dangerous)
    Vector2D opp_vel = opponent->vel();
    double speed = opp_vel.length();
    if ( speed > 0.5 )
    {
        Vector2D vel_unit = opp_vel / speed;
        Vector2D to_goal_unit = ( our_goal_pos - opp_pos ).normalizedVector();
        double heading_score = vel_unit.x * to_goal_unit.x
                              + vel_unit.y * to_goal_unit.y;
        if ( heading_score > 0.0 )
        {
            score += heading_score * 0.10;
        }
    }

    return std::min( score, 1.0 );
}

// ---------------------------------------------------------------------------
// Player Classification
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::classifyOurPlayers( const PlayerAgent * agent,
                                    std::vector< int > & back_players,
                                    std::vector< int > & middle_players,
                                    std::vector< int > & forward_players )
{
    const WorldModel & wm = agent->world();

    back_players.clear();
    middle_players.clear();
    forward_players.clear();

    // Classify by role type based on formation
    // Back: CenterBack(2,3), SideBack(4,5), DefensiveHalf(6)
    // Middle: OffensiveHalf(7,8)
    // Forward: SideForward(9,10), CenterForward(11)
    // Goalie(1) excluded

    static const int BACK_UNUMS[] = { 2, 3, 4, 5, 6 };
    static const int MIDDLE_UNUMS[] = { 7, 8 };
    static const int FORWARD_UNUMS[] = { 9, 10, 11 };

    for ( int i = 0; i < 5; ++i )
    {
        int unum = BACK_UNUMS[i];
        const AbstractPlayerObject * p = wm.ourPlayer( unum );
        if ( p && p->unum() != 0 )
        {
            // Don't include self - self is already executing this
            back_players.push_back( unum );
        }
    }

    for ( int i = 0; i < 2; ++i )
    {
        int unum = MIDDLE_UNUMS[i];
        const AbstractPlayerObject * p = wm.ourPlayer( unum );
        if ( p && p->unum() != 0 )
        {
            middle_players.push_back( unum );
        }
    }

    for ( int i = 0; i < 3; ++i )
    {
        int unum = FORWARD_UNUMS[i];
        const AbstractPlayerObject * p = wm.ourPlayer( unum );
        if ( p && p->unum() != 0 )
        {
            forward_players.push_back( unum );
        }
    }
}

// ---------------------------------------------------------------------------
// Opponent Classification
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::classifyOpponents( const PlayerAgent * agent,
                                   std::vector< std::pair< double, int > > & attackers,
                                   std::vector< std::pair< double, int > > & normal_opps )
{
    const WorldModel & wm = agent->world();

    attackers.clear();
    normal_opps.clear();

    const PlayerObject::Cont & opps = wm.opponents();
    for ( PlayerObject::Cont::const_iterator it = opps.begin();
          it != opps.end(); ++it )
    {
        const PlayerObject * opp = *it;
        if ( ! opp || opp->unum() == 0 || opp->goalie() )
        {
            continue;
        }
        if ( opp->posCount() >= 10 )
        {
            continue; // Too old observation
        }

        double danger = calculateDangerScore( agent, opp );

        if ( danger >= ATTACKER_DANGER_THRESHOLD )
        {
            attackers.push_back( std::make_pair( danger, opp->unum() ) );
        }
        else
        {
            normal_opps.push_back( std::make_pair( danger, opp->unum() ) );
        }
    }
}

// ---------------------------------------------------------------------------
// OMAM Matching Algorithm
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::runOmamMatching( const PlayerAgent * agent,
                                  const std::vector< int > & back_players,
                                  const std::vector< int > & middle_players,
                                  const std::vector< int > & forward_players,
                                  const std::vector< std::pair< double, int > > & attackers,
                                  const std::vector< std::pair< double, int > > & normal_opps,
                                  std::vector< std::pair< int, int > > & assignments )
{
    assignments.clear();

    std::vector< bool > player_used( 12, false );  // our player used flag
    std::vector< bool > opp_used( 12, false );      // opponent used flag

    // Available opponent pool
    std::vector< std::pair< double, int > > all_opponents;
    all_opponents.insert( all_opponents.end(), attackers.begin(), attackers.end() );
    all_opponents.insert( all_opponents.end(), normal_opps.begin(), normal_opps.end() );

    // ---- Layer 1: Back players -> Attackers (most dangerous) ----
    greedyAssign( agent, back_players, attackers, assignments, player_used, opp_used );

    // ---- Layer 2: Middle players -> unmarked Attackers ----
    // Check which attackers are still unmarked
    std::vector< std::pair< double, int > > unmarked_attackers;
    for ( const auto & att : attackers )
    {
        if ( ! opp_used[ att.second ] )
        {
            unmarked_attackers.push_back( att );
        }
    }
    greedyAssign( agent, middle_players, unmarked_attackers, assignments, player_used, opp_used );

    // ---- Layer 3: Idle players (unused Back + unused Middle + Forward) -> remaining ----
    std::vector< int > idle_players;

    // Add unused back players
    for ( int unum : back_players )
    {
        if ( ! player_used[ unum ] )
        {
            idle_players.push_back( unum );
        }
    }
    // Add unused middle players
    for ( int unum : middle_players )
    {
        if ( ! player_used[ unum ] )
        {
            idle_players.push_back( unum );
        }
    }
    // Add forward players (always in Layer 3 - they rarely need to mark deep)
    for ( int unum : forward_players )
    {
        if ( ! player_used[ unum ] )
        {
            idle_players.push_back( unum );
        }
    }

    // Remaining unmarked opponents (attackers first, then normal)
    std::vector< std::pair< double, int > > remaining_opps;
    for ( const auto & att : attackers )
    {
        if ( ! opp_used[ att.second ] )
        {
            remaining_opps.push_back( att );
        }
    }
    for ( const auto & opp : normal_opps )
    {
        if ( ! opp_used[ opp.second ] )
        {
            remaining_opps.push_back( opp );
        }
    }

    greedyAssign( agent, idle_players, remaining_opps, assignments, player_used, opp_used );

    dlog.addText( Logger::TEAM,
                  __FILE__": OMAM assigned %zu players", assignments.size() );
}

// ---------------------------------------------------------------------------
// Greedy Assignment Helper
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::greedyAssign( const PlayerAgent * agent,
                               const std::vector< int > & available_players,
                               const std::vector< std::pair< double, int > > & opponents,
                               std::vector< std::pair< int, int > > & assignments,
                               std::vector< bool > & player_used,
                               std::vector< bool > & opp_used )
{
    const WorldModel & wm = agent->world();

    // For each opponent (sorted by danger desc), find the best unassigned player
    for ( const auto & opp_pair : opponents )
    {
        int opp_unum = opp_pair.second;

        if ( opp_used[ opp_unum ] )
        {
            continue;
        }

        const AbstractPlayerObject * opp = wm.theirPlayer( opp_unum );
        if ( ! opp )
        {
            continue;
        }

        // Find k-top closest available players to this opponent
        std::vector< std::pair< double, int > > candidates; // (dist, unum)

        for ( int p_unum : available_players )
        {
            if ( player_used[ p_unum ] || p_unum == wm.self().unum() )
            {
                // Note: we don't exclude self here because self might be in the list
                // but we only check player_used
            }
            if ( player_used[ p_unum ] )
            {
                continue;
            }

            const AbstractPlayerObject * player = wm.ourPlayer( p_unum );
            if ( ! player || player->unum() == 0 )
            {
                continue;
            }

            double dist = player->pos().dist( opp->pos() );
            if ( dist <= MAX_ASSIGNMENT_DIST )
            {
                candidates.push_back( std::make_pair( dist, p_unum ) );
            }
        }

        if ( candidates.empty() )
        {
            continue;
        }

        // Sort by distance
        std::sort( candidates.begin(), candidates.end() );

        // Take the top-1 (closest) for greedy assignment
        // OPTIMIZATION: for more complex K-top exploration, we could try top-K
        int best_player = candidates[0].second;

        assignments.push_back( std::make_pair( best_player, opp_unum ) );
        player_used[ best_player ] = true;
        opp_used[ opp_unum ] = true;

        dlog.addText( Logger::TEAM,
                      __FILE__": assign our(%d) -> their(%d), dist=%.1f, danger=%.2f",
                      best_player, opp_unum, candidates[0].first, opp_pair.first );
    }
}

// ---------------------------------------------------------------------------
// Marking Position Calculation
// ---------------------------------------------------------------------------

Vector2D
Bhv_OmamMark::calculateMarkingPosition( const PlayerAgent * agent,
                                         const AbstractPlayerObject * target_opponent )
{
    const WorldModel & wm = agent->world();
    const ServerParam & sp = ServerParam::i();

    Vector2D opp_pos = target_opponent->pos();
    Vector2D our_goal( -sp.pitchHalfLength(), 0.0 );
    Vector2D ball_pos = wm.ball().pos();

    // Base position: on the line between opponent and our goal
    Vector2D opp_to_goal = our_goal - opp_pos;
    double opp_to_goal_dist = opp_to_goal.length();

    if ( opp_to_goal_dist < 0.1 )
    {
        // Opponent is at our goal - just stay close
        return opp_pos + Vector2D::polar2vector( MARKING_DISTANCE,
                                                   ( wm.self().pos() - opp_pos ).th() );
    }

    Vector2D opp_to_goal_unit = opp_to_goal / opp_to_goal_dist;

    // Marking position: MARKING_DISTANCE from opponent towards our goal
    Vector2D mark_pos = opp_pos + opp_to_goal_unit * MARKING_DISTANCE;

    // Bias towards ball side: shift perpendicular to goal line towards ball
    Vector2D opp_to_ball = ball_pos - opp_pos;
    Vector2D perp_unit( -opp_to_goal_unit.y, opp_to_goal_unit.x ); // perpendicular

    double ball_side = perp_unit.x * opp_to_ball.x
                      + perp_unit.y * opp_to_ball.y;
    mark_pos += perp_unit * ( ball_side > 0 ? MARKING_BALL_SIDE_BIAS
                                           : -MARKING_BALL_SIDE_BIAS );

    // Clamp to field boundaries
    double pitch_half_w = sp.pitchHalfWidth();
    double pitch_half_l = sp.pitchHalfLength();
    mark_pos.x = std::max( -pitch_half_l + 1.0, std::min( pitch_half_l - 1.0, mark_pos.x ) );
    mark_pos.y = std::max( -pitch_half_w + 1.0, std::min( pitch_half_w - 1.0, mark_pos.y ) );

    return mark_pos;
}
