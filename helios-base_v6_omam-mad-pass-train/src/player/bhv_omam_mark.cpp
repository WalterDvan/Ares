/***************************************************************************
 *  bhv_omam_mark.cpp - OMAM Marking Defense (Paper-Faithful)
 *
 *  Complete OMAM algorithm from CYRUS 2021 paper.
 *  Key design decisions:
 *  - OMAM is a layer in doMove(), NOT a replacement for UnmarkMove
 *  - Only players who get an assignment execute; others fall through to UnmarkMove
 *  - Marking position: ball-side (between opponent and ball), NOT goal-side
 *  - Danger score normalized to [0,1], threshold at 0.20 for attacker classification
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
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <cfloat>

using namespace rcsc;

// ---------------------------------------------------------------------------
// Constants (paper parameters)
// ---------------------------------------------------------------------------

static const int    K_TOP = 3;              // top-k candidates per player
static const double DANGER_THRESHOLD = 0.20; // min danger to be "Attacker"
static const double MARK_DIST = 1.5;        // meters to position from opponent
static const double PENALTY_AREA_X = -36.0;
static const double PENALTY_AREA_Y = 20.16;
static const double GOAL_HALF_WIDTH = 7.32;  // actually goal half-width
static const double OMAHA_MAX_DIST = 50.0;   // max distance to consider for matching

// Danger weightings (paper section 3.3)
static const double W_GOAL      = 0.35;  // distance to our goal
static const double W_BALL      = 0.30;  // distance to ball
static const double W_ANGLE     = 0.15;  // angle to goal center
static const double W_OWN_HALF  = 0.10;  // bonus if in our half
static const double W_PENALTY   = 0.15;  // bonus if in penalty area
static const double W_VELOCITY  = 0.10;  // speed toward our goal

// ---------------------------------------------------------------------------
// Main Entry
// ---------------------------------------------------------------------------

bool
Bhv_OmamMark::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    // Filter: only activate in normal play
    if ( ! shouldActivate( agent ) )
        return false;

    // Don't activate if ball is very far in their half (no need to mark)
    double ball_x = wm.ball().pos().x;
    if ( ball_x > 30.0 )
        return false;

    // Step 1: Classify our players
    std::vector< int > back, middle, forward;
    classifyOurPlayers( agent, back, middle, forward );

    // Step 2: Classify opponents
    std::vector< std::pair< double, int > > attackers;     // (danger, unum)
    std::vector< std::pair< double, int > > non_attackers; // (danger, unum)
    classifyOpponents( agent, attackers, non_attackers );

    if ( attackers.empty() )
        return false;

    // Step 3: Run OMAM matching
    std::vector< Assignment > assignments;
    omamMatch( agent, back, middle, forward, attackers, non_attackers, assignments );

    // Step 4: Find this player's assignment
    int self_unum = wm.self().unum();
    const Assignment * my_assign = nullptr;
    for ( const auto & a : assignments )
    {
        if ( a.our_unum == self_unum )
        {
            my_assign = &a;
            break;
        }
    }

    // No assignment -> fall through to UnmarkMove
    if ( ! my_assign )
        return false;

    // Step 5: Calculate marking position and go there
    Vector2D mark_pos = markPosition( agent, my_assign->opp_unum );

    dlog.addText( Logger::TEAM,
                  __FILE__": OMAM unum=%d marking opp=%d (danger=%.2f) -> (%.1f, %.1f)",
                  self_unum, my_assign->opp_unum, my_assign->opp_danger,
                  mark_pos.x, mark_pos.y );

    agent->debugClient().addMessage( "OMAM%d", my_assign->opp_unum );
    agent->debugClient().setTarget( mark_pos );
    agent->debugClient().addCircle( mark_pos, 1.0 );

    const AbstractPlayerObject * opp = wm.theirPlayer( my_assign->opp_unum );
    if ( opp )
        agent->debugClient().addLine( opp->pos(), mark_pos );

    goToMark( agent, mark_pos );
    return true;
}

// ---------------------------------------------------------------------------
// Activation check
// ---------------------------------------------------------------------------

bool
Bhv_OmamMark::shouldActivate( const rcsc::PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    // Only for field players, not goalie
    if ( wm.self().goalie() )
        return false;

    // Not during our set plays
    if ( wm.gameMode().isOurSetPlay( wm.ourSide() ) )
        return false;

    // Don't override if we can intercept the ball
    int self_min = wm.interceptTable().selfStep();
    int mate_min = wm.interceptTable().teammateStep();
    int opp_min  = wm.interceptTable().opponentStep();
    if ( self_min <= mate_min && self_min <= opp_min + 3 )
        return false;

    // Not if ball is in very far opponent half
    if ( wm.ball().pos().x > 35.0 )
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Danger Score
// ---------------------------------------------------------------------------

double
Bhv_OmamMark::dangerScore( const rcsc::PlayerAgent * agent, int opp_unum )
{
    const WorldModel & wm = agent->world();
    const ServerParam & sp = ServerParam::i();
    const AbstractPlayerObject * opp = wm.theirPlayer( opp_unum );

    if ( ! opp || ! opp->posCount() < 3 )
        return 0.0;

    Vector2D opp_pos = opp->pos();
    Vector2D ball_pos = wm.ball().pos();
    Vector2D our_goal( -sp.pitchHalfLength(), 0.0 );

    // 1. Distance to our goal (normalized, closer = more dangerous)
    double d_goal = opp_pos.dist( our_goal );
    double max_d = sp.pitchHalfLength() * 2.0;
    double score_goal = 1.0 - std::min( d_goal / max_d, 1.0 );

    // 2. Distance to ball (closer = more dangerous, ball-carrier is threat)
    double d_ball = opp_pos.dist( ball_pos );
    double score_ball = 1.0 - std::min( d_ball / 30.0, 1.0 );

    // 3. Angle to goal center (smaller angle = more direct threat)
    Vector2D opp_to_goal = our_goal - opp_pos;
    double angle_deg = std::fabs( opp_to_goal.th().degree() - opp->body().degree() );
    if ( angle_deg > 180.0 ) angle_deg = 360.0 - angle_deg;
    double score_angle = 1.0 - std::min( angle_deg / 180.0, 1.0 );

    // 4. In own half bonus
    double score_half = ( opp_pos.x < 0.0 ) ? 1.0 : 0.0;

    // 5. In penalty area bonus
    double score_penalty = ( opp_pos.x < PENALTY_AREA_X
                            && std::fabs( opp_pos.y ) < PENALTY_AREA_Y ) ? 1.0 : 0.0;

    // 6. Velocity toward our goal
    double score_vel = 0.0;
    if ( opp->velCount() < 3 && opp->vel().length() > 0.1 )
    {
        Vector2D vel_unit = opp->vel() / opp->vel().length();
        Vector2D goal_unit = ( our_goal - opp_pos );
        double goal_dist = goal_unit.length();
        if ( goal_dist > 0.1 )
        {
            goal_unit /= goal_dist;
            // dot product: positive = moving toward goal
            double dot = vel_unit.x * goal_unit.x + vel_unit.y * goal_unit.y;
            score_vel = std::max( 0.0, dot ); // 0~1
        }
    }

    double danger = W_GOAL * score_goal
                 + W_BALL * score_ball
                 + W_ANGLE * score_angle
                 + W_OWN_HALF * score_half
                 + W_PENALTY * score_penalty
                 + W_VELOCITY * score_vel;

    return std::max( 0.0, std::min( 1.0, danger ) );
}

// ---------------------------------------------------------------------------
// Classify our players
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::classifyOurPlayers( const rcsc::PlayerAgent * agent,
                                    std::vector< int > & back,
                                    std::vector< int > & middle,
                                    std::vector< int > & forward )
{
    back.clear();
    middle.clear();
    forward.clear();

    const WorldModel & wm = agent->world();

    for ( int unum = 2; unum <= 11; ++unum )
    {
        if ( unum == wm.self().unum() )
            continue; // skip self (handle in execute)

        const AbstractPlayerObject * mate = wm.ourPlayer( unum );
        if ( ! mate || ! mate->posCount() < 3 )
            continue;

        switch ( unum )
        {
        case 2: case 3: // CB, CB
        case 4: case 5: // SB, SB
        case 6:          // DH
            back.push_back( unum );
            break;
        case 7: case 8:  // OH, OH
            middle.push_back( unum );
            break;
        case 9: case 10: // SF, SF
        case 11:         // CF
            forward.push_back( unum );
            break;
        default:
            forward.push_back( unum );
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Classify opponents
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::classifyOpponents( const rcsc::PlayerAgent * agent,
                                   std::vector< std::pair< double, int > > & attackers,
                                   std::vector< std::pair< double, int > > & non_attackers )
{
    attackers.clear();
    non_attackers.clear();

    const WorldModel & wm = agent->world();

    std::vector< std::pair< double, int > > all_opps;

    for ( int unum = 1; unum <= 11; ++unum )
    {
        const AbstractPlayerObject * opp = wm.theirPlayer( unum );
        if ( ! opp || ! opp->posCount() < 3 )
            continue;

        double danger = dangerScore( agent, unum );
        all_opps.push_back( std::make_pair( danger, unum ) );
    }

    // Sort by danger descending
    std::sort( all_opps.begin(), all_opps.end(),
               []( const std::pair<double,int> & a, const std::pair<double,int> & b )
               { return a.first > b.first; } );

    for ( const auto & p : all_opps )
    {
        if ( p.first >= DANGER_THRESHOLD )
            attackers.push_back( p );
        else
            non_attackers.push_back( p );
    }
}

// ---------------------------------------------------------------------------
// Assignment cost (lower = better assignment)
// ---------------------------------------------------------------------------

double
Bhv_OmamMark::assignmentCost( const rcsc::PlayerAgent * agent,
                                int our_unum, int opp_unum )
{
    const WorldModel & wm = agent->world();

    const AbstractPlayerObject * mate = wm.ourPlayer( our_unum );
    const AbstractPlayerObject * opp = wm.theirPlayer( opp_unum );

    if ( ! mate || ! mate->posCount() < 3 || ! opp || ! opp->posCount() < 3 )
        return DBL_MAX;

    double dist = mate->pos().dist( opp->pos() );

    // Cost = distance (further away = higher cost)
    // Also penalize assigning a forward to a dangerous attacker (role mismatch)
    double cost = dist;

    if ( dist > OMAHA_MAX_DIST )
        return DBL_MAX;

    return cost;
}

// ---------------------------------------------------------------------------
// Greedy layer assignment: match avail_players to target_opps
// For each opponent, find the closest available player that wants to mark them.
// Paper uses K-top: each player keeps top-K nearest opponents.
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::greedyLayer( const rcsc::PlayerAgent * agent,
                             const std::vector< int > & avail_players,
                             const std::vector< std::pair< double, int > > & target_opps,
                             std::vector< Assignment > & assignments,
                             std::vector< bool > & player_used,
                             std::vector< bool > & opp_used )
{
    // For each opponent (sorted by danger desc), find the cheapest available player
    for ( const auto & opp_info : target_opps )
    {
        int opp_unum = opp_info.second;
        double opp_danger = opp_info.first;

        if ( opp_used[opp_unum] )
            continue;

        double best_cost = DBL_MAX;
        int best_player = -1;

        for ( int our_unum : avail_players )
        {
            if ( player_used[our_unum] )
                continue;

            double cost = assignmentCost( agent, our_unum, opp_unum );
            if ( cost < best_cost )
            {
                best_cost = cost;
                best_player = our_unum;
            }
        }

        if ( best_player >= 0 && best_cost < OMAHA_MAX_DIST )
        {
            player_used[best_player] = true;
            opp_used[opp_unum] = true;

            Assignment a;
            a.our_unum = best_player;
            a.opp_unum = opp_unum;
            a.opp_danger = opp_danger;
            a.total_cost = best_cost;
            assignments.push_back( a );
        }
    }
}

// ---------------------------------------------------------------------------
// OMAM layered matching
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::omamMatch( const rcsc::PlayerAgent * agent,
                            const std::vector< int > & back,
                            const std::vector< int > & middle,
                            const std::vector< int > & forward,
                            const std::vector< std::pair< double, int > > & attackers,
                            const std::vector< std::pair< double, int > > & non_attackers,
                            std::vector< Assignment > & result )
{
    result.clear();

    // Track which players/opponents are already assigned
    // Index: unum (1-11), use size 12 for direct indexing
    std::vector< bool > player_used( 12, false );
    std::vector< bool > opp_used( 12, false );

    // Mark self as unavailable for assignment (self is the one calling execute)
    const WorldModel & wm = agent->world();
    // Self is eligible for assignment too (the caller checks if self has assignment)

    // === Layer 1: Back players -> Attackers ===
    // Priority: most dangerous attackers first
    greedyLayer( agent, back, attackers, result, player_used, opp_used );

    // === Layer 2: Middle players -> remaining unmarked Attackers ===
    greedyLayer( agent, middle, attackers, result, player_used, opp_used );

    // === Layer 3: Forward players -> remaining Non-Attackers ===
    greedyLayer( agent, forward, non_attackers, result, player_used, opp_used );

    // === Layer 4: Any remaining Back/Middle -> any remaining opponents ===
    // Fill gaps: if there are still unassigned back/middle and unmarked opponents
    std::vector< int > remaining_back;
    for ( int u : back )
        if ( ! player_used[u] ) remaining_back.push_back( u );
    std::vector< int > remaining_mid;
    for ( int u : middle )
        if ( ! player_used[u] ) remaining_mid.push_back( u );

    std::vector< std::pair< double, int > > remaining_opps;
    for ( const auto & p : non_attackers )
        if ( ! opp_used[p.second] ) remaining_opps.push_back( p );
    for ( const auto & p : attackers )
        if ( ! opp_used[p.second] ) remaining_opps.push_back( p );

    std::sort( remaining_opps.begin(), remaining_opps.end(),
               []( const std::pair<double,int> & a, const std::pair<double,int> & b )
               { return a.first > b.first; } );

    greedyLayer( agent, remaining_mid, remaining_opps, result, player_used, opp_used );
    greedyLayer( agent, remaining_back, remaining_opps, result, player_used, opp_used );
}

// ---------------------------------------------------------------------------
// Marking position: ball-side (between opponent and ball)
// Paper says mark between opponent and ball, with slight goal-side bias
// ---------------------------------------------------------------------------

rcsc::Vector2D
Bhv_OmamMark::markPosition( const rcsc::PlayerAgent * agent, int opp_unum )
{
    const WorldModel & wm = agent->world();
    const ServerParam & sp = ServerParam::i();

    const AbstractPlayerObject * opp = wm.theirPlayer( opp_unum );
    if ( ! opp )
        return wm.ball().pos();

    Vector2D opp_pos = opp->pos();
    Vector2D ball_pos = wm.ball().pos();
    Vector2D our_goal( -sp.pitchHalfLength(), 0.0 );

    // Direction: opponent -> ball (ball-side marking)
    Vector2D opp_to_ball = ball_pos - opp_pos;
    double opp_to_ball_dist = opp_to_ball.length();

    Vector2D mark_pos;

    if ( opp_to_ball_dist > 0.1 )
    {
        Vector2D opp_to_ball_unit = opp_to_ball / opp_to_ball_dist;

        // Direction: opponent -> goal
        Vector2D opp_to_goal = our_goal - opp_pos;
        double opp_to_goal_dist = opp_to_goal.length();

        Vector2D combined_unit;
        if ( opp_to_goal_dist > 0.1 && opp_to_ball_dist > 0.1 )
        {
            Vector2D opp_to_goal_unit = opp_to_goal / opp_to_goal_dist;

            // Paper: marking position between opponent and ball (mostly ball-side)
            // With slight goal-side bias for positioning
            double ball_bias = 0.8;
            double goal_bias = 0.2;

            // If opponent has the ball (is kickable), go more goal-side
            double opp_ball_dist = opp_pos.dist( ball_pos );
            if ( opp_ball_dist < 1.5 )
            {
                // Opponent close to ball, stay goal-side to block shot
                ball_bias = 0.5;
                goal_bias = 0.5;
            }

            // If opponent is very close to our goal, prioritize goal-side
            if ( opp_pos.x < -30.0 )
            {
                double proximity = std::max( 0.0, -30.0 - opp_pos.x ) / 20.0;
                proximity = std::min( proximity, 0.4 );
                goal_bias += proximity;
                ball_bias -= proximity;
            }

            combined_unit = opp_to_ball_unit * ball_bias + opp_to_goal_unit * goal_bias;
            double comb_len = combined_unit.length();
            if ( comb_len > 0.01 )
                combined_unit /= comb_len;
            else
                combined_unit = opp_to_ball_unit;
        }
        else
        {
            combined_unit = opp_to_ball_unit;
        }

        // Position: MARK_DIST from opponent in combined direction
        mark_pos = opp_pos + combined_unit * MARK_DIST;

        // Also position slightly closer to opponent than MARK_DIST if ball is far
        // This keeps tighter marking
        if ( opp_to_ball_dist > 15.0 && opp_pos.dist( our_goal ) < 20.0 )
        {
            // Tight marking for close-to-goal opponents
            mark_pos = opp_pos + combined_unit * ( MARK_DIST * 0.7 );
        }
    }
    else
    {
        // Ball is on the opponent, go goal-side
        Vector2D opp_to_goal = our_goal - opp_pos;
        double gdist = opp_to_goal.length();
        if ( gdist > 0.1 )
            mark_pos = opp_pos + ( opp_to_goal / gdist ) * MARK_DIST;
        else
            mark_pos = opp_pos + Vector2D( -MARK_DIST, 0.0 );
    }

    // Clamp to field
    double hw = sp.pitchHalfWidth();
    double hl = sp.pitchHalfLength();
    mark_pos.x = std::max( -hl + 0.5, std::min( hl - 0.5, mark_pos.x ) );
    mark_pos.y = std::max( -hw + 0.5, std::min( hw - 0.5, mark_pos.y ) );

    return mark_pos;
}

// ---------------------------------------------------------------------------
// Go to marking position
// ---------------------------------------------------------------------------

void
Bhv_OmamMark::goToMark( PlayerAgent * agent, const Vector2D & pos )
{
    const WorldModel & wm = agent->world();

    double dist_thr = wm.ball().distFromSelf() * 0.1;
    if ( dist_thr < 1.0 ) dist_thr = 1.0;

    double dash_power = Strategy::get_normal_dash_power( wm );

    if ( ! Body_GoToPoint( pos, dist_thr, dash_power,
                           1 ).execute( agent ) )
    {
        Body_TurnToBall().execute( agent );
    }

    agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
}
