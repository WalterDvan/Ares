/***************************************************************************
 *  bhv_unmark_move.cpp - Unmark Move Behavior (CYRUS-style)
 *
 *  Algorithm overview:
 *  1. shouldAttemptUnmark(): Check preconditions
 *  2. buildPassPredictionTree(): BFS tree to find future passer
 *  3. calculateUnmarkPosition(): 24 candidate positions, pick best
 *  4. Move to the best position
 ****************************************************************************/

#include "bhv_unmark_move.h"

#include <rcsc/player/world_model.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/player_param.h>
#include <rcsc/geom/sector_2d.h>
#include <rcsc/geom/angle_deg.h>
#include <rcsc/soccer_math.h>

#include "basic_actions/body_go_to_point.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

// Static constants initialization
const double Bhv_UnmarkMove::MIN_CONFIDENCE_THRESHOLD = 0.3;
const double Bhv_UnmarkMove::OFFSIDE_MARGIN = 0.5;
const double Bhv_UnmarkMove::LANE_BASE_WIDTH = 2.5;
const double Bhv_UnmarkMove::RING_RADII[3] = { 2.0, 4.0, 8.0 };
const int Bhv_UnmarkMove::RING_ANGLES = 8;
const int Bhv_UnmarkMove::MAX_TREE_DEPTH = 4;
const int Bhv_UnmarkMove::TOP_N_CANDIDATES = 2;

/*----------------------------------------------------------------------------*/

bool
Bhv_UnmarkMove::execute( rcsc::PlayerAgent * agent )
{
    if ( !shouldAttemptUnmark( agent ) )
    {
        return false;
    }

    // Find current ball holder teammate
    const rcsc::WorldModel & wm = agent->world();
    int holder_unum = -1;
    rcsc::Vector2D holder_pos;

    if ( wm.kickableTeammate() )
    {
        holder_unum = wm.kickableTeammate()->unum();
        holder_pos = wm.kickableTeammate()->pos();
    }
    else if ( wm.lastKickerSide() == wm.ourSide()
              && wm.lastKickerUnum() > 0 )
    {
        holder_unum = wm.lastKickerUnum();
        const rcsc::AbstractPlayerObject * last_kicker = wm.ourPlayer( holder_unum );
        if ( last_kicker )
        {
            holder_pos = last_kicker->pos();
        }
        else
        {
            return false;
        }
    }
    else
    {
        // No identifiable holder
        return false;
    }

    // Build pass prediction tree to find future passer
    rcsc::Vector2D best_passer_pos;
    double confidence = 0.0;

    if ( !buildPassPredictionTree( agent, holder_unum, best_passer_pos, confidence ) )
    {
        return false;
    }

    if ( confidence < MIN_CONFIDENCE_THRESHOLD )
    {
        return false;
    }

    // Calculate best unmark position
    rcsc::Vector2D target_pos;
    if ( !calculateUnmarkPosition( agent, best_passer_pos, target_pos ) )
    {
        return false;
    }

    // Execute the move
    rcsc::Vector2D my_pos = wm.self().pos();
    double dash_power = rcsc::ServerParam::i().maxPower();
    double dist_to_target = my_pos.dist( target_pos );

    // Don't move if already close enough
    if ( dist_to_target < 1.0 )
    {
        return false;
    }

    // Adjust dash power for longer distances (save stamina for short ones)
    if ( dist_to_target < 3.0 )
    {
        dash_power = std::min( dash_power, wm.self().stamina() * 0.5 );
    }

    agent->debugClient().addMessage( "UnmarkMove" );
    agent->debugClient().setTarget( target_pos );
    agent->debugClient().addLine( my_pos, target_pos );
    agent->debugClient().addCircle( best_passer_pos, 2.0 );

    Body_GoToPoint( target_pos,
                           dist_to_target * 0.2, // dash threshold
                           dash_power,
                           1 ).execute( agent );

    return true;
}

/*----------------------------------------------------------------------------*/

bool
Bhv_UnmarkMove::shouldAttemptUnmark( const rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();

    // 1. Must not be in set play mode
    if ( wm.gameMode().isOurSetPlay( wm.ourSide() ) || wm.gameMode().isTheirSetPlay( wm.ourSide() ) )
    {
        return false;
    }

    // 2. Must not have the ball
    if ( wm.self().isKickable() )
    {
        return false;
    }

    // 3. A teammate must have (or just had) the ball
    if ( !wm.kickableTeammate() && wm.lastKickerSide() != wm.ourSide() )
    {
        return false;
    }

    // 4. Ball should be in a forward-ish area (our half toward opponent half)
    //    Only attempt when ball is beyond x > -10 (most of the field)
    if ( wm.ball().pos().x < -10.0 )
    {
        return false;
    }

    // 5. Player should have reasonable stamina
    if ( wm.self().stamina() < rcsc::ServerParam::i().recoverDecThrValue() * 3.0 )
    {
        return false;
    }

    return true;
}

/*----------------------------------------------------------------------------*/

bool
Bhv_UnmarkMove::buildPassPredictionTree( const rcsc::PlayerAgent * agent,
                                          int current_holder_unum,
                                          rcsc::Vector2D & best_passer_pos,
                                          double & confidence )
{
    const rcsc::WorldModel & wm = agent->world();
    int self_unum = wm.self().unum();

    // BFS tree: each node is (teammate_unum, accumulated_confidence)
    // We're looking for a chain of passes that eventually leads to self

    struct TreeNode {
        int unum;
        double confidence;
    };

    std::vector< TreeNode > current_level;
    std::vector< TreeNode > next_level;

    // Start with the current ball holder
    current_level.push_back( { current_holder_unum, 1.0 } );

    // Track the best passer who can pass to self
    int best_passer = -1;
    double best_score = 0.0;

    for ( int depth = 0; depth < MAX_TREE_DEPTH; ++depth )
    {
        next_level.clear();

        // Collect teammates we've already visited in this path
        std::set< int > visited;

        for ( const auto & node : current_level )
        {
            visited.insert( node.unum );
        }

        for ( const auto & node : current_level )
        {
            const rcsc::AbstractPlayerObject * holder = wm.ourPlayer( node.unum );
            if ( !holder )
            {
                continue;
            }

            rcsc::Vector2D holder_pos = holder->pos();

            // Evaluate all teammates as potential receivers from this holder
            std::vector< std::pair< double, int > > candidates; // (score, unum)

            for ( int tm = 1; tm <= 11; ++tm )
            {
                if ( tm == node.unum || visited.count( tm ) )
                {
                    continue;
                }

                const rcsc::AbstractPlayerObject * teammate = wm.ourPlayer( tm );
                if ( !teammate || teammate->unum() == 0 )
                {
                    continue;
                }

                double score = predictPassScore( agent, holder_pos, teammate->pos(), tm );
                candidates.push_back( { score, tm } );
            }

            // Sort by score descending, take top N
            std::sort( candidates.begin(), candidates.end(),
                       []( const auto & a, const auto & b ) { return a.first > b.first; } );

            int take_n = std::min( TOP_N_CANDIDATES, (int)candidates.size() );
            for ( int i = 0; i < take_n; ++i )
            {
                int tm_unum = candidates[i].second;
                double pass_conf = candidates[i].first;
                double new_confidence = node.confidence * pass_conf;

                // Check if this teammate is self - we found the path!
                if ( tm_unum == self_unum )
                {
                    if ( new_confidence > best_score )
                    {
                        best_score = new_confidence;
                        best_passer = node.unum;
                    }
                    continue;
                }

                // Add to next level for deeper search
                next_level.push_back( { tm_unum, new_confidence } );
            }
        }

        current_level = next_level;

        // If no more candidates to explore, stop
        if ( current_level.empty() )
        {
            break;
        }
    }

    if ( best_passer < 0 || best_score < 0.01 )
    {
        return false;
    }

    const rcsc::AbstractPlayerObject * passer = wm.ourPlayer( best_passer );
    if ( !passer )
    {
        return false;
    }

    best_passer_pos = passer->pos();
    confidence = best_score;
    return true;
}

/*----------------------------------------------------------------------------*/

double
Bhv_UnmarkMove::predictPassScore( const rcsc::PlayerAgent * agent,
                                   const rcsc::Vector2D & passer_pos,
                                   const rcsc::Vector2D & receiver_pos,
                                   int receiver_unum )
{
    const rcsc::WorldModel & wm = agent->world();
    const rcsc::AbstractPlayerObject * receiver = wm.ourPlayer( receiver_unum );
    if ( !receiver )
    {
        return 0.0;
    }

    // 1. Offside check
    if ( isOffside( agent, receiver_pos ) )
    {
        return 0.0;
    }

    double score = 0.0;

    // 2. Distance factor - prefer optimal pass distance (~18m), penalize extremes
    double dist = passer_pos.dist( receiver_pos );
    static const double OPTIMAL_PASS_DIST = 18.0;
    static const double DIST_SIGMA = 12.0;
    double dist_score = std::exp( -std::pow( dist - OPTIMAL_PASS_DIST, 2 )
                                  / ( 2.0 * DIST_SIGMA * DIST_SIGMA ) );

    // Penalize very short passes (< 5m) - usually not worth it for unmarking
    if ( dist < 3.0 )
    {
        dist_score *= dist / 3.0;
    }

    score += dist_score * 0.25;

    // 3. Forward preference - prefer passes that go toward opponent goal
    double forward_gain = receiver_pos.x - passer_pos.x;
    double max_forward = 52.68; // half field length
    double forward_score = std::max( 0.0, forward_gain / max_forward );

    // Extra boost for passes into opponent territory
    if ( forward_gain > 5.0 )
    {
        forward_score += 0.1;
    }

    score += forward_score * 0.20;

    // 4. Pass lane clearance - check if opponents block the pass
    double lane_score = checkPassLaneClearance( agent, passer_pos, receiver_pos );
    score += lane_score * 0.30;

    // 5. Receiver openness - prefer receivers far from opponents
    double openness = calculateOpenness( agent, receiver_pos );
    score += openness * 0.15;

    // 6. Proximity to opponent goal - prefer positions closer to opponent goal
    double goal_dist = std::fabs( receiver_pos.x - rcsc::ServerParam::i().pitchHalfLength() );
    double goal_score = 1.0 - std::min( 1.0, goal_dist / 40.0 );
    score += goal_score * 0.10;

    // Clamp to [0, 1]
    return std::max( 0.0, std::min( 1.0, score ) );
}

/*----------------------------------------------------------------------------*/

bool
Bhv_UnmarkMove::calculateUnmarkPosition( const rcsc::PlayerAgent * agent,
                                          const rcsc::Vector2D & passer_pos,
                                          rcsc::Vector2D & target_pos )
{
    const rcsc::WorldModel & wm = agent->world();
    double my_speed_max = wm.self().playerType().realSpeedMax();
    const rcsc::Vector2D self_pos = wm.self().pos();
    const rcsc::AngleDeg self_body = wm.self().body();
    const rcsc::ServerParam & SP = rcsc::ServerParam::i();

    // Generate 24 candidate positions:
    // 3 rings (radius 2, 4, 8 meters) x 8 angles
    // Positions are relative to current self position (CYRUS style)
    // self_pos and my_speed_max already set above

    struct Candidate {
        rcsc::Vector2D pos;
        double my_cycles;     // cycles for self to reach
        double opp_min_cycles; // minimum cycles for any opponent to reach
        double forwardness;   // how close to opponent goal
    };

    std::vector< Candidate > candidates;
    candidates.reserve( RING_ANGLES * 3 ); // 24

    for ( int ring = 0; ring < 3; ++ring )
    {
        double radius = RING_RADII[ring];

        for ( int a = 0; a < RING_ANGLES; ++a )
        {
            double angle = 360.0 * a / RING_ANGLES;

            rcsc::Vector2D candidate = self_pos + rcsc::Vector2D::polar2vector( radius, angle );

            // 1. Field bounds check
            if ( candidate.absX() > SP.pitchHalfLength() - 0.5
                 || candidate.absY() > SP.pitchHalfWidth() - 0.5 )
            {
                continue;
            }

            // 2. Offside check (only in opponent half)
            if ( candidate.x > 0.0 && isOffside( agent, candidate ) )
            {
                continue;
            }

            // 3. Don't go too far back
            if ( candidate.x < passer_pos.x - 10.0 )
            {
                continue;
            }

            Candidate c;

            c.pos = candidate;

            // Self cycles to reach (using player's actual max speed)
            c.my_cycles = cyclesToReach( agent, self_pos, candidate, my_speed_max );

            // Find minimum opponent cycles to reach
            double min_opp_cycles = 999999.0;

            for ( int opp = 1; opp <= 11; ++opp )
            {
                const rcsc::AbstractPlayerObject * opp_player = wm.theirPlayer( opp );
                if ( !opp_player || opp_player->unum() == 0 )
                {
                    continue;
                }

                double opp_dist = opp_player->pos().dist( candidate );
                double opp_speed = opp_player->playerTypePtr()
                                   ? opp_player->playerTypePtr()->realSpeedMax()
                                   : SP.defaultPlayerSpeedMax();

                // Simple estimate: distance / speed, assuming 2-3 cycles reaction
                double opp_cycles = opp_dist / opp_speed + 3.0;
                min_opp_cycles = std::min( min_opp_cycles, opp_cycles );
            }

            c.opp_min_cycles = min_opp_cycles;

            // Forwardness: higher x is closer to opponent goal
            c.forwardness = candidate.x;

            candidates.push_back( c );
        }
    }

    if ( candidates.empty() )
    {
        return false;
    }

    // Select best candidate:
    // Primary: self can arrive before opponents (my_cycles < opp_min_cycles)
    // Secondary: closest to opponent goal line
    Candidate * best = nullptr;
    double best_score = -1e9;

    for ( auto & c : candidates )
    {
        double score = 0.0;

        // Big bonus if we can arrive before opponents
        double arrival_advantage = c.opp_min_cycles - c.my_cycles;

        if ( arrival_advantage > 2.0 )
        {
            // Comfortable lead - very good
            score += 100.0;
        }
        else if ( arrival_advantage > 0.0 )
        {
            // Close but we're first - good
            score += 50.0;
        }
        else if ( arrival_advantage > -3.0 )
        {
            // Slightly behind opponent - still possible (they might not go there)
            score += 20.0;
        }
        else
        {
            // Way behind - not good
            score += 0.0;
        }

        // Forwardness bonus (normalize to [0, 1] range in our half)
        // Our half is negative to positive, so higher x = better
        double forward_score = ( c.pos.x + SP.pitchHalfLength() ) / ( 2.0 * SP.pitchHalfLength() );
        score += forward_score * 30.0;

        // Slight preference for positions with good pass lane from passer
        double lane = checkPassLaneClearance( agent, passer_pos, c.pos );
        score += lane * 10.0;

        // Penalize positions that are too far from passer (unrealistic pass)
        double dist_to_passer = passer_pos.dist( c.pos );
        if ( dist_to_passer > 35.0 )
        {
            score -= 20.0;
        }

        // Penalize positions too close to current position (no point moving)
        if ( c.my_cycles < 2 )
        {
            score -= 15.0;
        }

        if ( score > best_score )
        {
            best_score = score;
            best = &c;
        }
    }

    if ( !best || best_score < 20.0 )
    {
        return false;
    }

    target_pos = best->pos;
    return true;
}

/*----------------------------------------------------------------------------*/

int
Bhv_UnmarkMove::cyclesToReach( const rcsc::PlayerAgent * agent,
                                const rcsc::Vector2D & start_pos,
                                const rcsc::Vector2D & target_pos,
                                double player_speed_max )
{
    double dist = start_pos.dist( target_pos );

    if ( dist < 0.5 )
    {
        return 0;
    }

    // Use player's actual performance parameters
    const rcsc::PlayerType * ptype = &agent->world().self().playerType();
    double player_decay = ptype->playerDecay();
    double effective_speed = player_speed_max;

    // Simplified cycle estimation accounting for inertia
    int cycles = 0;
    double accumulated = 0.0;

    while ( accumulated < dist )
    {
        // First cycle: accelerate from 0
        if ( cycles == 0 )
        {
            accumulated += effective_speed * player_decay;
        }
        else
        {
            // Steady state
            accumulated += effective_speed * player_decay / ( 1.0 - player_decay * player_decay );
        }

        // Safety bound - don't loop forever
        if ( ++cycles > 200 )
        {
            break;
        }
    }

    // Add 1 cycle for turning (need to face direction before dashing)
    cycles += 1;

    return cycles;
}

/*----------------------------------------------------------------------------*/

double
Bhv_UnmarkMove::checkPassLaneClearance( const rcsc::PlayerAgent * agent,
                                         const rcsc::Vector2D & from,
                                         const rcsc::Vector2D & to )
{
    const rcsc::WorldModel & wm = agent->world();

    double dist = from.dist( to );
    if ( dist < 1.0 )
    {
        return 1.0;
    }

    // Lane width increases with distance
    double lane_width = LANE_BASE_WIDTH + dist * 0.05;

    int blocking_opponents = 0;

    for ( int opp = 1; opp <= 11; ++opp )
    {
        const rcsc::AbstractPlayerObject * opp_player = wm.theirPlayer( opp );
        if ( !opp_player || opp_player->unum() == 0 )
        {
            continue;
        }

        // Project opponent position onto the pass line
        rcsc::Vector2D opp_pos = opp_player->pos();
        rcsc::Vector2D pass_dir = ( to - from ).normalizedVector();

        rcsc::Vector2D to_opp = opp_pos - from;
        double projection = to_opp.innerProduct( pass_dir );

        // Only consider opponents between passer and receiver
        if ( projection < 1.0 || projection > dist - 1.0 )
        {
            continue;
        }

        // Calculate perpendicular distance from opponent to pass line
        rcsc::Vector2D closest_on_line = from + pass_dir * projection;
        double perp_dist = opp_pos.dist( closest_on_line );

        if ( perp_dist < lane_width )
        {
            blocking_opponents++;
        }
    }

    // Score: 1.0 = completely clear, 0.0 = completely blocked
    double clearance_score = 1.0 - blocking_opponents * 0.25;
    return std::max( 0.0, std::min( 1.0, clearance_score ) );
}

/*----------------------------------------------------------------------------*/

bool
Bhv_UnmarkMove::isOffside( const rcsc::PlayerAgent * agent,
                            const rcsc::Vector2D & pos )
{
    const rcsc::WorldModel & wm = agent->world();

    // Offside line is the x-coordinate of the second last opponent
    // (includes goalkeeper, so usually second last = last field player)
    double offside_line_x = wm.offsideLineX();

    return pos.x > offside_line_x - OFFSIDE_MARGIN;
}

/*----------------------------------------------------------------------------*/

double
Bhv_UnmarkMove::calculateOpenness( const rcsc::PlayerAgent * agent,
                                     const rcsc::Vector2D & pos )
{
    const rcsc::WorldModel & wm = agent->world();

    double min_dist = 999999.0;

    for ( int opp = 1; opp <= 11; ++opp )
    {
        const rcsc::AbstractPlayerObject * opp_player = wm.theirPlayer( opp );
        if ( !opp_player || opp_player->unum() == 0 )
        {
            continue;
        }

        double d = pos.dist( opp_player->pos() );
        min_dist = std::min( min_dist, d );
    }

    // Very open if > 10m from nearest opponent
    // Not open if < 3m
    if ( min_dist > 15.0 )
    {
        return 1.0;
    }
    else if ( min_dist < 3.0 )
    {
        return 0.0;
    }

    return ( min_dist - 3.0 ) / 12.0;
}
