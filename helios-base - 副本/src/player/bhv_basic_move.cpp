// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_basic_move.h"

#include "strategy.h"

#include "bhv_basic_tackle.h"

#include "basic_actions/basic_actions.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_intercept.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include "basic_actions/neck_turn_to_low_conf_teammate.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/intercept_table.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include "neck_offensive_intercept_neck.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace rcsc;

namespace {

struct MarkTask {
    const AbstractPlayerObject * opponent;
    double danger;
    bool attacker;
    unsigned long long mask_bit;
};

struct MarkAgent {
    int unum;
    int role;
    Vector2D pos;
    bool back;
    bool middle;
};

struct MarkCandidate {
    int task_index;
    double cost;
};

struct MarkSolution {
    bool valid;
    unsigned long long mask;
    double cost;
    std::vector< int > assigned_tasks;

    MarkSolution()
        : valid( false ),
          mask( 0 ),
          cost( std::numeric_limits< double >::max() )
    { }
};

bool is_back_role( const int role )
{
    return role >= 2 && role <= 5;
}

bool is_middle_role( const int role )
{
    return role >= 6 && role <= 8;
}

bool is_attack_opponent( const WorldModel & wm,
                         const AbstractPlayerObject & opponent )
{
    return opponent.pos().x > 10.0
        || ( opponent.pos().x > -5.0
             && opponent.pos().x > wm.ball().pos().x - 7.0 );
}

double get_danger_score( const WorldModel & wm,
                         const AbstractPlayerObject & opponent )
{
    const double goal_term = ( ServerParam::i().pitchHalfLength() + opponent.pos().x ) * 1.4;
    const double ball_term = std::max( 0.0, 22.0 - opponent.pos().dist( wm.ball().pos() ) );
    const double center_term = std::max( 0.0, 16.0 - opponent.pos().absY() ) * 0.35;
    return goal_term + ball_term + center_term;
}

double get_assignment_cost( const MarkAgent & agent,
                            const MarkTask & task )
{
    double cost = agent.pos.dist( task.opponent->pos() );

    if ( agent.back && ! task.attacker )
    {
        cost += 20.0;
    }
    else if ( ! agent.back && task.attacker )
    {
        cost += 3.0;
    }

    cost += std::max( 0.0, std::fabs( agent.pos.y - task.opponent->pos().y ) - 12.0 ) * 0.1;
    return cost;
}

void search_marking( const std::vector< MarkAgent > & agents,
                     const std::vector< std::vector< MarkCandidate > > & candidates,
                     const std::vector< MarkTask > & tasks,
                     std::size_t index,
                     unsigned int used_mask,
                     unsigned long long covered_mask,
                     double total_cost,
                     std::vector< int > & current_assignment,
                     MarkSolution * best )
{
    if ( index >= agents.size() )
    {
        if ( ! best->valid
             || covered_mask > best->mask
             || ( covered_mask == best->mask
                  && total_cost < best->cost ) )
        {
            best->valid = true;
            best->mask = covered_mask;
            best->cost = total_cost;
            best->assigned_tasks = current_assignment;
        }
        return;
    }

    search_marking( agents, candidates, tasks,
                    index + 1,
                    used_mask,
                    covered_mask,
                    total_cost,
                    current_assignment,
                    best );

    for ( std::vector< MarkCandidate >::const_iterator it = candidates[index].begin();
          it != candidates[index].end();
          ++it )
    {
        if ( used_mask & ( 1U << it->task_index ) )
        {
            continue;
        }

        current_assignment[index] = it->task_index;
        search_marking( agents, candidates, tasks,
                        index + 1,
                        used_mask | ( 1U << it->task_index ),
                        covered_mask | tasks[it->task_index].mask_bit,
                        total_cost + it->cost,
                        current_assignment,
                        best );
        current_assignment[index] = -1;
    }
}

bool get_mark_target( const WorldModel & wm,
                      Vector2D * target_point,
                      int * opponent_unum )
{
    if ( ! Strategy::i().isMarkerType( wm.self().unum() )
         || wm.self().goalie()
         || wm.gameMode().type() != GameMode::PlayOn )
    {
        return false;
    }

    std::vector< MarkTask > tasks;
    for ( PlayerObject::Cont::const_iterator it = wm.opponentsFromSelf().begin();
          it != wm.opponentsFromSelf().end();
          ++it )
    {
        const AbstractPlayerObject * opp = *it;
        if ( ! opp
             || opp->posCount() > 5
             || opp->isGhost()
             || opp->goalie()
             || opp->pos().x < -20.0 )
        {
            continue;
        }

        MarkTask task;
        task.opponent = opp;
        task.danger = get_danger_score( wm, *opp );
        task.attacker = is_attack_opponent( wm, *opp );
        task.mask_bit = 0;
        tasks.push_back( task );
    }

    if ( tasks.empty() )
    {
        return false;
    }

    std::sort( tasks.begin(), tasks.end(),
               []( const MarkTask & lhs, const MarkTask & rhs )
               {
                   return lhs.danger > rhs.danger;
               } );

    if ( tasks.size() > 6 )
    {
        tasks.resize( 6 );
    }

    for ( std::size_t i = 0; i < tasks.size(); ++i )
    {
        tasks[i].mask_bit = 1ULL << ( tasks.size() - 1 - i );
    }

    std::vector< MarkAgent > agents;
    for ( int unum = 1; unum <= 11; ++unum )
    {
        const int role = Strategy::i().roleNumber( unum );
        if ( unum == Strategy::i().goalieUnum()
             || role > 8 )
        {
            continue;
        }

        const AbstractPlayerObject * player = ( unum == wm.self().unum()
                                                ? &wm.self()
                                                : wm.ourPlayer( unum ) );
        if ( ! player
             || ( unum != wm.self().unum() && player->posCount() > 6 ) )
        {
            continue;
        }

        MarkAgent agent;
        agent.unum = unum;
        agent.role = role;
        agent.pos = player->pos();
        agent.back = is_back_role( role );
        agent.middle = is_middle_role( role );
        agents.push_back( agent );
    }

    if ( agents.empty() )
    {
        return false;
    }

    std::vector< std::vector< MarkCandidate > > candidates( agents.size() );
    for ( std::size_t i = 0; i < agents.size(); ++i )
    {
        for ( std::size_t t = 0; t < tasks.size(); ++t )
        {
            MarkCandidate candidate;
            candidate.task_index = static_cast< int >( t );
            candidate.cost = get_assignment_cost( agents[i], tasks[t] );
            candidates[i].push_back( candidate );
        }

        std::sort( candidates[i].begin(), candidates[i].end(),
                   []( const MarkCandidate & lhs, const MarkCandidate & rhs )
                   {
                       return lhs.cost < rhs.cost;
                   } );

        if ( candidates[i].size() > 3 )
        {
            candidates[i].resize( 3 );
        }
    }

    MarkSolution best;
    std::vector< int > current_assignment( agents.size(), -1 );
    search_marking( agents, candidates, tasks,
                    0, 0U, 0ULL, 0.0,
                    current_assignment,
                    &best );

    if ( ! best.valid )
    {
        return false;
    }

    for ( std::size_t i = 0; i < agents.size(); ++i )
    {
        if ( agents[i].unum != wm.self().unum()
             || best.assigned_tasks[i] < 0 )
        {
            continue;
        }

        const MarkTask & task = tasks[ best.assigned_tasks[i] ];
        Vector2D block_from = wm.ball().pos();
        if ( block_from.dist( task.opponent->pos() ) > 20.0 )
        {
            block_from.assign( -ServerParam::i().pitchHalfLength(), 0.0 );
        }

        Vector2D offset = block_from - task.opponent->pos();
        if ( offset.r() < 0.5 )
        {
            offset.assign( -1.0, 0.0 );
        }
        offset.setLength( task.attacker ? 1.8 : 2.4 );

        *target_point = task.opponent->pos() + offset;
        *opponent_unum = task.opponent->unum();
        return true;
    }

    return false;
}

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Bhv_BasicMove::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::TEAM,
                  __FILE__": Bhv_BasicMove" );

    //-----------------------------------------------
    // tackle
    if ( Bhv_BasicTackle( 0.8, 80.0 ).execute( agent ) )
    {
        return true;
    }

    const WorldModel & wm = agent->world();
    /*--------------------------------------------------------*/
    // chase ball
    const int self_min = wm.interceptTable().selfStep();
    const int mate_min = wm.interceptTable().teammateStep();
    const int opp_min = wm.interceptTable().opponentStep();

    if ( ! wm.kickableTeammate()
         && ( self_min <= 3
              || ( self_min <= mate_min
                   && self_min < opp_min + 3 )
              )
         )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": intercept" );
        Body_Intercept().execute( agent );
        agent->setNeckAction( new Neck_OffensiveInterceptNeck() );

        return true;
    }

    Vector2D target_point = Strategy::i().getPosition( wm.self().unum() );
    double dash_power = Strategy::get_normal_dash_power( wm );
    int marking_opponent = Unum_Unknown;

    if ( get_mark_target( wm, &target_point, &marking_opponent ) )
    {
        dash_power = ServerParam::i().maxDashPower();
        agent->debugClient().addMessage( "Mark%d", marking_opponent );
    }

    double dist_thr = wm.ball().distFromSelf() * 0.1;
    if ( dist_thr < 1.0 ) dist_thr = 1.0;
    if ( marking_opponent != Unum_Unknown )
    {
        dist_thr = 0.7;
    }

    dlog.addText( Logger::TEAM,
                  __FILE__": Bhv_BasicMove target=(%.1f %.1f) dist_thr=%.2f",
                  target_point.x, target_point.y,
                  dist_thr );

    agent->debugClient().addMessage( "BasicMove%.0f", dash_power );
    agent->debugClient().setTarget( target_point );
    agent->debugClient().addCircle( target_point, dist_thr );

    if ( ! Body_GoToPoint( target_point, dist_thr, dash_power
                           ).execute( agent ) )
    {
        Body_TurnToBall().execute( agent );
    }

    if ( wm.kickableOpponent()
         && wm.ball().distFromSelf() < 18.0 )
    {
        agent->setNeckAction( new Neck_TurnToBall() );
    }
    else
    {
        agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
    }

    return true;
}
