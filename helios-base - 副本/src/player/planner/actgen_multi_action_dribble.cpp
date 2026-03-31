// -*-c++-*-

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "actgen_multi_action_dribble.h"

#include "dribble.h"
#include "predict_state.h"
#include "action_state_pair.h"

#include <rcsc/player/world_model.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace rcsc;

namespace {

const int ANGLE_DIVS = 8;
const double ANGLE_STEP = 360.0 / ANGLE_DIVS;
const int DIST_DIVS = 3;
const double DIST_STEP = 1.75;
const int OPPONENT_PREDICT_CYCLE = 1;

Vector2D predict_opponent_pos( const AbstractPlayerObject & opponent,
                               const PredictBallObject & ball )
{
    Vector2D predicted = opponent.pos();

    if ( opponent.pos().dist( ball.pos() ) < 10.0 )
    {
        predicted += opponent.vel() * OPPONENT_PREDICT_CYCLE;
    }

    return predicted;
}

void generate_dribbles_from_state( std::vector< ActionStatePair > * result,
                                   const PredictState & base_state,
                                   const AbstractPlayerObject & holder,
                                   const int prefix_kick_count,
                                   const int prefix_turn_count,
                                   const int prefix_dash_count,
                                   const char * description )
{
    static GameTime s_last_call_time( 0, 0 );
    static int s_action_count = 0;

    if ( base_state.currentTime() != s_last_call_time )
    {
        s_action_count = 0;
        s_last_call_time = base_state.currentTime();
    }

    const ServerParam & SP = ServerParam::i();
    const double max_x = SP.pitchHalfLength() - 1.0;
    const double max_y = SP.pitchHalfWidth() - 1.0;
    const int bonus_step = 2;
    const PlayerType * ptype = holder.playerTypePtr();

    if ( ! ptype )
    {
        return;
    }

    for ( int a = 0; a < ANGLE_DIVS; ++a )
    {
        const AngleDeg target_angle = ANGLE_STEP * a;

        if ( holder.pos().x < 16.0
             && target_angle.abs() > 100.0 )
        {
            continue;
        }

        if ( holder.pos().x < -36.0
             && holder.pos().absY() < 20.0
             && target_angle.abs() > 45.0 )
        {
            continue;
        }

        const Vector2D unit_vec = Vector2D::from_polar( 1.0, target_angle );
        for ( int d = 1; d <= DIST_DIVS; ++d )
        {
            const double holder_move_dist = DIST_STEP * d;
            const Vector2D target_point
                = holder.pos()
                + unit_vec.setLengthVector( holder_move_dist );

            if ( target_point.absX() > max_x
                 || target_point.absY() > max_y )
            {
                continue;
            }

            const int dribble_step
                = 1 + 1
                + ptype->cyclesToReachDistance( holder_move_dist - ptype->kickableArea() * 0.5 );

            bool exist_opponent = false;
            for ( AbstractPlayerObject::Cont::const_iterator o = base_state.theirPlayers().begin();
                  o != base_state.theirPlayers().end();
                  ++o )
            {
                const Vector2D predicted_pos = predict_opponent_pos( **o, base_state.ball() );
                const double opp_move_dist = predicted_pos.dist( target_point );
                const int opp_step
                    = 1
                    + (*o)->playerTypePtr()->cyclesToReachDistance( opp_move_dist - ptype->kickableArea() );

                if ( opp_step - bonus_step <= dribble_step )
                {
                    exist_opponent = true;
                    break;
                }
            }

            if ( exist_opponent )
            {
                continue;
            }

            const double ball_speed = SP.firstBallSpeed( base_state.ball().pos().dist( target_point ),
                                                         dribble_step );

            PredictState::ConstPtr result_state( new PredictState( base_state,
                                                                   dribble_step,
                                                                   holder.unum(),
                                                                   target_point ) );
            CooperativeAction::Ptr action( new Dribble( holder.unum(),
                                                        target_point,
                                                        ball_speed,
                                                        prefix_kick_count + 1,
                                                        prefix_turn_count + 1,
                                                        prefix_dash_count + std::max( 0, dribble_step - 2 ),
                                                        description ) );
            ++s_action_count;
            action->setIndex( s_action_count );
            result->push_back( ActionStatePair( action, result_state ) );
        }
    }
}

}

void
ActGen_MultiActionDribble::generate( std::vector< ActionStatePair > * result,
                                     const PredictState & state,
                                     const WorldModel &,
                                     const std::vector< ActionStatePair > & path ) const
{
    if ( ! path.empty() )
    {
        return;
    }

    const AbstractPlayerObject * holder = state.ballHolder();
    if ( ! holder
         || ! holder->playerTypePtr() )
    {
        return;
    }

    const double kickable_area = holder->playerTypePtr()->kickableArea();
    const double safe_ball_dist = std::max( 0.3, kickable_area - 0.25 );
    const Vector2D ball_next = state.ball().pos() + state.ball().vel();

    std::vector< PredictState::ConstPtr > pre_states;
    std::vector< const char * > descriptions;
    std::vector< int > kick_counts;
    std::vector< int > turn_counts;
    std::vector< int > dash_counts;

    for ( int a = -1; a <= 1; ++a )
    {
        const AngleDeg angle = holder->body() + a * 35.0;
        const Vector2D offset = Vector2D::from_polar( safe_ball_dist * 0.6, angle );
        const Vector2D pre_ball_pos = holder->pos() + offset;

        if ( holder->pos().dist( pre_ball_pos ) <= kickable_area - 0.1 )
        {
            pre_states.push_back( PredictState::ConstPtr( new PredictState( state,
                                                                            1,
                                                                            holder->unum(),
                                                                            holder->pos(),
                                                                            pre_ball_pos ) ) );
            descriptions.push_back( "madKickDribble" );
            kick_counts.push_back( 1 );
            turn_counts.push_back( 0 );
            dash_counts.push_back( 0 );
        }
    }

    for ( int side = -1; side <= 1; side += 2 )
    {
        const AngleDeg move_angle = ( state.ball().pos() - holder->pos() ).th() + side * 90.0;
        const Vector2D holder_pos = holder->pos() + Vector2D::from_polar( 0.8, move_angle );

        if ( holder_pos.dist( ball_next ) <= kickable_area - 0.1 )
        {
            pre_states.push_back( PredictState::ConstPtr( new PredictState( state,
                                                                            1,
                                                                            holder->unum(),
                                                                            holder_pos,
                                                                            ball_next ) ) );
            descriptions.push_back( "madMoveDribble" );
            kick_counts.push_back( 0 );
            turn_counts.push_back( 0 );
            dash_counts.push_back( 1 );
        }
    }

    if ( holder->pos().dist( ball_next ) <= kickable_area - 0.1 )
    {
        pre_states.push_back( PredictState::ConstPtr( new PredictState( state,
                                                                        1,
                                                                        holder->unum(),
                                                                        holder->pos(),
                                                                        ball_next ) ) );
        descriptions.push_back( "madTurnDribble" );
        kick_counts.push_back( 0 );
        turn_counts.push_back( 1 );
        dash_counts.push_back( 0 );
    }

    for ( std::size_t i = 0; i < pre_states.size(); ++i )
    {
        const AbstractPlayerObject * pre_holder = pre_states[i]->ballHolder();
        if ( ! pre_holder )
        {
            continue;
        }

        generate_dribbles_from_state( result,
                                      *pre_states[i],
                                      *pre_holder,
                                      kick_counts[i],
                                      turn_counts[i],
                                      dash_counts[i],
                                      descriptions[i] );
    }
}
