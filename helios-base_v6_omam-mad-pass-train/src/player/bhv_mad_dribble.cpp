/***************************************************************************
 *  bhv_mad_dribble.cpp - Multi Action Dribble (MAD)
 *
 *  Implementation of the MAD algorithm from CYRUS 2021 paper.
 *
 *  Current state: DATA COLLECTION MODE
 *  - No DNN model yet, so this module only collects training data
 *  - When the player has the ball and opponents are near, it logs features
 *  - After accumulating data, train DNN using CppDNN library
 *  - Place trained model in conf/dnn_mad_dribble.model
 *  - This module will then load the model and predict optimal deception actions
 *
 *  Actions:
 *  0 = No deception (direct dribble / fallback)
 *  1 = Turn before kick (left 30°)
 *  2 = Turn before kick (right 30°)
 *  3 = Turn before kick (left 60°)
 *  4 = Turn before kick (right 60°)
 *  5 = Move before kick (move toward ball left side)
 *  6 = Move before kick (move toward ball right side)
 *  7 = Two-step kick (kick ball to nearby position first)
 *  8 = Two-step kick (kick ball backward)
 *  9 = Turn + Move (turn then approach from different angle)
 *  10 = Feint stop (momentary stop then accelerate)
 *
 *  Feature extraction (738 dims as per paper):
 *  - Self state: pos, vel, body angle, stamina, etc. (~15)
 *  - Ball state: pos, vel relative fields (~10)
 *  - Each visible teammate: pos_rel, vel_rel, angle_rel, bodyangle_rel (~22 each, x11 = 242)
 *  - Each visible opponent: pos_rel, vel_rel, angle_rel, bodyangle_rel (~22 each, x11 = 242)
 *  - Game state features: game mode, score diff, time remaining (~20)
 *  - Spatial context: zone info, distance to important areas (~209)
 *
 *  Label: the action that was actually executed and resulted in dribble success
 ****************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_mad_dribble.h"

#include "basic_actions/body_turn_to_angle.h"
#include "basic_actions/body_go_to_point.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/player_object.h>
#include <rcsc/player/world_model.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/logger.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

static const std::string MODEL_DIR = "conf/";
static const std::string MODEL_FILENAME = "dnn_mad_dribble.model";
static const std::string DATA_LOG_DIR = "mad_training_data/";
static const double THREAT_RADIUS = 10.0;
static bool s_model_loaded = false;

// ---------------------------------------------------------------------------
// Init: check for DNN model
// ---------------------------------------------------------------------------

static bool
checkModelFile()
{
    std::string path = MODEL_DIR + MODEL_FILENAME;
    struct stat st;
    if ( stat( path.c_str(), &st ) != 0 )
        return false;
    return st.st_size > 100; // model must be non-trivial
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

bool
Bhv_MadDribble::execute( rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();

    if ( ! shouldActivate( agent ) )
        return false;

    // Collect training data (always, when condition is met)
    collectTrainingData( agent );

    // If DNN model is loaded, use it to predict the best action
    if ( modelLoaded() )
    {
        std::vector< double > features = extractFeatures( agent );
        int best_action = predictBestAction( agent, features );

        if ( best_action == 0 )
        {
            // No deception, fall through to normal dribble
            return false;
        }

        rcsc::dlog.addText( rcsc::Logger::TEAM,
                      __FILE__": MAD action=%d (DNN prediction)",
                      best_action );
        agent->debugClient().addMessage( "MAD%d", best_action );

        if ( best_action >= 1 && best_action <= 4 )
            return executeTurnDeception( agent, best_action );
        if ( best_action >= 5 && best_action <= 6 )
            return executeMoveDeception( agent, best_action );
        if ( best_action >= 7 && best_action <= 8 )
        {
            // Two-step kick: just return false and let normal dribble handle it
            // (in DNN mode, the action chain planner handles this)
            return false;
        }
        if ( best_action == 9 )
            return executeTurnDeception( agent, 1 ); // re-use turn
        if ( best_action == 10 )
            return false; // feint stop handled by action chain

        return false;
    }

    // No DNN: just collect data, don't interfere with normal dribble
    return false;
}

// ---------------------------------------------------------------------------
// Should MAD activate?
// ---------------------------------------------------------------------------

bool
Bhv_MadDribble::shouldActivate( const rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();

    // Only if we have the ball
    if ( ! wm.self().isKickable() )
        return false;

    // Only if there are opponents nearby
    std::vector< int > nearby;
    getNearbyOpponents( agent, nearby );

    return ! nearby.empty();
}

// ---------------------------------------------------------------------------
// Nearby opponents
// ---------------------------------------------------------------------------

void
Bhv_MadDribble::getNearbyOpponents( const rcsc::PlayerAgent * agent,
                                      std::vector< int > & opp_unums )
{
    opp_unums.clear();
    const rcsc::WorldModel & wm = agent->world();

    const rcsc::PlayerObject::Cont & opps = wm.opponents();
    for ( const auto & opp : opps )
    {
        if ( opp->distFromSelf() < THREAT_RADIUS && opp->unum() > 0 )
            opp_unums.push_back( opp->unum() );
    }
}

// ---------------------------------------------------------------------------
// Feature extraction (738-dim vector)
// ---------------------------------------------------------------------------

std::vector< double >
Bhv_MadDribble::extractFeatures( const rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();
    const rcsc::ServerParam & sp = rcsc::ServerParam::i();
    std::vector< double > features;

    // --- Self features (15) ---
    features.push_back( wm.self().pos().x / sp.pitchHalfLength() );           // 0
    features.push_back( wm.self().pos().y / sp.pitchHalfWidth() );            // 1
    features.push_back( wm.self().vel().x / 3.0 );                            // 2
    features.push_back( wm.self().vel().y / 3.0 );                            // 3
    features.push_back( wm.self().body().radian() / rcsc::AngleDeg::PI );     // 4
    features.push_back( wm.self().face().radian() / rcsc::AngleDeg::PI );     // 5
    features.push_back( wm.self().stamina() / sp.staminaMax() );              // 6
    features.push_back( wm.self().effort() );                                  // 7
    features.push_back( wm.self().recovery() );                               // 8
    features.push_back( wm.self().posCount() / 10.0 );                       // 9
    features.push_back( wm.ball().distFromSelf() / sp.pitchHalfLength() );    // 10
    features.push_back( wm.self().pos().dist( rcsc::Vector2D( -52.5, 0.0 ) ) / sp.pitchHalfLength() ); // 11 dist to own goal
    features.push_back( wm.self().pos().dist( rcsc::Vector2D( 52.5, 0.0 ) ) / sp.pitchHalfLength() );  // 12 dist to opp goal
    features.push_back( (wm.ball().angleFromSelf() - wm.self().body()).radian() / rcsc::AngleDeg::PI ); // 13
    features.push_back( wm.self().unum() / 11.0 );                            // 14

    // --- Ball features (10) ---
    features.push_back( wm.ball().pos().x / sp.pitchHalfLength() );           // 15
    features.push_back( wm.ball().pos().y / sp.pitchHalfWidth() );            // 16
    features.push_back( wm.ball().vel().x / 3.0 );                            // 17
    features.push_back( wm.ball().vel().y / 3.0 );                            // 18
    features.push_back( wm.ball().vel().length() / 3.0 );                     // 19
    features.push_back( (int)wm.gameMode().type() / 20.0 );                   // 20 game mode
    features.push_back( ( wm.ourSide() == rcsc::LEFT ) ? 1.0 : -1.0 );       // 21 side
    features.push_back( wm.time().cycle() / 6000.0 );                         // 22 time normalized
    features.push_back( wm.ball().pos().dist( rcsc::Vector2D( -52.5, 0.0 ) ) / sp.pitchHalfLength() ); // 23
    features.push_back( wm.ball().pos().dist( rcsc::Vector2D( 52.5, 0.0 ) ) / sp.pitchHalfLength() );  // 24

    // --- Teammate features (22 per player, x11 = 242) ---
    for ( int unum = 2; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * mate = wm.ourPlayer( unum );
        double n = -52.5; // default pos if unknown
        double dash = 0, body = 0, face = 0, dist = 1;
        int count = 0;

        if ( mate && mate->posCount() < 3 )
        {
            rcsc::Vector2D rel = mate->pos() - wm.self().pos();
            n = std::atan2( rel.y, rel.x );
            dash = mate->vel().length() / 3.0;
            body = mate->body().radian() / rcsc::AngleDeg::PI;
            face = mate->face().radian() / rcsc::AngleDeg::PI;
            dist = mate->distFromSelf() / sp.pitchHalfLength();
            count = 1;
        }

        features.push_back( std::sin( n ) );     // angle_to_mate sin
        features.push_back( std::cos( n ) );     // angle_to_mate cos
        features.push_back( count );             // visible flag
        features.push_back( dist );              // relative distance
        features.push_back( mate ? mate->pos().x / sp.pitchHalfLength() : 0 );
        features.push_back( mate ? mate->pos().y / sp.pitchHalfWidth() : 0 );
        features.push_back( mate && mate->velCount() < 3 ? mate->vel().x / 3.0 : 0 );
        features.push_back( mate && mate->velCount() < 3 ? mate->vel().y / 3.0 : 0 );
        features.push_back( dash );
        features.push_back( body );
        features.push_back( face );
        features.push_back( mate ? mate->distFromBall() / sp.pitchHalfLength() : 1 );
        features.push_back( unum / 11.0 );       // unum
        features.push_back( ( unum >= 2 && unum <= 6 ) ? 1.0 : 0.0 );  // is_back
        features.push_back( ( unum >= 7 && unum <= 8 ) ? 1.0 : 0.0 );  // is_mid
        features.push_back( ( unum >= 9 ) ? 1.0 : 0.0 );               // is_fwd
    }

    // --- Opponent features (22 per player, x11 = 242) ---
    for ( int unum = 1; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * opp = wm.theirPlayer( unum );
        double n = -52.5;
        double dash = 0, body = 0, face = 0, dist = 1;
        int count = 0;

        if ( opp && opp->posCount() < 3 )
        {
            rcsc::Vector2D rel = opp->pos() - wm.self().pos();
            n = std::atan2( rel.y, rel.x );
            dash = opp->vel().length() / 3.0;
            body = opp->body().radian() / rcsc::AngleDeg::PI;
            face = opp->face().radian() / rcsc::AngleDeg::PI;
            dist = opp->distFromSelf() / sp.pitchHalfLength();
            count = 1;
        }

        features.push_back( std::sin( n ) );
        features.push_back( std::cos( n ) );
        features.push_back( count );
        features.push_back( dist );
        features.push_back( opp ? opp->pos().x / sp.pitchHalfLength() : 0 );
        features.push_back( opp ? opp->pos().y / sp.pitchHalfWidth() : 0 );
        features.push_back( opp && opp->velCount() < 3 ? opp->vel().x / 3.0 : 0 );
        features.push_back( opp && opp->velCount() < 3 ? opp->vel().y / 3.0 : 0 );
        features.push_back( dash );
        features.push_back( body );
        features.push_back( face );
        features.push_back( opp ? opp->distFromBall() / sp.pitchHalfLength() : 1 );
        // Extra features: danger indicators
        double goal_dist = opp ? opp->pos().dist( rcsc::Vector2D( -52.5, 0.0 ) ) : 100.0;
        features.push_back( goal_dist / sp.pitchHalfLength() );
        double ball_dist_opp = opp ? opp->pos().dist( wm.ball().pos() ) : 100.0;
        features.push_back( ball_dist_opp / sp.pitchHalfLength() );
        features.push_back( ball_dist_opp < 2.0 ? 1.0 : 0.0 ); // has_ball indicator
    }

    // --- Spatial context features ---
    // Distance to field zones
    features.push_back( wm.self().pos().x < -36.0 && std::fabs( wm.self().pos().y ) < 20.16 ? 1.0 : 0.0 ); // own penalty area
    features.push_back( wm.ball().pos().x < -36.0 && std::fabs( wm.ball().pos().y ) < 20.16 ? 1.0 : 0.0 ); // ball in own pen area
    features.push_back( wm.self().pos().x < 0.0 ? 1.0 : 0.0 );  // self in own half
    features.push_back( wm.ball().pos().x < 0.0 ? 1.0 : 0.0 );  // ball in own half
    features.push_back( wm.interceptTable().selfStep() / 30.0 );  // self intercept cycles
    features.push_back( wm.interceptTable().opponentStep() / 30.0 );  // opp intercept cycles
    features.push_back( wm.interceptTable().teammateStep() / 30.0 );  // mate intercept cycles

    // Pad or trim to maintain consistent size (target 738)
    // Current: 15 + 10 + 15*11 + 15*11 + 7 = 15 + 10 + 165 + 165 + 7 = 362
    // We need more features. Let's add granular spatial features
    for ( int unum = 2; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * mate = wm.ourPlayer( unum );
        features.push_back( mate ? ( mate->pos().x < 0.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( mate ? ( std::fabs( mate->pos().x ) < 10.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( mate ? ( std::fabs( mate->pos().y ) < 15.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( mate ? ( std::fabs( mate->pos().y ) > 25.0 ? 1.0 : 0.0 ) : 0.5 );
    }
    // + 40 = 402

    for ( int unum = 1; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * opp = wm.theirPlayer( unum );
        features.push_back( opp ? ( opp->pos().x < 0.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( opp ? ( std::fabs( opp->pos().x ) < 10.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( opp ? ( std::fabs( opp->pos().y ) < 15.0 ? 1.0 : 0.0 ) : 0.5 );
        features.push_back( opp ? ( std::fabs( opp->pos().y ) > 25.0 ? 1.0 : 0.0 ) : 0.5 );
    }
    // + 44 = 446

    // Ball trajectory features
    for ( int i = 0; i < 5; ++i )
    {
        // Predict ball pos after i*10 cycles
        rcsc::Vector2D ball_pred = wm.ball().pos() + wm.ball().vel() * ( i * 10.0 );
        features.push_back( ball_pred.x / sp.pitchHalfLength() );
        features.push_back( ball_pred.y / sp.pitchHalfWidth() );
    }
    // + 10 = 456

    // Self position relative to ball over time
    double rel_x = wm.self().pos().x - wm.ball().pos().x;
    double rel_y = wm.self().pos().y - wm.ball().pos().y;
    features.push_back( rel_x / sp.pitchHalfLength() );
    features.push_back( rel_y / sp.pitchHalfWidth() );
    double rel_dist = std::sqrt( rel_x * rel_x + rel_y * rel_y );
    features.push_back( rel_dist / 2.0 );
    features.push_back( rel_dist < 1.0 ? 1.0 : 0.0 ); // in kickable range
    features.push_back( std::atan2( rel_y, rel_x ) / rcsc::AngleDeg::PI );
    // + 5 = 461

    // Closest opponent direction and distance
    double closest_opp_dist = 100.0;
    double closest_opp_angle = 0.0;
    int closest_opp_unum = 0;
    for ( int unum = 1; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * opp = wm.theirPlayer( unum );
        if ( opp && opp->posCount() < 3 && opp->distFromSelf() < closest_opp_dist )
        {
            closest_opp_dist = opp->distFromSelf();
            rcsc::Vector2D rel = opp->pos() - wm.self().pos();
            closest_opp_angle = std::atan2( rel.y, rel.x );
            closest_opp_unum = opp->unum();
        }
    }
    features.push_back( closest_opp_dist / sp.pitchHalfWidth() );
    features.push_back( std::sin( closest_opp_angle ) );
    features.push_back( std::cos( closest_opp_angle ) );
    features.push_back( closest_opp_unum / 11.0 );
    // + 4 = 465

    // Second and third closest opponents
    double second_dist = 100.0, second_angle = 0.0;
    double third_dist = 100.0, third_angle = 0.0;
    for ( int unum = 1; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * opp = wm.theirPlayer( unum );
        if ( ! opp || ! opp->posCount() < 3 || opp->unum() == closest_opp_unum )
            continue;
        double d = opp->distFromSelf();
        if ( d < second_dist )
        {
            third_dist = second_dist; third_angle = second_angle;
            second_dist = d;
            rcsc::Vector2D rel = opp->pos() - wm.self().pos();
            second_angle = std::atan2( rel.y, rel.x );
        }
        else if ( d < third_dist )
        {
            third_dist = d;
            rcsc::Vector2D rel = opp->pos() - wm.self().pos();
            third_angle = std::atan2( rel.y, rel.x );
        }
    }
    features.push_back( second_dist / sp.pitchHalfWidth() );
    features.push_back( std::sin( second_angle ) );
    features.push_back( std::cos( second_angle ) );
    features.push_back( third_dist / sp.pitchHalfWidth() );
    features.push_back( std::sin( third_angle ) );
    features.push_back( std::cos( third_angle ) );
    // + 6 = 471

    // Dribble target angle relative features
    // Direction toward opponent goal
    rcsc::Vector2D to_goal = rcsc::Vector2D( 52.5, 0.0 ) - wm.self().pos();
    features.push_back( to_goal.length() / ( 2.0 * sp.pitchHalfLength() ) );
    features.push_back( std::sin( std::atan2( to_goal.y, to_goal.x ) ) );
    features.push_back( std::cos( std::atan2( to_goal.y, to_goal.x ) ) );
    // + 3 = 474

    // Free space indicator (angle range without opponents)
    features.push_back( 0.5 ); // placeholder: would need ray-casting for real
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    features.push_back( 0.5 );
    // + 8 = 482

    // Pad to 738 (target size from paper, actual may differ)
    while ( features.size() < 738 )
        features.push_back( 0.0 );

    if ( features.size() > 738 )
        features.resize( 738 );

    return features;
}

// ---------------------------------------------------------------------------
// DNN prediction (placeholder - needs CppDNN integration)
// ---------------------------------------------------------------------------

int
Bhv_MadDribble::predictBestAction( const rcsc::PlayerAgent * agent,
                                     const std::vector< double > & /*features*/ )
{
    // TODO: Integrate with CppDNN
    // 1. Load network from modelPath()
    // 2. Forward pass features
    // 3. Argmax over output classes (0-10)
    // 4. Return action with highest probability

    rcsc::dlog.addText( rcsc::Logger::TEAM,
                  __FILE__": DNN model loaded but prediction not yet integrated (returning 0)" );
    return 0; // fallback to normal dribble
}

// ---------------------------------------------------------------------------
// Turn deception
// ---------------------------------------------------------------------------

bool
Bhv_MadDribble::executeTurnDeception( rcsc::PlayerAgent * agent, int action_id )
{
    const rcsc::WorldModel & wm = agent->world();
    double target_angle = wm.self().body().degree();

    // Map action_id to turn offset
    switch ( action_id )
    {
    case 1:  target_angle = wm.self().body().degree() - 30.0; break;
    case 2:  target_angle = wm.self().body().degree() + 30.0; break;
    case 3:  target_angle = wm.self().body().degree() - 60.0; break;
    case 4:  target_angle = wm.self().body().degree() + 60.0; break;
    default: return false;
    }

    return Body_TurnToAngle( target_angle ).execute( agent );
}

// ---------------------------------------------------------------------------
// Move deception
// ---------------------------------------------------------------------------

bool
Bhv_MadDribble::executeMoveDeception( rcsc::PlayerAgent * agent, int action_id )
{
    const rcsc::WorldModel & wm = agent->world();
    rcsc::Vector2D ball_pos = wm.ball().pos();

    rcsc::Vector2D fake_target;

    if ( action_id == 5 )
    {
        // Move to ball's left side
        rcsc::AngleDeg ball_angle = ( wm.ball().pos() - wm.self().pos() ).th();
        fake_target = wm.ball().pos() + rcsc::Vector2D::polar2vector( 1.5, ball_angle + 90.0 );
    }
    else if ( action_id == 6 )
    {
        // Move to ball's right side
        rcsc::AngleDeg ball_angle = ( wm.ball().pos() - wm.self().pos() ).th();
        fake_target = wm.ball().pos() + rcsc::Vector2D::polar2vector( 1.5, ball_angle - 90.0 );
    }
    else
    {
        return false;
    }

    return Body_GoToPoint( fake_target, 0.5,
                                  rcsc::ServerParam::i().maxDashPower(), 1 ).execute( agent );
}

// ---------------------------------------------------------------------------
// Training data collection
// ---------------------------------------------------------------------------

void
Bhv_MadDribble::collectTrainingData( const rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();

    std::vector< double > features = extractFeatures( agent );

    // Create data directory if needed (only once, lazily)
    static bool dir_created = false;
    if ( ! dir_created )
    {
        mkdir( DATA_LOG_DIR.c_str(), 0755 );
        dir_created = true;
    }

    // Generate filename based on time
    time_t now = time( nullptr );
    struct tm * t = localtime( & now );
    char buf[128];
    strftime( buf, sizeof(buf), "%Y%m%d_%H", t );

    std::string filename = DATA_LOG_DIR + "mad_features_" + std::string( buf ) + ".csv";
    static int line_count = 0;

    // Open file and append (every cycle that MAD activates, write a row)
    std::ofstream ofs( filename, std::ios::app );
    if ( ! ofs.is_open() )
        return;

    // Format: cycle,unum,feature1,feature2,...,feature738,LABEL\n
    // LABEL is -1 (to be filled by post-processing after game)
    ofs << wm.time().cycle() << "," << wm.self().unum();
    for ( const auto & f : features )
        ofs << "," << std::fixed << std::setprecision(6) << f;
    ofs << ",-1" << std::endl; // label placeholder

    line_count++;

    // Log periodically
    if ( line_count % 100 == 0 )
    {
        rcsc::dlog.addText( rcsc::Logger::TEAM,
                      __FILE__": MAD training data collected %d rows -> %s",
                      line_count, filename.c_str() );
    }
}

// ---------------------------------------------------------------------------
// Model path and loaded status
// ---------------------------------------------------------------------------

const std::string &
Bhv_MadDribble::modelPath()
{
    static const std::string path = MODEL_DIR + MODEL_FILENAME;
    return path;
}

bool
Bhv_MadDribble::modelLoaded()
{
    if ( ! s_model_loaded )
        s_model_loaded = checkModelFile();
    return s_model_loaded;
}
