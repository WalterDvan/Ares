/***************************************************************************
 *  bhv_pass_data_collector.cpp - Pass Prediction Data Collector
 *
 *  Passive data collection module. Hooks into action() each cycle.
 *  When a pass event is detected (holder change), logs features + label.
 *
 *  Output: pass_training_data/pass_features_YYYYMMDD_HH.csv
 *  Columns: cycle, passer_unum, receiver_unum, event_type, f1..fN
 ****************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "bhv_pass_data_collector.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/player_object.h>
#include <rcsc/player/world_model.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

// Static state tracking
bool Bhv_PassDataCollector::s_initialized = false;
int  Bhv_PassDataCollector::s_prev_ball_holder_unum = -1;
bool Bhv_PassDataCollector::s_prev_was_kickable = false;
int  Bhv_PassDataCollector::s_prev_cycle = -1;

static const std::string DATA_LOG_DIR = "pass_training_data/";

// ---------------------------------------------------------------------------
// Execute (called every cycle from action())
// ---------------------------------------------------------------------------

bool
Bhv_PassDataCollector::execute( rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();

    // Initialize
    if ( ! s_initialized )
    {
        s_prev_ball_holder_unum = -1;
        s_prev_was_kickable = false;
        s_prev_cycle = -1;

        ensureDir();
        s_initialized = true;
    }

    // Detect consecutive cycles (skip if same cycle or gap > 2)
    if ( s_prev_cycle >= 0 && wm.time().cycle() - s_prev_cycle > 2 )
    {
        s_prev_ball_holder_unum = -1;
        s_prev_was_kickable = false;
    }
    s_prev_cycle = wm.time().cycle();

    // Determine current ball holder
    int current_holder = -1;
    bool is_kickable = wm.self().isKickable();

    if ( is_kickable )
        current_holder = wm.self().unum();
    else
    {
        // Check teammates
        const rcsc::PlayerObject::Cont & mates = wm.teammates();
        for ( const auto & m : mates )
        {
            if ( m->isKickable() && m->unum() > 0 )
            {
                current_holder = m->unum();
                break;
            }
        }
        // Check opponents
        if ( current_holder < 0 )
        {
            const rcsc::PlayerObject::Cont & opps = wm.opponents();
            for ( const auto & o : opps )
            {
                if ( o->isKickable() && o->unum() > 0 )
                {
                    current_holder = -2; // opponent holds ball
                    break;
                }
            }
        }
    }

    // Detect pass event
    int event_type = 0;
    int passing_unum = -1;
    int receiving_unum = -1;

    if ( s_prev_ball_holder_unum > 0 && current_holder > 0
         && s_prev_ball_holder_unum != current_holder )
    {
        // Our player passed to our player -> successful pass
        event_type = 1;
        passing_unum = s_prev_ball_holder_unum;
        receiving_unum = current_holder;
    }
    else if ( s_prev_ball_holder_unum > 0 && current_holder == -2 )
    {
        // Our player passed but opponent intercepted
        event_type = 2;
        passing_unum = s_prev_ball_holder_unum;
    }
    else if ( s_prev_ball_holder_unum > 0 && current_holder == -1
              && !wm.ball().vel().r() ) // ball stopped, no one has it
    {
        // Ball went out of play or lost
        event_type = 3;
        passing_unum = s_prev_ball_holder_unum;
    }

    s_prev_ball_holder_unum = current_holder;
    s_prev_was_kickable = is_kickable;

    if ( event_type == 0 )
        return false;

    // Extract features and write training example
    std::vector< double > features = extractPassFeatures( agent );
    writeExample( agent, features, event_type );

    rcsc::dlog.addText( rcsc::Logger::TEAM,
                  __FILE__": Pass event type=%d passer=%d receiver=%d",
                  event_type, passing_unum, receiving_unum );

    return true; // data was collected
}

// ---------------------------------------------------------------------------
// Feature extraction for pass prediction
// ---------------------------------------------------------------------------

std::vector< double >
Bhv_PassDataCollector::extractPassFeatures( const rcsc::PlayerAgent * agent )
{
    const rcsc::WorldModel & wm = agent->world();
    const rcsc::ServerParam & sp = rcsc::ServerParam::i();
    std::vector< double > features;

    // Self state (15)
    features.push_back( wm.self().pos().x / sp.pitchHalfLength() );
    features.push_back( wm.self().pos().y / sp.pitchHalfWidth() );
    features.push_back( wm.self().vel().x / 3.0 );
    features.push_back( wm.self().vel().y / 3.0 );
    features.push_back( wm.self().body().radian() / rcsc::AngleDeg::PI );
    features.push_back( wm.self().face().radian() / rcsc::AngleDeg::PI );
    features.push_back( wm.self().stamina() / sp.staminaMax() );
    features.push_back( wm.self().effort() );
    features.push_back( wm.self().recovery() );

    // Ball state (10)
    features.push_back( wm.ball().pos().x / sp.pitchHalfLength() );
    features.push_back( wm.ball().pos().y / sp.pitchHalfWidth() );
    features.push_back( wm.ball().vel().x / 3.0 );
    features.push_back( wm.ball().vel().y / 3.0 );
    features.push_back( wm.ball().vel().length() / 3.0 );

    // Each teammate (10 per player x 10 = 100)
    for ( int unum = 2; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * m = wm.ourPlayer( unum );
        if ( m && m->posCount() < 3 )
        {
            features.push_back( m->pos().x / sp.pitchHalfLength() );
            features.push_back( m->pos().y / sp.pitchHalfWidth() );
            features.push_back( m->vel().x / 3.0 );
            features.push_back( m->vel().y / 3.0 );
            features.push_back( m->body().radian() / rcsc::AngleDeg::PI );
            features.push_back( m->distFromBall() / sp.pitchHalfLength() );
            features.push_back( m->distFromSelf() / sp.pitchHalfLength() );
            features.push_back( (unum >= 2 && unum <= 6) ? 1.0 : 0.0 ); // is_back
            features.push_back( (unum >= 7 && unum <= 8) ? 1.0 : 0.0 ); // is_mid
            features.push_back( m->distFromBall() < sp.defaultKickableArea() ? 1.0 : 0.0 );  // has_ball
        }
        else
        {
            for ( int j = 0; j < 10; ++j ) features.push_back( 0.0 );
        }
    }

    // Each opponent (10 per player x 11 = 110)
    for ( int unum = 1; unum <= 11; ++unum )
    {
        const rcsc::AbstractPlayerObject * o = wm.theirPlayer( unum );
        if ( o && o->posCount() < 3 )
        {
            features.push_back( o->pos().x / sp.pitchHalfLength() );
            features.push_back( o->pos().y / sp.pitchHalfWidth() );
            features.push_back( o->vel().x / 3.0 );
            features.push_back( o->vel().y / 3.0 );
            features.push_back( o->body().radian() / rcsc::AngleDeg::PI );
            features.push_back( o->distFromBall() / sp.pitchHalfLength() );
            features.push_back( o->distFromSelf() / sp.pitchHalfLength() );
            features.push_back( o->pos().x < -30.0 ? 1.0 : 0.0 );
            features.push_back( o->pos().dist( rcsc::Vector2D(-52.5,0) ) / sp.pitchHalfLength() );
            features.push_back( o->distFromBall() < sp.defaultKickableArea() ? 1.0 : 0.0 );
        }
        else
        {
            for ( int j = 0; j < 10; ++j ) features.push_back( 0.0 );
        }
    }

    // Game state (5)
    features.push_back( (int)wm.gameMode().type() / 20.0 );
    features.push_back( ( wm.ourSide() == rcsc::LEFT ) ? 1.0 : -1.0 );
    features.push_back( wm.time().cycle() / 6000.0 );
    features.push_back( wm.self().pos().dist( rcsc::Vector2D(52.5,0) ) / sp.pitchHalfLength() );
    features.push_back( wm.ball().pos().dist( rcsc::Vector2D(-52.5,0) ) / sp.pitchHalfLength() );

    // Clip to reasonable size
    while ( features.size() < 300 ) features.push_back( 0.0 );
    if ( features.size() > 300 ) features.resize( 300 );

    return features;
}

// ---------------------------------------------------------------------------
// Write example to CSV
// ---------------------------------------------------------------------------

void
Bhv_PassDataCollector::writeExample( const rcsc::PlayerAgent * agent,
                                       const std::vector< double > & features,
                                       int label )
{
    const rcsc::WorldModel & wm = agent->world();

    time_t now = time( nullptr );
    struct tm * t = localtime( & now );
    char buf[64];
    strftime( buf, sizeof(buf), "%Y%m%d_%H", t );

    std::string filename = DATA_LOG_DIR + "pass_features_" + std::string( buf ) + ".csv";

    std::ofstream ofs( filename, std::ios::app );
    if ( ! ofs.is_open() )
        return;

    ofs << wm.time().cycle();
    for ( const auto & f : features )
        ofs << "," << std::fixed << std::setprecision(6) << f;
    ofs << "," << label << std::endl;
}

// ---------------------------------------------------------------------------
// Ensure data directory exists
// ---------------------------------------------------------------------------

void
Bhv_PassDataCollector::ensureDir()
{
    mkdir( DATA_LOG_DIR.c_str(), 0755 );
}
