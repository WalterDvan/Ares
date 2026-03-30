/***************************************************************************
 *  bhv_pass_data_collector.h - Pass Prediction Data Collector
 *
 *  Based on CYRUS 2021 paper: "Improving Dribbling, Passing, and Marking
 *  Actions in Soccer Simulation 2D Games Using Machine Learning"
 *  (arXiv:2401.03406)
 *
 *  Collects training data for DNN-based pass decision/prediction:
 *  - When a player receives a pass: log the game state features + pass outcome
 *  - When a player makes a pass: log features + target receiver + outcome
 *  - Features follow the paper's specification
 *  - Labels: pass success (1) or failure (0), pass direction bins
 *
 *  This is a passive observer that hooks into the action chain.
 *  It does NOT change any behavior - only writes data files.
 ****************************************************************************/

#ifndef BHV_PASS_DATA_COLLECTOR_H
#define BHV_PASS_DATA_COLLECTOR_H

#include <rcsc/player/player_agent.h>
#include <vector>
#include <string>

class Bhv_PassDataCollector {
public:

    /*!
     * \brief Call every cycle to check if a relevant pass event occurred.
     * Returns true if data was collected this cycle.
     * Passive: never affects player behavior.
     */
    static bool execute( rcsc::PlayerAgent * agent );

private:

    //! Track previous cycle state for event detection
    static bool s_initialized;
    static int  s_prev_ball_holder_unum;  // -1 = no one, 0 = opponent, 1-11 = our player
    static bool s_prev_was_kickable;
    static int  s_prev_cycle;

    //! Extract features for pass prediction (similar structure to MAD)
    static std::vector< double > extractPassFeatures( const rcsc::PlayerAgent * agent );

    /*!
     * \brief Detect if a pass was just completed (ball changed holder)
     * Returns: {event_type, passing_unum, receiving_unum}
     * event_type: 0=none, 1=successful_pass, 2=intercepted, 3=bad_pass
     */
    static void detectPassEvent( const rcsc::PlayerAgent * agent,
                                  int & event_type,
                                  int & passing_unum,
                                  int & receiving_unum );

    //! Write a training example to CSV
    static void writeExample( const rcsc::PlayerAgent * agent,
                               const std::vector< double > & features,
                               int label );

    static void ensureDir();
};

#endif
