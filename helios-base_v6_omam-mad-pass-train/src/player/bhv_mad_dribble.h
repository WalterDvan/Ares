/***************************************************************************
 *  bhv_mad_dribble.h - Multi Action Dribble (MAD)
 *
 *  Based on CYRUS 2021 paper: "Improving Dribbling, Passing, and Marking
 *  Actions in Soccer Simulation 2D Games Using Machine Learning"
 *  (arXiv:2401.03406)
 *
 *  MAD adds a deception step BEFORE normal dribble:
 *  1. Two-Step Kick: kick ball to another position in kickable area
 *  2. Move Before First Kick: move around the ball first
 *  3. Turn Before First Kick: face a different direction first
 *
 *  When DNN model is available: use DNN to predict opponent reactions
 *  and select the optimal deception action.
 *
 *  Without DNN (current): collect training data from log files.
 *  Falls back to basic dribble when no DNN model is loaded.
 ****************************************************************************/

#ifndef BHV_MAD_DRIBBLE_H
#define BHV_MAD_DRIBBLE_H

#include <rcsc/player/player_agent.h>
#include <rcsc/geom/vector_2d.h>
#include <rcsc/geom/angle_deg.h>
#include <vector>
#include <string>

class Bhv_MadDribble {
public:

    /*!
     * \brief Try MAD dribble. Returns true if executed, false to fallback.
     * Only activates when:
     * - Our player has the ball (is kicker)
     * - There are opponents nearby within threat radius
     * - DNN model is loaded OR data collection mode is on
     */
    static bool execute( rcsc::PlayerAgent * agent );

private:

    //! Check if MAD should activate
    static bool shouldActivate( const rcsc::PlayerAgent * agent );

    //! Get list of nearby opponents (within threat radius ~10m)
    static void getNearbyOpponents( const rcsc::PlayerAgent * agent,
                                     std::vector< int > & opp_unums );

    //! Extract 738 features from world state for DNN input
    static std::vector< double > extractFeatures( const rcsc::PlayerAgent * agent );

    //! DNN prediction: given features, predict best action (0-10)
    static int predictBestAction( const rcsc::PlayerAgent * agent,
                                   const std::vector< double > & features );

    //! Execute "Turn Before First Kick" deception
    static bool executeTurnDeception( rcsc::PlayerAgent * agent, int action_id );

    //! Execute "Move Before First Kick" deception
    static bool executeMoveDeception( rcsc::PlayerAgent * agent, int action_id );

    //! Collect training features and write to log file
    static void collectTrainingData( const rcsc::PlayerAgent * agent );

    //! Path to DNN model file (empty = no DNN, data collection mode)
    static const std::string & modelPath();
    static bool modelLoaded();
};

#endif
