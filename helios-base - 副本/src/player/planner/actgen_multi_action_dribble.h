// -*-c++-*-

#ifndef ACTGEN_MULTI_ACTION_DRIBBLE_H
#define ACTGEN_MULTI_ACTION_DRIBBLE_H

#include "action_generator.h"

class ActGen_MultiActionDribble
    : public ActionGenerator {
public:
    virtual
    void generate( std::vector< ActionStatePair > * result,
                   const PredictState & state,
                   const rcsc::WorldModel & wm,
                   const std::vector< ActionStatePair > & path ) const;
};

#endif
