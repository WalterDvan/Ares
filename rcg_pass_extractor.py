#!/usr/bin/env python3
"""
RCG Pass Training Data Extractor
=================================
Extracts pass training features from rcssserver RCG v4/v5/v6 game logs
without modifying the player source code.

Usage:
    python rcg_pass_extractor.py <rcg_file_or_dir> [output_dir]

Detection logic:
  - Track ball possession by proximity (within kickable zone)
  - When ball transitions from one player to teammate = successful pass
  - When ball transitions to opponent = intercepted pass
  - When ball goes out of play or nobody controls it = dead ball

Output: CSV file with one row per detected pass event, binary label (1=success, 2=intercepted).
"""

import sys
import os
import re
import csv
import math
import glob
import struct
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    side: str         # 'l' or 'r'
    unum: int         # uniform number 1-11
    ptype: int        # player type id
    state: int        # hex state flags
    x: float          # position x
    y: float          # position y
    vx: float         # velocity x
    vy: float         # velocity y
    body: float       # body angle (degrees)
    neck: float       # neck angle relative to body (degrees)
    stamina: float = 0.0
    effort: float = 0.0
    recovery: float = 0.0
    # command counts
    kick_count: int = 0
    dash_count: int = 0

    @property
    def team_id(self) -> str:
        return self.side

    @property
    def full_id(self) -> str:
        return f"{self.side}{self.unum}"


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class ShowFrame:
    time: int         # game cycle
    playmode: str
    ball: BallState
    players: List[PlayerState]
    team_l_name: str = ''
    team_r_name: str = ''
    score_l: int = 0
    score_r: int = 0


@dataclass
class ServerParams:
    player_size: float = 0.3
    kickable_margin: float = 0.7
    control_radius: float = 2.0
    ball_size: float = 0.085
    ball_decay: float = 0.94
    ball_speed_max: float = 3.0
    player_speed_max: float = 1.05
    pitch_half_length: float = 52.5
    pitch_half_width: float = 34.0
    goal_width: float = 14.02
    max_dash_power: float = 100.0
    dash_power_rate: float = 0.006
    kick_power_rate: float = 0.027
    player_rand: float = 0.1


@dataclass
class PassEvent:
    """Detected pass event with features for ML training."""
    # Metadata
    match_file: str = ''
    cycle: int = 0
    half: int = 1  # 1st or 2nd half

    # Label: 1=successful, 2=intercepted
    label: int = 0

    # Kicker info
    kicker_side: str = ''
    kicker_unum: int = 0
    kicker_x: float = 0.0
    kicker_y: float = 0.0
    kicker_body: float = 0.0
    kicker_vx: float = 0.0
    kicker_vy: float = 0.0
    kicker_stamina: float = 0.0
    kicker_effort: float = 0.0
    kicker_dist_to_goal: float = 0.0  # distance to opponent goal center
    kicker_dist_to_own_goal: float = 0.0

    # Receiver info (if label=1)
    receiver_side: str = ''
    receiver_unum: int = 0
    receiver_x: float = 0.0
    receiver_y: float = 0.0
    receiver_vx: float = 0.0
    receiver_vy: float = 0.0
    receiver_stamina: float = 0.0

    # Ball info
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_vx: float = 0.0
    ball_vy: float = 0.0
    ball_speed: float = 0.0

    # Spatial features
    pass_distance: float = 0.0    # distance from kicker to ball landing/receiver
    pass_angle: float = 0.0       # angle of ball velocity relative to kicker's body
    pass_direction_x: float = 0.0  # normalized direction from kicker to receiver
    pass_direction_y: float = 0.0
    forward_pass: int = 0         # 1 if pass goes toward opponent goal

    # Opponent features
    nearest_opp_dist: float = 0.0
    nearest_opp_x: float = 0.0
    nearest_opp_y: float = 0.0
    # Number of opponents in a cone between kicker and receiver
    opp_in_pass_lane: int = 0
    opp_min_dist_to_pass_line: float = 0.0

    # Teammate features
    nearby_teammate_count: int = 0
    nearest_teammate_support_dist: float = 0.0

    # Pressure
    opp_before_kicker_dist: List[float] = field(default_factory=list)
    opp_density_10m: int = 0  # opponents within 10m of kicker


# ---------------------------------------------------------------------------
# RCG Parser
# ---------------------------------------------------------------------------

class RCGParser:
    """Parse RCG v4/v5/v6 text format."""

    def __init__(self):
        self.server_params = ServerParams()
        self.frames: List[ShowFrame] = []
        self.playmode_map: Dict[int, str] = {}  # time -> playmode

    def parse(self, filepath: str) -> List[ShowFrame]:
        """Parse an RCG file and return list of ShowFrame."""
        self.frames = []

        # Handle gzipped files
        if filepath.endswith('.gz'):
            import gzip
            fh = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
        else:
            fh = open(filepath, 'r', encoding='utf-8', errors='replace')

        try:
            lines = fh.readlines()
        finally:
            fh.close()

        if not lines:
            return []

        # Check header
        header = lines[0].strip()
        if not header.startswith('ULG'):
            # Might be old binary format - skip
            print(f"  Skipping {filepath}: unsupported format ({header[:20]})")
            return []

        version = int(header[3:])
        if version not in (4, 5, 6):
            print(f"  Skipping {filepath}: unsupported version {version}")
            return []

        # Parse header lines (server_param, player_param, player_type)
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith('(server_param'):
                self._parse_server_param(stripped)
            elif stripped.startswith('(player_type'):
                pass  # player types parsed on demand
            elif stripped.startswith('(show'):
                frame = self._parse_show(stripped)
                if frame:
                    self.frames.append(frame)

        return self.frames

    def _parse_server_param(self, line: str):
        """Extract server params."""
        for m in re.finditer(r'\((\w+)\s+([^\)]+)\)', line):
            key = m.group(1)
            val_str = m.group(2).strip()
            try:
                val = float(val_str)
            except ValueError:
                continue
            if hasattr(self.server_params, key):
                setattr(self.server_params, key, val)

    def _parse_show(self, line: str) -> Optional[ShowFrame]:
        """Parse a (show ...) line."""
        try:
            # Format: (show <time> ((b) <bx> <by> <bvx> <bvy>) ...players...)
            # May also contain team info and playmode

            # Use regex to extract components
            m = re.match(r'\(show\s+(\d+)', line)
            if not m:
                return None
            time = int(m.group(1))

            # Extract ball
            ball_match = re.search(r'\(\(b\)\s+([^)]+)\)', line)
            if not ball_match:
                return None
            ball_parts = ball_match.group(1).split()
            ball = BallState(
                x=float(ball_parts[0]),
                y=float(ball_parts[1]),
                vx=float(ball_parts[2]),
                vy=float(ball_parts[3])
            )

            # Extract team info if present
            team_l_name = ''
            team_r_name = ''
            score_l = 0
            score_r = 0
            team_match = re.search(r'\(tm\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)', line)
            if team_match:
                team_l_name = team_match.group(1)
                team_r_name = team_match.group(2)
                score_l = int(team_match.group(3))
                score_r = int(team_match.group(4))

            # Extract players - each player looks like:
            # ((l <unum>) <type> <state> <x> <y> <vx> <vy> <body> <neck> ...)
            players = []
            player_pattern = re.finditer(
                r'\(\([lr]\s+(\d+)\)\s+(\d+)\s+(0x[0-9a-fA-F]+|\d+)\s+'
                r'([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+'
                r'([-0-9.]+)\s+([-0-9.]+)',
                line
            )
            for pm in player_pattern:
                side = pm.group(0)[2]  # 'l' or 'r' from '((l'
                unum = int(pm.group(1))
                ptype = int(pm.group(2))
                state = int(pm.group(3), 16) if pm.group(3).startswith('0x') else int(pm.group(3))
                x = float(pm.group(4))
                y = float(pm.group(5))
                vx = float(pm.group(6))
                vy = float(pm.group(7))
                body = float(pm.group(8))
                neck = float(pm.group(9))

                players.append(PlayerState(
                    side=side, unum=unum, ptype=ptype, state=state,
                    x=x, y=y, vx=vx, vy=vy, body=body, neck=neck
                ))

            # Determine playmode from stored playmode_map
            playmode = self.playmode_map.get(time, 'play_on')

            frame = ShowFrame(
                time=time, playmode=playmode,
                ball=ball, players=players,
                team_l_name=team_l_name, team_r_name=team_r_name,
                score_l=score_l, score_r=score_r
            )
            return frame

        except Exception as e:
            return None


def parse_playmode_line(line: str) -> Tuple[int, str]:
    """Parse (playmode <time> <mode_name>) line."""
    m = re.match(r'\(playmode\s+(\d+)\s+(\S+)\)', line)
    if m:
        return int(m.group(1)), m.group(2).rstrip(')')
    return -1, ''


def dist(x1, y1, x2, y2) -> float:
    """Euclidean distance."""
    return math.hypot(x2 - x1, y2 - y1)


def angle_diff_deg(a, b) -> float:
    """Angle difference in degrees, normalized to [-180, 180]."""
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return d


def point_to_line_dist(px, py, x1, y1, x2, y2) -> float:
    """Point-to-line-segment distance."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-8:
        return dist(px, py, x1, y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


# ---------------------------------------------------------------------------
# Pass Detector
# ---------------------------------------------------------------------------

class PassDetector:
    """Detect pass events from parsed RCG frames."""

    # Playmodes where play is active (ball is in play)
    ACTIVE_PLAYMODES = {
        'play_on', 'kick_in_l', 'kick_in_r',
        'free_kick_l', 'free_kick_r',
        'corner_kick_l', 'corner_kick_r',
        'goal_kick_l', 'goal_kick_r',
        'indirect_free_kick_l', 'indirect_free_kick_r',
        'offside_l', 'offside_r',
    }

    STOP_PLAYMODES = {
        'before_kick_off', 'time_over', 'goal_l', 'goal_r',
        'first_half_over', 'pause', 'half_time',
        'penalty_setup_l', 'penalty_setup_r',
        'penalty_ready_l', 'penalty_ready_r',
        'foul_charge_l', 'foul_charge_r',
        'foul_push_l', 'foul_push_r',
        'back_pass_l', 'back_pass_r',
        'catch_fault_l', 'catch_fault_r',
    }

    def __init__(self, server_params: ServerParams):
        self.sp = server_params
        self.kickable_dist = server_params.player_size + server_params.kickable_margin + 0.1

    def find_possessor(self, frame: ShowFrame) -> Optional[PlayerState]:
        """Find the player closest to the ball within kickable range."""
        bx, by = frame.ball.x, frame.ball.y
        best = None
        best_dist = self.kickable_dist

        for p in frame.players:
            d = dist(bx, by, p.x, p.y)
            if d < best_dist:
                best_dist = d
                best = p
        return best

    def detect_passes(self, frames: List[ShowFrame], match_file: str = '') -> List[PassEvent]:
        """Detect pass events from a sequence of frames."""
        events: List[PassEvent] = []

        # Track ball possession over time
        # possessor: (side, unum) or None
        prev_possessor = None
        possess_start_frame = None
        pass_ball_vx = 0.0
        pass_ball_vy = 0.0
        kick_frame = None  # frame where ball was kicked
        kick_ball_speed = 0.0
        min_pass_travel = 2.0  # minimum ball travel distance for a pass (meters)
        max_pass_duration = 60  # max cycles to wait for pass outcome

        playmode_map: Dict[int, str] = {}

        # Build playmode map from frames
        # (We update this externally via parse_playmode_line, but also
        #  handle the case where playmodes are embedded in show frames in some formats)

        for i, frame in enumerate(frames):
            pm = frame.playmode

            # Check for ball stop playmodes - reset possession
            if pm in self.STOP_PLAYMODES:
                prev_possessor = None
                possess_start_frame = None
                kick_frame = None
                continue

            if pm not in self.ACTIVE_PLAYMODES:
                continue

            possessor = self.find_possessor(frame)
            poss_id = (possessor.side, possessor.unum) if possessor else None

            if prev_possessor is not None and kick_frame is not None:
                cycles_since_kick = frame.time - kick_frame.time

                if cycles_since_kick > max_pass_duration:
                    # Timeout - no clear receiver, likely a dribble or lost ball
                    kick_frame = None
                    prev_possessor = poss_id if poss_id else prev_possessor
                    continue

                # Check if a new player gained possession
                if poss_id is not None and poss_id != prev_possessor:
                    kick_side, kick_unum = prev_possessor
                    recv_side, recv_unum = poss_id

                    # Find kicker frame data
                    kicker_state = None
                    for p in kick_frame.players:
                        if p.side == kick_side and p.unum == kick_unum:
                            kicker_state = p
                            break

                    ball_travel = dist(kick_frame.ball.x, kick_frame.ball.y,
                                       frame.ball.x, frame.ball.y)

                    # Only count as a pass if ball traveled far enough
                    if ball_travel >= min_pass_travel:
                        evt = PassEvent()
                        evt.match_file = os.path.basename(match_file)
                        evt.cycle = kick_frame.time
                        evt.half = 1 if kick_frame.time <= 3000 else 2

                        # Kicker info
                        evt.kicker_side = kick_side
                        evt.kicker_unum = kick_unum
                        if kicker_state:
                            evt.kicker_x = kicker_state.x
                            evt.kicker_y = kicker_state.y
                            evt.kicker_body = kicker_state.body
                            evt.kicker_vx = kicker_state.vx
                            evt.kicker_vy = kicker_state.vy
                            evt.kicker_stamina = kicker_state.stamina
                            evt.kicker_effort = kicker_state.effort

                        # Ball info (at kick moment)
                        evt.ball_x = kick_frame.ball.x
                        evt.ball_y = kick_frame.ball.y
                        evt.ball_vx = kick_frame.ball.vx
                        evt.ball_vy = kick_frame.ball.vy
                        evt.ball_speed = kick_frame.ball.speed

                        # Pass geometry
                        evt.pass_distance = ball_travel
                        evt.pass_direction_x = (frame.ball.x - kick_frame.ball.x) / ball_travel if ball_travel > 0 else 0
                        evt.pass_direction_y = (frame.ball.y - kick_frame.ball.y) / ball_travel if ball_travel > 0 else 0

                        # Pass angle relative to kicker body
                        if kicker_state and evt.ball_speed > 0.1:
                            kick_angle = math.atan2(kick_frame.ball.vy, kick_frame.ball.vx)
                            body_rad = math.radians(kicker_state.body)
                            evt.pass_angle = math.degrees(kick_angle - body_rad)
                        else:
                            evt.pass_angle = 0

                        # Forward pass check
                        if kick_side == 'l':
                            evt.forward_pass = 1 if (frame.ball.x > kick_frame.ball.x + 1.0) else 0
                        else:
                            evt.forward_pass = 1 if (frame.ball.x < kick_frame.ball.x - 1.0) else 0

                        # Goal distances
                        if kick_side == 'l':
                            evt.kicker_dist_to_goal = dist(kicker_state.x if kicker_state else 0,
                                                           kicker_state.y if kicker_state else 0,
                                                           self.sp.pitch_half_length, 0)
                            evt.kicker_dist_to_own_goal = dist(kicker_state.x if kicker_state else 0,
                                                               kicker_state.y if kicker_state else 0,
                                                               -self.sp.pitch_half_length, 0)
                        else:
                            evt.kicker_dist_to_goal = dist(kicker_state.x if kicker_state else 0,
                                                           kicker_state.y if kicker_state else 0,
                                                           -self.sp.pitch_half_length, 0)
                            evt.kicker_dist_to_own_goal = dist(kicker_state.x if kicker_state else 0,
                                                               kicker_state.y if kicker_state else 0,
                                                               self.sp.pitch_half_length, 0)

                        # Label
                        if recv_side == kick_side:
                            evt.label = 1  # successful pass - same team
                            evt.receiver_side = recv_side
                            evt.receiver_unum = recv_unum
                            for p in frame.players:
                                if p.side == recv_side and p.unum == recv_unum:
                                    evt.receiver_x = p.x
                                    evt.receiver_y = p.y
                                    evt.receiver_vx = p.vx
                                    evt.receiver_vy = p.vy
                                    evt.receiver_stamina = p.stamina
                                    break
                        else:
                            evt.label = 2  # intercepted - opponent got it

                        # Opponent pressure features at kick moment
                        opp_players = [p for p in kick_frame.players if p.side != kick_side]
                        own_players = [p for p in kick_frame.players
                                       if p.side == kick_side and not (p.side == kick_side and p.unum == kick_unum)]

                        # Nearest opponent
                        if kicker_state:
                            opp_dists = [dist(kicker_state.x, kicker_state.y, op.x, op.y) for op in opp_players]
                            if opp_dists:
                                min_idx = opp_dists.index(min(opp_dists))
                                evt.nearest_opp_dist = opp_dists[min_idx]
                                evt.nearest_opp_x = opp_players[min_idx].x
                                evt.nearest_opp_y = opp_players[min_idx].y

                                # All opponent distances (up to 11)
                                opp_dists_sorted = sorted(opp_dists)
                                evt.opp_before_kicker_dist = opp_dists_sorted[:11]

                                # Opponents within 10m
                                evt.opp_density_10m = sum(1 for d in opp_dists if d < 10.0)

                        # Opponents in pass lane (cone from kicker to receiver)
                        if kicker_state and ball_travel > 0:
                            lane_width = 2.0  # meters half-width
                            for op in opp_players:
                                d = point_to_line_dist(op.x, op.y,
                                                       kick_frame.ball.x, kick_frame.ball.y,
                                                       frame.ball.x, frame.ball.y)
                                if d < lane_width:
                                    evt.opp_in_pass_lane += 1

                                # Min dist to pass line
                                d_line = point_to_line_dist(op.x, op.y,
                                                            kick_frame.ball.x, kick_frame.ball.y,
                                                            frame.ball.x, frame.ball.y)
                                if d_line < evt.opp_min_dist_to_pass_line or evt.opp_min_dist_to_pass_line == 0:
                                    evt.opp_min_dist_to_pass_line = d_line

                        # Nearby teammates
                        if kicker_state:
                            tm_dists = [dist(kicker_state.x, kicker_state.y, tm.x, tm.y) for tm in own_players]
                            tm_dists.sort()
                            evt.nearby_teammate_count = sum(1 for d in tm_dists if d < 15.0)
                            if tm_dists:
                                evt.nearest_teammate_support_dist = tm_dists[0]

                        events.append(evt)

                    # Update possession
                    prev_possessor = poss_id
                    kick_frame = None
                elif poss_id is not None and poss_id == prev_possessor:
                    # Same player still has ball - might be dribbling
                    kick_frame = None  # reset, not a pass

            elif poss_id is not None:
                # New possessor (or continuing) without pending kick
                if prev_possessor is not None and prev_possessor != poss_id:
                    # Someone else got the ball but no kick was tracked
                    pass  # transition without detected kick - skip
                prev_possessor = poss_id
                kick_frame = None
            else:
                # No one has ball
                if prev_possessor is not None and kick_frame is None:
                    # Ball was just released (kicked) - mark kick frame
                    kick_frame = frame
                    # Ball should be moving away from last possessor
                    if prev_possessor:
                        last_pos = None
                        for p in frame.players:
                            if p.side == prev_possessor[0] and p.unum == prev_possessor[1]:
                                last_pos = p
                                break
                        if last_pos and frame.ball.speed < 0.3:
                            # Ball barely moving - maybe not a kick
                            kick_frame = None
                            prev_possessor = None
                    else:
                        prev_possessor = None

        return events


# ---------------------------------------------------------------------------
# CSV Output
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    'match_file', 'cycle', 'half', 'label',
    'kicker_side', 'kicker_unum', 'kicker_x', 'kicker_y', 'kicker_body',
    'kicker_vx', 'kicker_vy', 'kicker_stamina', 'kicker_effort',
    'kicker_dist_to_goal', 'kicker_dist_to_own_goal',
    'receiver_side', 'receiver_unum', 'receiver_x', 'receiver_y',
    'receiver_vx', 'receiver_vy', 'receiver_stamina',
    'ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_speed',
    'pass_distance', 'pass_angle', 'pass_direction_x', 'pass_direction_y',
    'forward_pass',
    'nearest_opp_dist', 'nearest_opp_x', 'nearest_opp_y',
    'opp_in_pass_lane', 'opp_min_dist_to_pass_line',
    'nearby_teammate_count', 'nearest_teammate_support_dist',
    'opp_density_10m',
]


def pass_event_to_row(evt: PassEvent) -> dict:
    """Convert PassEvent to a flat dict for CSV."""
    row = {}
    for name in FIELD_NAMES:
        val = getattr(evt, name, '')
        if isinstance(val, list):
            val = ';'.join(f'{v:.2f}' for v in val)
        row[name] = val
    return row


def write_csv(events: List[PassEvent], output_path: str):
    """Write pass events to CSV."""
    if not events:
        print(f"  No pass events to write.")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for evt in events:
            writer.writerow(pass_event_to_row(evt))

    success = sum(1 for e in events if e.label == 1)
    intercepted = sum(1 for e in events if e.label == 2)
    print(f"  Wrote {len(events)} pass events to {output_path}")
    print(f"    Successful: {success}, Intercepted: {intercepted}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_rcg_file(filepath: str, output_dir: str) -> List[PassEvent]:
    """Process a single RCG file and return detected pass events."""
    print(f"\nProcessing: {os.path.basename(filepath)}")

    parser = RCGParser()

    # Also parse playmode lines
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Build playmode map
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('(playmode'):
            t, pm = parse_playmode_line(stripped)
            if t >= 0:
                parser.playmode_map[t] = pm

    # Parse show frames
    frames = parser.parse(filepath)
    if not frames:
        print(f"  No frames found.")
        return []

    # Inject playmode into frames
    for frame in frames:
        if frame.time in parser.playmode_map:
            frame.playmode = parser.playmode_map[frame.time]

    print(f"  Parsed {len(frames)} frames")

    # Detect passes
    detector = PassDetector(parser.server_params)
    events = detector.detect_passes(frames, filepath)

    success = sum(1 for e in events if e.label == 1)
    intercepted = sum(1 for e in events if e.label == 2)
    print(f"  Detected {len(events)} passes (success={success}, intercepted={intercepted})")

    return events


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python rcg_pass_extractor.py <rcg_file_or_dir> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './pass_training_data'

    os.makedirs(output_dir, exist_ok=True)

    # Collect RCG files
    if os.path.isfile(input_path):
        rcg_files = [input_path]
    elif os.path.isdir(input_path):
        rcg_files = sorted(glob.glob(os.path.join(input_path, '*.rcg')))
        rcg_files += sorted(glob.glob(os.path.join(input_path, '*.rcg.gz')))
        # 递归搜索子目录（并行batch每个port一个子目录）
        rcg_files += sorted(glob.glob(os.path.join(input_path, '**/*.rcg'), recursive=True))
        rcg_files += sorted(glob.glob(os.path.join(input_path, '**/*.rcg.gz'), recursive=True))
        rcg_files = sorted(set(rcg_files))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not rcg_files:
        print(f"No RCG files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(rcg_files)} RCG file(s)")

    all_events: List[PassEvent] = []

    for filepath in rcg_files:
        events = process_rcg_file(filepath, output_dir)
        all_events.extend(events)

    # Write aggregated CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(output_dir, f'pass_features_{timestamp}.csv')
    write_csv(all_events, output_csv)

    # Also per-file
    if len(rcg_files) > 1:
        for filepath in rcg_files:
            file_events = [e for e in all_events if e.match_file == os.path.basename(filepath)]
            if file_events:
                out_path = os.path.join(output_dir,
                    os.path.basename(filepath).replace('.rcg.gz', '').replace('.rcg', '') + '_passes.csv')
                write_csv(file_events, out_path)

    # Summary
    total = len(all_events)
    success = sum(1 for e in all_events if e.label == 1)
    intercepted = sum(1 for e in all_events if e.label == 2)
    print(f"\n{'='*60}")
    print(f"Total: {total} passes from {len(rcg_files)} match(es)")
    print(f"  Successful: {success} ({100*success/max(total,1):.1f}%)")
    print(f"  Intercepted: {intercepted} ({100*intercepted/max(total,1):.1f}%)")
    print(f"Output: {output_csv}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
 