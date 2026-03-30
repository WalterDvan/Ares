Ares Training Data Collector v6
每场比赛有独立match_N目录，训练数据和RCG按场隔离。

目录结构:
  logs/match_001/mad_training_data/*.csv
  logs/match_001/pass_training_data/*.csv
  logs/match_001/*.rcg
  logs/match_002/...

用法:
  bash batch_train.sh -d ~/helios-base/build/bin -p 4 -m 800 -l ~/match_logs
  python3 rcg_pass_extractor.py ~/match_logs ~/pass_data
