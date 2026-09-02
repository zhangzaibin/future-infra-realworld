# 260806
102条episode，学长检查可行，继续采集
# 260807
’r'只能在录制中舍弃当前episode，添加脚本用于录制后删除末尾N条episode

## 删除末尾episode
用 bash wrapper（自动找 conda 环境）
bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 1

或直接用 Python
python scripts/delete_last_episodes.py --dataset ./datasets/pick_and_place --last 1

加 --yes 跳过确认
bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 1 --yes

录制完一条episode后出现报错：
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/configs/parser.py", line 233, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/scripts/lerobot_record.py", line 547, in record
    dataset.save_episode()
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/datasets/lerobot_dataset.py", line 1238, in save_episode
    self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/datasets/lerobot_dataset.py", line 410, in save_episode
    self._save_episode_metadata(episode_dict)
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/datasets/lerobot_dataset.py", line 353, in _save_episode_metadata
    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
  File "/home/ubuntu/projects/Atlas_lerobot/src/lerobot/datasets/utils.py", line 103, in update_chunk_file_indices
    file_idx += 1
TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'
FATAL: exception not rethrown
已中止 (核心已转储)
