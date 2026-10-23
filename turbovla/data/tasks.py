"""Read official LeRobot v3 task instructions with a legacy JSONL fallback."""
import json
from pathlib import Path


def load_tasks(root):
    root = Path(root)
    parquet = root / "meta/tasks.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        rows = pq.read_table(parquet).to_pylist()
    else:
        rows = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    tasks = {}
    for row in rows:
        task_id = int(row["task_index"])
        instruction = row.get("task", row.get("__index_level_0__"))
        if not isinstance(instruction, str) or not instruction.strip() or task_id in tasks:
            raise ValueError(f"Invalid or duplicate task instruction: {task_id}")
        tasks[task_id] = instruction
    if not tasks:
        raise ValueError("Empty task metadata")
    return tasks
