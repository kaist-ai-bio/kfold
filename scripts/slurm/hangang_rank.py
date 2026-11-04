"""Discover the actual Slurm rank-zero host before distributed initialization."""

import argparse
import json
import os
import socket
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path


def configure_rendezvous(root, env, hostname, timeout=180):
    rank = int(env["SLURM_PROCID"])
    world_size = int(env["SLURM_NTASKS"])
    local_rank = int(env["SLURM_LOCALID"])
    step = env.get("SLURM_STEP_ID", env.get("SLURM_STEPID", ""))
    if not step.isdigit():
        raise ValueError(f"Expected a numeric Slurm step ID, got {step!r}")
    if not 0 <= rank < world_size or not 0 <= local_rank < 4:
        raise ValueError("Invalid Slurm rank or local GPU index")
    directory = Path(root) / f"rendezvous-step-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = directory / "master.json"
    job = env["SLURM_JOB_ID"]
    port = 20000 + int(job) % 20000 + int(step) % 1000
    if rank == 0:
        payload = dict(job_id=job, step_id=step, host=hostname, port=port)
        temporary = endpoint.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(endpoint)
    deadline = time.monotonic() + timeout
    while not endpoint.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Slurm rank zero did not publish its rendezvous address")
        time.sleep(0.05)
    payload = json.loads(endpoint.read_text())
    if payload["job_id"] != job or payload["step_id"] != step:
        raise ValueError("Stale rendezvous endpoint from another job or step")
    env["MASTER_ADDR"] = payload["host"]
    env["MASTER_PORT"] = str(payload["port"])
    # This launcher uses plain srun, with no torchrun agent/store to reuse.
    env["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(key, None)
    context = dict(
        job_id=job,
        step_id=step,
        rank=rank,
        local_rank=local_rank,
        slurm_node_id=env["SLURM_NODEID"],
        hostname=hostname,
        master_address=payload["host"],
        master_port=payload["port"],
        world_size=world_size,
    )
    (directory / f"rank-{rank}.json").write_text(json.dumps(context) + "\n")
    return context, directory


def preflight(context, directory, timeout=180):
    from multiprocessing.connection import Listener

    # Exercise the same AF_UNIX listener used by DataLoader tensor sharing.
    with Listener() as listener:
        print("IPC_PREFLIGHT_PASSED", listener.address, flush=True)

    import torch
    import torch.distributed as dist

    local_rank, rank, size = (
        context["local_rank"],
        context["rank"],
        context["world_size"],
    )
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expected all four local GPUs visible to each Slurm rank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=timeout),
    )
    value = torch.tensor([rank, 1], device=f"cuda:{local_rank}", dtype=torch.float64)
    dist.all_reduce(value)
    assert value.tolist() == [size * (size - 1) // 2, size]
    # Small, bounded probe at approximately the trainable adapter payload size.
    # This measures the allocated ranks, rather than inferring transport speed
    # from a successful sixteen-byte connectivity collective.
    payload = torch.ones(1024 * 1024, device=value.device, dtype=torch.float32)
    for _ in range(3):
        payload.fill_(1)
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    dist.barrier()
    started = time.monotonic()
    iterations = 5
    for _ in range(iterations):
        payload.fill_(1)
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    seconds = (time.monotonic() - started) / iterations
    assert payload[0].item() == size
    if rank == 0:
        records = [
            json.loads((directory / f"rank-{i}.json").read_text()) for i in range(size)
        ]
        hosts = Counter(row["hostname"] for row in records)
        assert len(hosts) == int(os.environ["SLURM_JOB_NUM_NODES"])
        assert all(count == 4 for count in hosts.values())
        assert len({(r["hostname"], r["local_rank"]) for r in records}) == size
        receipt = dict(
            status="passed",
            ranks=size,
            nodes=len(hosts),
            master=context["master_address"],
            all_reduce=value.tolist(),
            collective_probe=dict(
                bytes=payload.numel() * payload.element_size(),
                iterations=iterations,
                mean_seconds=seconds,
                payload_gib_per_second=payload.numel()
                * payload.element_size()
                / seconds
                / 2**30,
            ),
            transport_environment={
                key: os.environ.get(key)
                for key in (
                    "NCCL_NET",
                    "FI_PROVIDER",
                    "NCCL_IB_DISABLE",
                    "NCCL_PROTO",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                )
            },
        )
        (directory / "preflight-passed.json").write_text(json.dumps(receipt) + "\n")
        print("DISTRIBUTED_PREFLIGHT_PASSED", json.dumps(receipt), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    context, directory = configure_rendezvous(
        os.environ["RUN_ROOT"], os.environ, socket.gethostname()
    )
    print("SLURM_RANK_CONTEXT", json.dumps(context), flush=True)
    if args.preflight:
        preflight(context, directory)
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Missing training command")
        os.execv(sys.executable, [sys.executable, "-B", *command])


if __name__ == "__main__":
    main()
