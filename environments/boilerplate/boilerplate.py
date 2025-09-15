import asyncio
import atexit
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Dataclasses & Functions ---

@dataclass
class Pod:
    """Represents a single provisioned pod."""
    id: str
    name: str
    ssh_details: dict = field(default_factory=dict)
    is_primary: bool = False

async def run_async_command_streamed(command: str, timeout: int = 1800, stream_output: bool = True, log_error: bool = True) -> Tuple[str, str]:
    """Runs a shell command asynchronously, streams its output, and returns the full output."""
    if stream_output:
        logger.info(f"Executing command: {command}")
    else:
        # Log silently but still indicate that a command is being run
        logger.debug(f"Executing command silently: {command}")

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    all_stdout = []
    all_stderr = []

    async def read_stream(stream, log_prefix, output_list):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded_line = line.decode().strip()
            if decoded_line:
                if stream_output:
                    logger.info(f"[{log_prefix}] {decoded_line}")
                output_list.append(decoded_line)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                read_stream(proc.stdout, "STDOUT", all_stdout),
                read_stream(proc.stderr, "STDERR", all_stderr),
                proc.wait()
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    if proc.returncode != 0:
        if log_error:
            logger.error(f"Command failed with exit code {proc.returncode}: {command}")
            if all_stdout:
                logger.error(f"--- FAILED COMMAND STDOUT ---\n" + "\n".join(all_stdout))
            if all_stderr:
                logger.error(f"--- FAILED COMMAND STDERR ---\n" + "\n".join(all_stderr))
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")

    return "\n".join(all_stdout), "\n".join(all_stderr)

# --- Pod and Job Management ---

class PodPoolManager:
    """Manages a pool of available pods for concurrent jobs."""
    def __init__(self):
        self._queue = asyncio.Queue()
        self._all_pods = []
        self._provisioning_task: Optional[asyncio.Task] = None

    def add_pod(self, pod: Pod):
        self._all_pods.append(pod)
        self._queue.put_nowait(pod)
        logger.info(f"Pod {pod.name} is ready and added to the pool. {self._queue.qsize()} available.")

    async def acquire(self) -> Pod:
        """Acquires a pod from the pool, waiting if necessary.
        
        This method is now aware of the background provisioning task. It will
        return a pod as soon as one is available, but if the provisioning task
        fails, it will raise the exception from that task to unblock waiters.
        """
        logger.info(f"Requesting a pod. {self._queue.qsize()} available...")

        # If provisioning is in progress, we need to handle the race condition.
        if self._provisioning_task and not self._provisioning_task.done():
            logger.info("Provisioning is still in progress, waiting for the next available pod...")
            get_pod_task = asyncio.create_task(self._queue.get())
            
            done, pending = await asyncio.wait(
                [get_pod_task, self._provisioning_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if get_pod_task in done:
                # Happy path: a pod became available.
                for task in pending:
                    task.cancel()  # Cancel the wait on the provisioning task.
                pod = get_pod_task.result()
                logger.info(f"Acquired pod: {pod.name} ({pod.id})")
                return pod
            
            if self._provisioning_task in done:
                # The provisioning task finished before we got a pod.
                get_pod_task.cancel()  # Clean up the pending queue getter.
                
                # Re-raise any exception from the provisioning task.
                if self._provisioning_task.exception():
                    raise self._provisioning_task.exception()
                
                # If we are here, provisioning finished without error, but we lost the race.
                # The queue might be empty now, but other rollouts will eventually release pods.
                # So we just fall through to the simple `await self._queue.get()` below.
                logger.info("Provisioning finished. Waiting for a pod to be released.")

        # This is the simple case: provisioning is done (or was never started).
        # We just wait for a pod to appear in the queue.
        pod = await self._queue.get()
        logger.info(f"Acquired pod: {pod.name} ({pod.id})")
        return pod

    def release(self, pod: Pod):
        self._queue.put_nowait(pod)
        logger.info(f"Released pod: {pod.name}. {self._queue.qsize()} now available.")

    def get_all_pods(self) -> List[Pod]:
        return list(self._all_pods)

    def set_provisioning_task(self, task: asyncio.Task):
        self._provisioning_task = task

# --- Reward Function ---
def get_reward_from_state(state: dict, **kwargs) -> float:
    """Extracts the pre-calculated reward from the state dictionary."""
    return state.get("reward", 0.0)

# --- Verifiers Environment ---

class NanoGPTSpeedrunEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        num_pods: int,
        min_pods: int = 2,
        gpu_type: str = "H100_80GB",
        socket_type: Optional[str] = None,
        on_demand: Union[bool, str] = False,
        debug: bool = True,
        **kwargs
    ):
        if num_pods < 1:
            raise ValueError("num_pods must be at least 1.")
        if min_pods > num_pods:
            raise ValueError("min_pods cannot be greater than num_pods.")
        if isinstance(on_demand, str) and on_demand.lower() != "all":
            raise ValueError("If 'on_demand' is a string, it must be 'all'.")

        dummy_dataset = Dataset.from_dict({"prompt": ["start_run"], "answer": [""]})
        
        super().__init__(dataset=dummy_dataset, **kwargs)
        self.rubric = vf.Rubric(funcs=[get_reward_from_state])
        
        self.num_pods = num_pods
        self.min_pods = min_pods
        self.gpu_type = gpu_type
        self.socket_type = socket_type
        self.on_demand = on_demand.lower() if isinstance(on_demand, str) else on_demand
        self.debug = debug
        
        self.pod_pool = PodPoolManager()
        self._provisioning_started = False
        
        atexit.register(self._cleanup_pods)
        logger.info("Environment initialized. Pod provisioning will start on the first rollout.")

    def _cleanup_pods(self):
        logger.info("Cleanup initiated: Terminating all provisioned pods...")
        all_pods = self.pod_pool.get_all_pods()
        if not all_pods:
            logger.info("No pods to terminate.")
            return
        # Using asyncio.run() is generally safe in atexit for cleanup.
        # It creates a new event loop to run the async cleanup tasks.
        try:
            asyncio.run(self._terminate_pods_async(all_pods))
        except RuntimeError as e:
            # This can happen if an event loop is already running on the thread.
            logger.warning(f"Could not create new event loop for cleanup: {e}. Attempting to use existing one.")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._terminate_pods_async(all_pods))
            except RuntimeError as e2:
                logger.error(f"Could not find a running event loop for async cleanup: {e2}")


    async def _ensure_provisioning_started(self):
        if not self._provisioning_started:
            logger.info("First rollout call: starting background pod provisioning...")
            self._provisioning_started = True
            provisioning_task = asyncio.create_task(self._provision_pods(self.num_pods, self.min_pods))
            self.pod_pool.set_provisioning_task(provisioning_task)

    async def _terminate_one_pod_with_retry(self, pod: Pod, max_retries: int = 3, delay_seconds: int = 10):
        """Tries to terminate a single pod with retries."""
        cmd = f"prime pods terminate {pod.id} --yes"
        for attempt in range(max_retries):
            try:
                await run_async_command_streamed(cmd, timeout=180, stream_output=False)
                logger.info(f"Successfully terminated pod: {pod.name}")
                return
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} to terminate pod {pod.name} failed. "
                    f"Retrying in {delay_seconds}s... Error: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay_seconds)
        
        logger.error(f"Failed to terminate pod {pod.name} after {max_retries} attempts.")


    async def _terminate_pods_async(self, pods: List[Pod]):
        tasks = []
        for pod in pods:
            tasks.append(self._terminate_one_pod_with_retry(pod))
            logger.info(f"Queueing termination for pod: {pod.name} ({pod.id})")
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _provision_one_pod(self, pod_name: str, offset: int, is_primary: bool, use_on_demand: bool) -> Pod:
        script_dir = Path(__file__).parent
        provision_script_path = script_dir / "provision_rig.sh"
        
        provision_cmd = (
            f"{provision_script_path} --pod-name {pod_name} --offset {offset} {self.gpu_type}"
            f"{' --on-demand' if use_on_demand else ''}"
        )
        if self.socket_type:
            provision_cmd += f" --socket {self.socket_type}"

        stdout, _ = await run_async_command_streamed(provision_cmd, timeout=1800, stream_output=True)
        
        pod_id = ""
        for line in stdout.splitlines():
            if line.startswith("FINAL_POD_ID:"):
                pod_id = line.split(":", 1)[1].strip()
                break
        
        if not pod_id:
            raise RuntimeError(f"Could not extract FINAL_POD_ID for {pod_name} from provision output.")
        
        status_cmd = f"prime pods status {pod_id} --output json"
        stdout, _ = await run_async_command_streamed(status_cmd, stream_output=False)
        status_json = json.loads(stdout)
        ssh_str = status_json.get("ssh")
        if not ssh_str:
            raise RuntimeError(f"Could not get SSH details for pod {pod_id}")
            
        ssh_parts = ssh_str.split()
        user_host = ssh_parts[0]
        port = ssh_parts[2]
        
        config_cmd = "prime config view"
        stdout, _ = await run_async_command_streamed(config_cmd, stream_output=False)
        ssh_key_path = ""
        for line in stdout.splitlines():
            if "SSH Key Path" in line:
                ssh_key_path = line.split('│')[2].strip()

        if not ssh_key_path:
            raise RuntimeError("Could not find 'SSH Key Path' in 'prime config view' output.")

        return Pod(
            id=pod_id, 
            name=pod_name, 
            ssh_details={
                "user_host": user_host,
                "port": port,
                "key_path": ssh_key_path
            },
            is_primary=is_primary
        )

    async def _get_availability_data(self) -> List[dict]:
        """Fetches and returns the raw availability data."""
        gpu_count = 8 # Hardcoded for NanoGPT speedrun
        cmd = f"prime availability list --gpu-type {self.gpu_type} --gpu-count {gpu_count} --output json"
        try:
            stdout, _ = await run_async_command_streamed(cmd, stream_output=False)
            availability_data = json.loads(stdout)
            logger.info(f"Full availability data from prime-cli:\n{json.dumps(availability_data, indent=2)}")
            return availability_data.get("gpu_resources", [])
        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"Could not perform availability check due to an error: {e}")
            return []

    async def _check_availability(self, num_required: int):
        """Performs a pre-flight check to see if enough instances are available."""
        logger.info(f"Performing pre-flight check for at least {num_required} available pods...")
        
        resources = await self._get_availability_data()
        if not resources:
            logger.warning("Availability data is empty. Proceeding with provisioning attempt...")
            return

        filtered_resources = []
        for res in resources:
            # Filter by socket type if specified
            if self.socket_type and res.get("socket") != self.socket_type:
                continue
            
            # Filter by on_demand type
            is_spot = res.get("is_spot", False)
            if self.on_demand == "all":
                pass # Include all types
            elif self.on_demand is True and is_spot:
                continue # Require on-demand, but this is spot
            elif self.on_demand is False and not is_spot:
                continue # Require spot, but this is on-demand
                
            filtered_resources.append(res)

        available_count = len(filtered_resources)
        logger.info(f"Found {available_count} matching instances available.")

        if available_count < num_required:
            raise RuntimeError(
                f"Pre-flight check failed: Not enough pods available. "
                f"Required: {num_required}, Found: {available_count}."
            )

    async def _provision_pods(self, num_pods: int, min_pods: int):
        await self._check_availability(min_pods)

        tasks = []
        pod_names = [f"nanogpt-speedrun-worker-{uuid.uuid4().hex[:8]}" for _ in range(num_pods)]

        if self.on_demand != "all":
            # Simple case: all pods are either spot or on-demand
            use_on_demand = bool(self.on_demand)
            logger.info(f"Provisioning {num_pods} pods of the same type (on-demand: {use_on_demand}).")
            for i, pod_name in enumerate(pod_names):
                is_primary = (i == 0)
                task = asyncio.create_task(self._provision_one_pod(pod_name, i, is_primary, use_on_demand))
                tasks.append(task)
        else:
            # "all" case: prioritize spot, fallback to on-demand
            logger.info("Provisioning with 'all' strategy: prioritizing spot, falling back to on-demand.")
            all_resources = await self._get_availability_data()
            
            # Filter resources by socket type
            filtered_resources = [
                r for r in all_resources 
                if not self.socket_type or r.get("socket") == self.socket_type
            ]

            spot_resources = [r for r in filtered_resources if r.get("is_spot", False)]
            ondemand_resources = [r for r in filtered_resources if not r.get("is_spot", False)]
            
            logger.info(f"Found {len(spot_resources)} available spot and {len(ondemand_resources)} available on-demand instances.")

            pod_idx = 0
            # --- Attempt to provision with SPOT instances first ---
            for offset in range(len(spot_resources)):
                if pod_idx >= num_pods: break
                pod_name = pod_names[pod_idx]
                is_primary = (pod_idx == 0)
                logger.info(f"Queueing SPOT provisioning for {pod_name} (offset {offset}).")
                task = asyncio.create_task(self._provision_one_pod(pod_name, offset, is_primary, use_on_demand=False))
                tasks.append(task)
                pod_idx += 1
            
            # --- Fallback to ON-DEMAND for the remainder ---
            if pod_idx < num_pods:
                logger.info(f"Spot capacity met. Attempting to provision remaining {num_pods - pod_idx} pods with on-demand instances.")
                for offset in range(len(ondemand_resources)):
                    if pod_idx >= num_pods: break
                    pod_name = pod_names[pod_idx]
                    is_primary = (pod_idx == 0) # Should always be false here if at least one spot was found
                    logger.info(f"Queueing ON-DEMAND provisioning for {pod_name} (offset {offset}).")
                    task = asyncio.create_task(self._provision_one_pod(pod_name, offset, is_primary, use_on_demand=True))
                    tasks.append(task)
                    pod_idx += 1

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_pods = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Pod):
                self.pod_pool.add_pod(result)
                successful_pods.append(result)
            else:
                failed_count += 1
                # The pod name might not be in the original list if provisioning failed to meet capacity
                pod_name_or_index = pod_names[i] if i < len(pod_names) else f"task index {i}"
                logger.error(f"Provisioning failed for {pod_name_or_index}: {result}")

        num_successful = len(successful_pods)
        logger.info(f"Provisioning attempt complete. Successful: {num_successful}, Failed: {failed_count}")

        if num_successful < min_pods:
            logger.error(
                f"Final count of successful pods ({num_successful}) is less than min_pods ({min_pods}). "
                f"Terminating all successful pods."
            )
            if successful_pods:
                await self._terminate_pods_async(successful_pods)
            raise RuntimeError("Pod provisioning failed to meet the minimum requirement.")
    
    async def _stream_log_from_pod(self, pod: Pod, remote_log_path: str):
        ssh_base = (
            f"ssh -i {pod.ssh_details['key_path']} -p {pod.ssh_details['port']} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        )
        check_file_cmd = f"{ssh_base} {pod.ssh_details['user_host']} 'while [ ! -f {remote_log_path} ]; do sleep 1; done'"
        await run_async_command_streamed(check_file_cmd, timeout=60, stream_output=False)
        
        stream_cmd = f"{ssh_base} {pod.ssh_details['user_host']} 'tail -f {remote_log_path}'"
        await run_async_command_streamed(stream_cmd)

    async def _run_job_on_pod(self, pod: Pod, script_path: str) -> dict:
        job_uuid = uuid.uuid4().hex[:8]
        remote_script_path = f"/tmp/train_job_{job_uuid}.py"
        remote_result_path = "/root/job_result.json"
        remote_failure_flag = "/root/job_failed.flag"
        remote_job_output_log = "/root/job_output.log"

        ssh_base = (
            f"ssh -i {pod.ssh_details['key_path']} -p {pod.ssh_details['port']} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        )
        scp_base = (
            f"scp -i {pod.ssh_details['key_path']} -P {pod.ssh_details['port']} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        )
        
        script_dir = Path(__file__).parent
        run_job_script_path = script_dir / "run_job.sh"
        remote_run_job_path = "/root/run_job.sh"
        
        upload_run_job_cmd = f"{scp_base} {run_job_script_path} {pod.ssh_details['user_host']}:{remote_run_job_path}"
        await run_async_command_streamed(upload_run_job_cmd, stream_output=False)
        
        chmod_cmd = f"{ssh_base} {pod.ssh_details['user_host']} 'chmod +x {remote_run_job_path}'"
        await run_async_command_streamed(chmod_cmd, stream_output=False)
        
        upload_cmd = f"{scp_base} {script_path} {pod.ssh_details['user_host']}:{remote_script_path}"
        await run_async_command_streamed(upload_cmd, stream_output=False)
        
        start_cmd = f"{ssh_base} {pod.ssh_details['user_host']} 'nohup /root/run_job.sh {remote_script_path} > {remote_job_output_log} 2>&1 &'"
        await run_async_command_streamed(start_cmd, stream_output=False)
        
        log_streaming_task = None
        should_stream = self.debug and pod.is_primary
        if should_stream:
            log_streaming_task = asyncio.create_task(self._stream_log_from_pod(pod, remote_job_output_log))

        check_cmd = f"{ssh_base} {pod.ssh_details['user_host']} 'test -f {remote_result_path} || test -f {remote_failure_flag}'"
        try:
            while True:
                try:
                    await run_async_command_streamed(check_cmd, timeout=45, stream_output=False, log_error=False)
                    break
                except RuntimeError:
                    if not should_stream:
                        logger.info(f"Job on non-primary pod {pod.name} still running, checking again in 30s...")
                    await asyncio.sleep(30)
                except TimeoutError:
                    logger.warning(f"SSH check timed out for {pod.name}. Retrying.")
                    await asyncio.sleep(5)
        finally:
            if log_streaming_task:
                log_streaming_task.cancel()

        try:
            # Check if the failure flag exists. If this command succeeds, the file is there.
            await run_async_command_streamed(f"{ssh_base} {pod.ssh_details['user_host']} 'test -f {remote_failure_flag}'", stream_output=False, log_error=False)
            logger.error(f"Job failed on pod {pod.name}. Failure flag found. Check logs for details.")
            return {"status": "failed", "reason": "failure_flag_found"}
        except RuntimeError:
            # This is the expected path: failure flag was not found. Now check for result file.
            pass

        try:
            local_result_path = f"/tmp/result_{job_uuid}.json"
            download_cmd = f"{scp_base} {pod.ssh_details['user_host']}:{remote_result_path} {local_result_path}"
            await run_async_command_streamed(download_cmd, stream_output=False)
            with open(local_result_path) as f:
                result = json.load(f)
            result["status"] = "success"
            Path(local_result_path).unlink()
            return result
        except (RuntimeError, FileNotFoundError):
            logger.error(f"Job on pod {pod.name} appears complete, but the result file could not be downloaded.")
            logger.error("This usually means the training script finished without crashing but failed to produce a result JSON.")
            return {"status": "failed", "reason": "result_file_not_found"}

    async def rollout(self, client, model, prompt, answer, task, info, sampling_args, **kwargs) -> tuple[Messages, State]:
        await self._ensure_provisioning_started()

        pod = await self.pod_pool.acquire()
        try:
            script_path = Path(__file__).parent / "train_gpt.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Training script not found at {script_path}")

            result = await self._run_job_on_pod(pod, str(script_path))

            reward = 0.0
            if result["status"] == "success":
                time_ms = float(result.get("training_time_ms", float('inf')))
                reward = 1000 / (time_ms + 1e-6) 
            
            completion_message = {
                "role": "assistant",
                "content": f"Training run completed on pod {pod.name}. Status: {result['status']}. Reward: {reward:.4f}"
            }
            
            state = {"reward": reward, "result": result}
            return [completion_message], state

        finally:
            self.pod_pool.release(pod)

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        return True

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        return [], state


def load_environment(**kwargs) -> vf.Environment:
    if 'num_pods' not in kwargs:
        raise ValueError("The 'num_pods' argument is mandatory for this environment.")
    
    return NanoGPTSpeedrunEnv(**kwargs)