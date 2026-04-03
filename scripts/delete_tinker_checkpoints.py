import asyncio

from tinker import ServiceClient

client = ServiceClient().create_rest_client()


async def main():
    for offset in range(0, 1000, 20):
        training_runs = await client.list_training_runs_async(offset=offset, limit=20)
        for run in training_runs.training_runs:
            checkpoints = await client.list_checkpoints_async(run.training_run_id)
            for checkpoint in checkpoints.checkpoints:
                print(f"Deleting checkpoint {checkpoint.checkpoint_id} for training run {run.training_run_id}")
                await client.delete_checkpoint_async(
                    training_run_id=run.training_run_id, checkpoint_id=checkpoint.checkpoint_id
                )


if __name__ == "__main__":
    asyncio.run(main())
