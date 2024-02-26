import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from .base_sample import TSampleGenerator, TSampleParser

worker_sample_generator: TSampleGenerator | None = None


def _init_worker(generator_type: type[TSampleGenerator], args: TSampleParser) -> None:
    global worker_sample_generator

    worker_sample_generator = generator_type(args)


def _generate_sample(index: int, save_dir: Path) -> None:
    global worker_sample_generator
    assert worker_sample_generator is not None
    sample = worker_sample_generator.generate_sample()
    sample.save(save_dir / f"{index}.json")
    sample.generate_image().save(save_dir / f"{index}.png")


def bulk(generator_type: type[TSampleGenerator], args: TSampleParser) -> None:
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.save(str((args.save_dir / "parameters.json").resolve()), False)

    print(f"Saving new samples and images to {args.save_dir}")

    # Generate samples in parallel on every available CPU core
    with mp.Pool(
        mp.cpu_count(), initializer=_init_worker, initargs=(generator_type, args)
    ) as pool:
        jobs = [
            pool.apply_async(_generate_sample, (i, args.save_dir))
            for i in range(args.num_samples)
        ]

        for job in tqdm(jobs, desc="Generating samples", ncols=80):
            job.get()


def single(generator_type: type[TSampleGenerator], args: TSampleParser) -> None:
    sample = generator_type(args).generate_sample()

    print(f"Saving new sample and image to {args.save_dir}")
    sample.save(args.save_dir / "sample.json")
    sample.generate_image().save(args.save_dir / "sample.png")
