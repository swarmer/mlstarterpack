import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import traceback

from PIL import Image

from .images import find_images, get_resized_dims


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand',
                                       required=True)

    convert_parser = subparsers.add_parser(
        'convert',
        help=(
            'Convert image files to .bmp so no costly jpeg (or other) '
            'decoding is required'
        ),
    )
    convert_parser.add_argument('sourcedir', type=str, nargs=1)
    convert_parser.add_argument('targetdir', type=str, nargs=1)
    convert_parser.add_argument(
        '--target-smaller-size', type=int, required=False, default=224,
    )
    convert_parser.set_defaults(func=_convert)

    return parser.parse_args()


@dataclass
class ConvertJob:
    source_path: Path
    target_path: Path
    target_smaller_size: int


def _process_image(job: ConvertJob):
    try:
        image: Image.Image = Image.open(job.source_path)

        new_height, new_width = get_resized_dims(
            image.height, image.width,
            job.target_smaller_size,
        )
        image = image.resize(
            (new_width, new_height),
            Image.LANCZOS,
        )

        image.save(job.target_path)
        print(f'Processed {job.source_path} -> {job.target_path}')
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()


def _convert(args):
    sourcedir = Path(args.sourcedir)
    targetdir = Path(args.targetdir)

    os.makedirs(targetdir, exist_ok=True)

    jobs = []
    for source_path in find_images(args.sourcedir):
        source_name = source_path.relative_to(sourcedir)

        target_name = Path(str(source_name).replace(source_name.suffix, '.bmp'))
        target_path = targetdir / target_name

        if target_path.exists() and target_path.stat().st_size > 0:
            continue

        jobs.append(ConvertJob(
            source_path, target_path,
            args.target_smaller_size,
        ))

    print(f'{len(jobs)} files to process')

    executor = ProcessPoolExecutor()
    iterator = executor.map(_process_image, jobs)
    for _ in iterator:
        pass
    print('Done')


def main():
    args = parse_arguments()
    args.func(args)


if __name__ == '__main__':
    main()
